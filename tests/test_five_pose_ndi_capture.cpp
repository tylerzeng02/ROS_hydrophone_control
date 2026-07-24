#include <array>
#include <chrono>
#include <cmath>
#include <conio.h>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "dynamixel_motor.h"
#include "robot_calibration.h"
#include "ndicapi.h"

namespace {

// Thrown when the user presses the skip key while waiting on NDI tracking;
// caught in main()'s per-pose loop so one bad pose doesn't abort the run.
struct PoseSkippedByUser {};

enum class NdiToolStatus {
    Detected,
    Missing,
    OutOfVolume,
    Disabled,
    LowQuality
};

const char* toolStatusLabel(NdiToolStatus status) {
    switch (status) {
        case NdiToolStatus::Detected:    return "DETECTED";
        case NdiToolStatus::Missing:     return "MISSING";
        case NdiToolStatus::OutOfVolume: return "OUT_OF_VOLUME";
        case NdiToolStatus::Disabled:    return "DISABLED";
        case NdiToolStatus::LowQuality:  return "LOW_QUALITY";
    }
    return "UNKNOWN";
}

// Prints "<toolName> tool: <STATUS>" only the first time it's called and
// whenever the status differs from the last printed value, so a steady
// status doesn't spam the console.
void printToolStatusIfChanged(
    const char* toolName,
    NdiToolStatus current,
    NdiToolStatus& lastPrinted,
    bool& everPrinted
) {
    if (everPrinted && current == lastPrinted) {
        return;
    }
    std::cout << toolName << " tool: " << toolStatusLabel(current) << '\n';
    lastPrinted = current;
    everPrinted = true;
}

// Non-blocking check for the pause ('p') / skip ('s') hotkeys. Call once per
// polling iteration. Throws PoseSkippedByUser if skip is pressed (either
// directly or while paused). Returns how long this call spent paused, so
// callers can shift their own elapsed-time budgets forward by that amount.
std::chrono::milliseconds handleUserControls() {
    if (!_kbhit()) {
        return std::chrono::milliseconds(0);
    }

    const int key = _getch();

    if (key == 's' || key == 'S') {
        throw PoseSkippedByUser{};
    }

    if (key == 'p' || key == 'P') {
        const auto pauseStart = std::chrono::steady_clock::now();
        std::cout
            << "\nProcess paused. Press 'p' to resume, or 's' to skip "
            << "this pose.\n";

        while (true) {
            if (_kbhit()) {
                const int resumeKey = _getch();
                if (resumeKey == 'p' || resumeKey == 'P') {
                    std::cout << "Resuming detection...\n";
                    break;
                }
                if (resumeKey == 's' || resumeKey == 'S') {
                    throw PoseSkippedByUser{};
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - pauseStart
        );
    }

    return std::chrono::milliseconds(0);
}

constexpr std::size_t JOINT_COUNT = 7;
constexpr std::size_t POSE_COUNT = 100;

constexpr const char* CYTON_DEVICE = "COM4";
constexpr int CYTON_BAUD_RATE = 1000000;
constexpr float CYTON_PROTOCOL_VERSION = 1.0F;

constexpr const char* NDI_DEVICE = "COM3";

constexpr const char* MOVING_TOOL_ROM =
    R"(C:\Users\ConformalUser\Desktop\Spectra\8700339- Polaris Passive 4-Marker Rigid Body 2(1).rom)";

constexpr const char* FIXED_TOOL_ROM =
    R"(C:\Users\ConformalUser\Desktop\Spectra\8700449- Polaris Passive 4-Marker Rigid Body 3(1).rom)";

constexpr const char* OUTPUT_CSV = "five_pose_ndi_capture.csv";

constexpr uint16_t MOVING_SPEED = 40;
constexpr int MOTOR_TOLERANCE_TICKS = 10;
constexpr int MOVE_TIMEOUT_SECONDS = 30;
constexpr int SETTLING_TIME_MS = 750;

// If the combined position error across all joints reads exactly the same
// value for this many consecutive ~100ms checks, the arm has effectively
// stopped moving (stalled short of tolerance, e.g. friction/backlash) --
// no point waiting out the full MOVE_TIMEOUT_SECONDS in that case, so give
// it a short grace period instead.
constexpr int STALL_REPEATS_TO_DETECT = 3;
constexpr int STALL_GRACE_SECONDS = 5;

constexpr int NDI_REQUIRED_VALID_SAMPLES = 30;
constexpr int NDI_MAX_ATTEMPTS = 1000;
constexpr int NDI_SAMPLE_INTERVAL_MS = 20;
constexpr int REQUIRED_VISIBLE_MARKERS = 4;
constexpr double MAX_NDI_ERROR = 0.50;

// Generated for kinematic-calibration data collection: joints 0, 1, 3, 4, 5
// sweep broadly across their safe tick ranges (jointCalibrations
// minTick/maxTick, with a 100-tick safety margin from each extreme). Joint 2
// (shoulder_yaw) uses an extra 100-tick margin beyond that (200 total from
// each extreme) per an additional caution request. Joint 6 (wrist_roll) is
// deliberately restricted to a narrow window around its zero tick (2048),
// since large wrist_roll rotation reorients the moving marker's face away
// from the Polaris (the marker's reflective side only faces one direction)
// and the tool goes untrackable. Widen the joint-6 range only after
// empirically sweeping it (watch the "Moving tool: DETECTED/MISSING"
// status output) to find how far it can actually rotate while staying
// visible to the tracker.
const std::array<std::array<uint16_t, JOINT_COUNT>, POSE_COUNT> TARGET_POSES = {{
    {{580, 1366, 1716, 1461, 2520, 1071, 2140}},
    {{3412, 2614, 1222, 1187, 2815, 1179, 1962}},
    {{3427, 1783, 2293, 1806, 3115, 1274, 1728}},
    {{1341, 1532, 3106, 1055, 1968, 947, 2170}},
    {{600, 2322, 1138, 1838, 2667, 2183, 2089}},
    {{2316, 2670, 1147, 1724, 1368, 1779, 2018}},
    {{850, 1315, 2487, 2103, 2453, 1062, 1765}},
    {{1404, 1235, 2958, 2244, 1262, 1040, 2366}},
    {{2620, 1775, 2391, 1382, 2486, 2483, 1690}},
    {{2744, 3126, 3016, 1551, 1638, 1617, 2122}},
    {{2877, 1152, 1451, 3159, 1924, 2458, 2123}},
    {{2644, 1182, 1322, 2080, 2631, 1088, 2322}},
    {{2383, 1951, 1574, 1159, 2239, 2023, 1727}},
    {{909, 2916, 1164, 1989, 2264, 1685, 1871}},
    {{2128, 1076, 1833, 3166, 2637, 1738, 1762}},
    {{2307, 1458, 1735, 2410, 1089, 3041, 1870}},
    {{525, 2969, 1893, 941, 1290, 1841, 1934}},
    {{1320, 1271, 2850, 1564, 1651, 2049, 1894}},
    {{922, 1363, 2323, 2729, 1377, 2315, 2186}},
    {{982, 2546, 2942, 2559, 2518, 916, 2405}},
    {{2354, 2633, 1462, 1559, 2256, 1593, 2024}},
    {{714, 2133, 1426, 2456, 2810, 2170, 1830}},
    {{476, 1872, 2467, 2695, 2141, 1989, 2413}},
    {{616, 2856, 2148, 2323, 1756, 2265, 1966}},
    {{3488, 1867, 2999, 991, 1111, 1023, 2174}},
    {{1528, 2505, 1176, 1627, 1094, 3166, 1806}},
    {{1528, 2263, 1883, 2301, 1764, 1528, 2273}},
    {{1549, 2312, 2859, 2197, 1955, 917, 1954}},
    {{2001, 1194, 1702, 2774, 1964, 2328, 2134}},
    {{2472, 1374, 1249, 990, 2952, 1815, 1933}},
    {{1859, 1232, 1330, 1083, 2903, 1992, 1967}},
    {{2581, 2780, 3017, 1362, 1187, 3204, 2356}},
    {{1713, 1374, 1614, 1200, 2710, 1486, 2211}},
    {{2405, 1235, 2102, 2698, 1407, 2494, 1940}},
    {{3615, 1200, 1674, 2019, 1889, 2006, 2028}},
    {{835, 1876, 2063, 1371, 2305, 2450, 2249}},
    {{1668, 1489, 2798, 1902, 1834, 3105, 2299}},
    {{2939, 1011, 2697, 2894, 1163, 1780, 1766}},
    {{2642, 1951, 1776, 2052, 1364, 2142, 2360}},
    {{1222, 3004, 1194, 2596, 1219, 3144, 1711}},
    {{1551, 2495, 2151, 2230, 1415, 2639, 2153}},
    {{1204, 1119, 2221, 2527, 1947, 1331, 1887}},
    {{3549, 2320, 1972, 2080, 1830, 1027, 2278}},
    {{2334, 2894, 2544, 980, 1253, 1628, 1940}},
    {{3158, 2678, 2462, 1325, 2979, 2111, 1987}},
    {{3544, 1842, 1664, 1980, 1412, 1242, 1875}},
    {{2994, 3066, 1728, 2726, 1163, 2419, 1663}},
    {{1797, 1243, 1271, 2055, 2615, 3212, 2203}},
    {{1046, 2652, 2681, 1513, 1319, 1526, 1733}},
    {{1411, 1553, 2906, 2488, 2228, 1308, 1882}},
    {{691, 2748, 1168, 2416, 3135, 948, 1813}},
    {{3451, 2438, 1443, 2705, 2289, 1297, 2100}},
    {{3380, 1642, 1163, 1542, 1152, 1576, 2240}},
    {{2409, 1185, 2602, 1697, 2260, 1932, 2045}},
    {{1503, 2692, 1658, 1470, 2519, 2529, 2295}},
    {{2519, 1045, 1250, 2870, 1487, 2373, 2292}},
    {{3558, 2116, 2134, 2033, 2209, 2311, 2293}},
    {{2509, 1868, 2140, 2037, 1879, 2244, 2260}},
    {{1777, 2696, 2970, 1023, 1721, 2575, 2046}},
    {{3329, 1690, 3055, 2181, 2046, 2126, 2188}},
    {{1396, 2101, 2141, 2911, 2258, 1123, 2161}},
    {{2855, 1145, 1621, 1230, 1766, 2613, 2377}},
    {{3470, 1575, 1905, 1040, 1902, 1623, 2361}},
    {{1076, 2794, 2149, 2185, 2801, 2539, 2396}},
    {{2331, 952, 1629, 1912, 1299, 2162, 2018}},
    {{2300, 2114, 1154, 1280, 2235, 2378, 1927}},
    {{886, 1213, 1840, 2790, 2449, 2301, 1879}},
    {{637, 3102, 2248, 3137, 1343, 2303, 2064}},
    {{775, 2979, 2305, 1182, 1721, 926, 1744}},
    {{3337, 2445, 2523, 2463, 1821, 2602, 1750}},
    {{2791, 2508, 2021, 1167, 1419, 2853, 2327}},
    {{709, 2226, 2093, 2738, 2331, 1723, 2186}},
    {{724, 1259, 1652, 1277, 2784, 2111, 2376}},
    {{721, 1253, 2799, 1171, 1813, 2369, 1986}},
    {{808, 2584, 1823, 1018, 2525, 3146, 1810}},
    {{2164, 1531, 1777, 2637, 1762, 2012, 1905}},
    {{2128, 1618, 2220, 2732, 2177, 1477, 2037}},
    {{3207, 2375, 1649, 1950, 1944, 1721, 2195}},
    {{2298, 1886, 2237, 2165, 1449, 1928, 1937}},
    {{3014, 2951, 1519, 2659, 1736, 1359, 1813}},
    {{2957, 1474, 1306, 2552, 3118, 3241, 2021}},
    {{2657, 2286, 2997, 1149, 2818, 2554, 2309}},
    {{887, 1774, 1462, 2649, 1146, 2359, 1681}},
    {{2098, 2902, 2488, 951, 2526, 2739, 1797}},
    {{2723, 2555, 1118, 977, 2632, 1948, 2375}},
    {{802, 2145, 1438, 2137, 2312, 2272, 1852}},
    {{2301, 3129, 2053, 2464, 2794, 2211, 2049}},
    {{2044, 2761, 2186, 1434, 2659, 1810, 2285}},
    {{701, 3011, 2341, 2733, 1966, 2087, 1669}},
    {{2268, 969, 2202, 2259, 1537, 1202, 1885}},
    {{2736, 2730, 1894, 1657, 1667, 1915, 2411}},
    {{1719, 1021, 2925, 1882, 1069, 1209, 1984}},
    {{1285, 2294, 3062, 2362, 1952, 1751, 1674}},
    {{2260, 2940, 2331, 2599, 2414, 910, 1889}},
    {{3341, 1387, 2105, 2730, 2754, 2943, 1675}},
    {{3587, 1645, 2026, 2560, 2218, 960, 2331}},
    {{944, 2832, 3062, 2897, 1794, 2352, 2214}},
    {{872, 2861, 1570, 2372, 1475, 1785, 2024}},
    {{1154, 1895, 1868, 2421, 2409, 1510, 2026}},
    {{3405, 1262, 1339, 1617, 2333, 2086, 2268}},
}};

struct NdiPoseSample {
    double q0 = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double txMm = 0.0;
    double tyMm = 0.0;
    double tzMm = 0.0;
    double error = 0.0;
    unsigned long frameNumber = 0;
    int visibleMarkerCount = 0;
};

struct AveragedNdiPose {
    NdiPoseSample pose;
    int acceptedSamples = 0;
};

struct DualToolCapture {
    AveragedNdiPose movingInCamera;
    AveragedNdiPose fixedInCamera;
    NdiPoseSample movingRelativeToFixed;
};

void waitForEnter(const std::string& message) {
    std::cout << message << std::flush;
    std::string line;
    std::getline(std::cin, line);
}

void disableAll(DynamixelMotor& motor, const std::vector<int>& motorIds) {
    for (int id : motorIds) {
        motor.disableTorque(id);
    }
}

std::vector<uint16_t> readActualTicks(
    DynamixelMotor& motor,
    const std::vector<int>& motorIds
) {
    std::vector<uint16_t> ticks;
    ticks.reserve(motorIds.size());

    for (int id : motorIds) {
        uint16_t position = 0;
        if (!motor.readPosition(id, position)) {
            throw std::runtime_error(
                "Failed to read motor " + std::to_string(id)
            );
        }
        if (!motor.isPositionSafe(id, position)) {
            throw std::runtime_error(
                "Motor " + std::to_string(id) +
                " is outside its calibrated safe range."
            );
        }
        ticks.push_back(position);
    }

    return ticks;
}

std::vector<double> ticksToRadiansVector(
    const std::vector<uint16_t>& ticks
) {
    if (ticks.size() != JOINT_COUNT ||
        jointCalibrations.size() < JOINT_COUNT) {
        throw std::runtime_error("Invalid joint calibration data.");
    }

    std::vector<double> radians;
    radians.reserve(JOINT_COUNT);

    for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
        radians.push_back(
            ticksToRadians(jointCalibrations[i], ticks[i])
        );
    }

    return radians;
}

bool targetPoseIsSafe(
    const std::array<uint16_t, JOINT_COUNT>& pose
) {
    if (jointCalibrations.size() < JOINT_COUNT) {
        return false;
    }

    for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
        const JointCalibration& joint = jointCalibrations[i];
        if (pose[i] < joint.minTick || pose[i] > joint.maxTick) {
            std::cerr
                << "Pose rejected: motor " << joint.id
                << " target " << pose[i]
                << " is outside [" << joint.minTick
                << ", " << joint.maxTick << "].\n";
            return false;
        }
    }

    return true;
}

void normalizeQuaternion(NdiPoseSample& pose) {
    const double norm = std::sqrt(
        pose.q0 * pose.q0 +
        pose.qx * pose.qx +
        pose.qy * pose.qy +
        pose.qz * pose.qz
    );

    if (norm <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("NDI quaternion is invalid.");
    }

    pose.q0 /= norm;
    pose.qx /= norm;
    pose.qy /= norm;
    pose.qz /= norm;
}

std::array<double, 3> rotateVectorByQuaternion(
    double q0,
    double qx,
    double qy,
    double qz,
    const std::array<double, 3>& v
) {
    const double vx = v[0];
    const double vy = v[1];
    const double vz = v[2];

    const double tx = 2.0 * (qy * vz - qz * vy);
    const double ty = 2.0 * (qz * vx - qx * vz);
    const double tz = 2.0 * (qx * vy - qy * vx);

    return {{
        vx + q0 * tx + (qy * tz - qz * ty),
        vy + q0 * ty + (qz * tx - qx * tz),
        vz + q0 * tz + (qx * ty - qy * tx)
    }};
}

NdiPoseSample computeMovingRelativeToFixed(
    const NdiPoseSample& movingInCamera,
    const NdiPoseSample& fixedInCamera
) {
    NdiPoseSample moving = movingInCamera;
    NdiPoseSample fixed = fixedInCamera;
    normalizeQuaternion(moving);
    normalizeQuaternion(fixed);

    // q_fixed_inverse = conjugate(q_fixed), because q_fixed is unit length.
    const double fw = fixed.q0;
    const double fx = -fixed.qx;
    const double fy = -fixed.qy;
    const double fz = -fixed.qz;

    NdiPoseSample relative;

    // q_relative = q_fixed_inverse * q_moving
    relative.q0 =
        fw * moving.q0 - fx * moving.qx -
        fy * moving.qy - fz * moving.qz;
    relative.qx =
        fw * moving.qx + fx * moving.q0 +
        fy * moving.qz - fz * moving.qy;
    relative.qy =
        fw * moving.qy - fx * moving.qz +
        fy * moving.q0 + fz * moving.qx;
    relative.qz =
        fw * moving.qz + fx * moving.qy -
        fy * moving.qx + fz * moving.q0;

    const std::array<double, 3> cameraDifference = {{
        moving.txMm - fixed.txMm,
        moving.tyMm - fixed.tyMm,
        moving.tzMm - fixed.tzMm
    }};

    const std::array<double, 3> fixedFrameTranslation =
        rotateVectorByQuaternion(fw, fx, fy, fz, cameraDifference);

    relative.txMm = fixedFrameTranslation[0];
    relative.tyMm = fixedFrameTranslation[1];
    relative.tzMm = fixedFrameTranslation[2];

    relative.error = std::sqrt(
        moving.error * moving.error +
        fixed.error * fixed.error
    );
    relative.frameNumber = moving.frameNumber;
    relative.visibleMarkerCount = (std::min)(
        moving.visibleMarkerCount,
        fixed.visibleMarkerCount
    );

    normalizeQuaternion(relative);
    return relative;
}

class NdiTracker {
public:
    NdiTracker(
        const char* device,
        const char* movingRomPath,
        const char* fixedRomPath
    )
        : device_(device),
          movingRomPath_(movingRomPath),
          fixedRomPath_(fixedRomPath) {}

    ~NdiTracker() {
        shutdown();
    }

    void initialize() {
        tracker_ = ndiOpenSerial(device_);
        if (tracker_ == nullptr) {
            throw std::runtime_error(
                std::string("Could not open NDI device: ") + device_
            );
        }

        ndiTSTOP(tracker_);

        ndiINIT(tracker_);
        requireNoNdiError("INIT");

        movingToolHandle_ = allocateAndInitializeTool(
            movingRomPath_,
            "moving"
        );
        fixedToolHandle_ = allocateAndInitializeTool(
            fixedRomPath_,
            "fixed"
        );

        printHandleDiagnostics(movingToolHandle_, "moving");
        printHandleDiagnostics(fixedToolHandle_, "fixed");

        enableTool(movingToolHandle_, "moving");
        enableTool(fixedToolHandle_, "fixed");

        ndiTSTART(tracker_);
        requireNoNdiError("TSTART");
        tracking_ = true;

        std::cout
            << "NDI tracking started.\n"
            << "Moving tool handle: 0x"
            << std::hex << movingToolHandle_ << '\n'
            << "Fixed tool handle: 0x"
            << fixedToolHandle_ << std::dec << '\n';

        waitForBothToolsVisible();
    }

    DualToolCapture collectBothTools() {
        waitForBothToolsVisible();

        std::cout
            << "Collecting " << NDI_REQUIRED_VALID_SAMPLES
            << " synchronized samples. Press 'p' to pause, 's' to skip "
            << "this pose.\n";

        std::vector<NdiPoseSample> movingAccepted;
        std::vector<NdiPoseSample> fixedAccepted;

        movingAccepted.reserve(NDI_REQUIRED_VALID_SAMPLES);
        fixedAccepted.reserve(NDI_REQUIRED_VALID_SAMPLES);

        int movingInvalidCount = 0;
        int fixedInvalidCount = 0;
        int rejectedPairCount = 0;

        NdiToolStatus lastMovingStatus = NdiToolStatus::Detected;
        NdiToolStatus lastFixedStatus = NdiToolStatus::Detected;
        bool movingStatusPrinted = false;
        bool fixedStatusPrinted = false;

        for (int attempt = 0;
             attempt < NDI_MAX_ATTEMPTS &&
             static_cast<int>(movingAccepted.size()) <
                 NDI_REQUIRED_VALID_SAMPLES;
             ++attempt) {

            handleUserControls();

            requestBxUpdate();

            printToolStatusIfChanged(
                "Moving",
                classifyToolStatus(movingToolHandle_),
                lastMovingStatus,
                movingStatusPrinted
            );
            printToolStatusIfChanged(
                "Fixed",
                classifyToolStatus(fixedToolHandle_),
                lastFixedStatus,
                fixedStatusPrinted
            );

            NdiPoseSample moving;
            NdiPoseSample fixed;

            const bool movingValid = tryReadToolFromCurrentBx(
                movingToolHandle_,
                "moving",
                moving,
                movingInvalidCount,
                false
            );

            const bool fixedValid = tryReadToolFromCurrentBx(
                fixedToolHandle_,
                "fixed",
                fixed,
                fixedInvalidCount,
                false
            );

            if (movingValid && fixedValid) {
                movingAccepted.push_back(moving);
                fixedAccepted.push_back(fixed);
            } else {
                ++rejectedPairCount;
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(NDI_SAMPLE_INTERVAL_MS)
            );
        }

        if (static_cast<int>(movingAccepted.size()) <
            NDI_REQUIRED_VALID_SAMPLES) {
            throw std::runtime_error(
                "Not enough synchronized NDI BX samples. Accepted " +
                std::to_string(movingAccepted.size()) + " of " +
                std::to_string(NDI_REQUIRED_VALID_SAMPLES) +
                ". Moving invalid: " +
                std::to_string(movingInvalidCount) +
                ", fixed invalid: " +
                std::to_string(fixedInvalidCount) +
                ", rejected pairs: " +
                std::to_string(rejectedPairCount) + "."
            );
        }

        DualToolCapture capture;
        capture.movingInCamera = averageSamples(movingAccepted);
        capture.fixedInCamera = averageSamples(fixedAccepted);
        capture.movingRelativeToFixed = computeMovingRelativeToFixed(
            capture.movingInCamera.pose,
            capture.fixedInCamera.pose
        );

        return capture;
    }

    void shutdown() noexcept {
        if (tracker_ == nullptr) {
            return;
        }

        if (tracking_) {
            ndiTSTOP(tracker_);
            tracking_ = false;
        }

        releaseTool(movingToolHandle_);
        releaseTool(fixedToolHandle_);

        ndiCloseSerial(tracker_);
        tracker_ = nullptr;
    }

private:
    int allocateAndInitializeTool(
        const char* romPath,
        const char* toolName
    ) {
        ndiPHRQ(tracker_, "********", "0", "1", "**", "**");
        requireNoNdiError("PHRQ");

        const int handle = ndiGetPHRQHandle(tracker_);
        if (handle <= 0) {
            throw std::runtime_error(
                std::string("NDI did not allocate the ") +
                toolName + " passive tool handle."
            );
        }

        if (ndiPVWRFromFile(
                tracker_,
                handle,
                const_cast<char*>(romPath)
            ) != NDI_OKAY) {
            throw std::runtime_error(
                std::string("Could not load ") + toolName +
                " ROM file: " + romPath
            );
        }
        requireNoNdiError("PVWRFromFile");

        ndiPINIT(tracker_, handle);
        requireNoNdiError("PINIT");

        std::cout
            << "Loaded " << toolName
            << " tool ROM into handle 0x"
            << std::hex << handle << std::dec << ".\n";

        return handle;
    }

    void printHandleDiagnostics(
        int handle,
        const char* toolName
    ) {
        ndiPHINF(
            tracker_,
            handle,
            NDI_BASIC |
            NDI_PART_NUMBER |
            NDI_MARKER_TYPE |
            NDI_PORT_LOCATION
        );
        requireNoNdiError("PHINF");

        const int status = ndiGetPHINFPortStatus(tracker_);
        const int markerType = ndiGetPHINFMarkerType(tracker_);

        // ndicapi writes a fixed-width tool information field.
        // Allocate extra space and force termination before printing.
        char toolInfo[128] = {};
        ndiGetPHINFToolInfo(tracker_, toolInfo);
        toolInfo[127] = '\0';

        std::cout
            << "PHINF " << toolName
            << " handle 0x" << std::hex << handle << std::dec
            << " | status=0x" << std::hex << status << std::dec
            << " | markerType=" << markerType
            << " | toolInfo=\"";

        for (int i = 0; i < 30 && toolInfo[i] != '\0'; ++i) {
            const unsigned char c =
                static_cast<unsigned char>(toolInfo[i]);

            if (c >= 32 && c <= 126) {
                std::cout << toolInfo[i];
            } else {
                std::cout << '.';
            }
        }

        std::cout << "\"\n";
    }

    void enableTool(int handle, const char* toolName) {
        ndiPENA(tracker_, handle, NDI_DYNAMIC);
        requireNoNdiError("PENA");

        std::cout
            << "Enabled " << toolName
            << " tool handle 0x"
            << std::hex << handle << std::dec << ".\n";
    }

    void requestBxUpdate() {
        ndiCommand(
            tracker_,
            "BX:%04X",
            NDI_XFORMS_AND_STATUS |
            NDI_ADDITIONAL_INFO |
            NDI_3D_MARKER_POSITIONS |
            NDI_NOT_NORMALLY_REPORTED
        );
        requireNoNdiError("BX");
    }

    // Classifies the most recent BX reply for one tool into a simple status
    // label, purely for live console feedback (does not affect which
    // samples are accepted into the pose average).
    NdiToolStatus classifyToolStatus(int toolHandle) {
        float transform[8] = {};
        const int result = ndiGetBXTransform(tracker_, toolHandle, transform);

        if (result == NDI_DISABLED) {
            return NdiToolStatus::Disabled;
        }
        if (result == NDI_MISSING) {
            return NdiToolStatus::Missing;
        }

        const int status = ndiGetBXPortStatus(tracker_, toolHandle);
        if ((status & NDI_OUT_OF_VOLUME) != 0) {
            return NdiToolStatus::OutOfVolume;
        }

        if (!std::isfinite(transform[7]) || transform[7] > MAX_NDI_ERROR) {
            return NdiToolStatus::LowQuality;
        }

        return NdiToolStatus::Detected;
    }

    void waitForBothToolsVisible() {
        constexpr int STARTUP_ATTEMPTS = 600;

        std::cout
            << "Waiting for both NDI tools to become visible. Press 'p' "
            << "to pause, 's' to skip this pose.\n";

        NdiToolStatus lastMovingStatus = NdiToolStatus::Detected;
        NdiToolStatus lastFixedStatus = NdiToolStatus::Detected;
        bool movingStatusPrinted = false;
        bool fixedStatusPrinted = false;

        for (int attempt = 1; attempt <= STARTUP_ATTEMPTS; ++attempt) {
            handleUserControls();

            requestBxUpdate();

            printToolStatusIfChanged(
                "Moving",
                classifyToolStatus(movingToolHandle_),
                lastMovingStatus,
                movingStatusPrinted
            );
            printToolStatusIfChanged(
                "Fixed",
                classifyToolStatus(fixedToolHandle_),
                lastFixedStatus,
                fixedStatusPrinted
            );

            NdiPoseSample moving;
            NdiPoseSample fixed;
            int movingInvalid = 0;
            int fixedInvalid = 0;

            const bool movingValid = tryReadToolFromCurrentBx(
                movingToolHandle_,
                "moving",
                moving,
                movingInvalid,
                false
            );

            const bool fixedValid = tryReadToolFromCurrentBx(
                fixedToolHandle_,
                "fixed",
                fixed,
                fixedInvalid,
                false
            );

            if (movingValid && fixedValid) {
                std::cout
                    << "Both NDI tools are visible through BX.\n";
                return;
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(50)
            );
        }

        throw std::runtime_error(
            "BX did not produce valid transforms for both tools after " +
            std::to_string(STARTUP_ATTEMPTS * 50 / 1000) +
            " seconds. Check that both tools have a clear, unobstructed "
            "line of sight to the tracker."
        );
    }

    bool tryReadToolFromCurrentBx(
        int toolHandle,
        const char* toolName,
        NdiPoseSample& sample,
        int& invalidCount,
        bool printErrors
    ) {
        float transform[8] = {};

        const int result = ndiGetBXTransform(
            tracker_,
            toolHandle,
            transform
        );

        const int portStatus = ndiGetBXPortStatus(
            tracker_,
            toolHandle
        );

        const unsigned long frame = ndiGetBXFrame(
            tracker_,
            toolHandle
        );

        if (result != NDI_OKAY) {
            ++invalidCount;

            if (printErrors) {
                std::cerr
                    << "NDI " << toolName
                    << " BX transform rejected: result="
                    << result
                    << ", status=0x"
                    << std::hex << portStatus << std::dec
                    << ", frame=" << frame << ".\n";
            }
            return false;
        }

        if ((portStatus & NDI_ENABLED) == 0 ||
            (portStatus & NDI_OUT_OF_VOLUME) != 0) {
            ++invalidCount;

            if (printErrors) {
                std::cerr
                    << "NDI " << toolName
                    << " BX port status rejected: status=0x"
                    << std::hex << portStatus << std::dec
                    << ", frame=" << frame << ".\n";
            }
            return false;
        }

        sample.q0 = transform[0];
        sample.qx = transform[1];
        sample.qy = transform[2];
        sample.qz = transform[3];
        sample.txMm = transform[4];
        sample.tyMm = transform[5];
        sample.tzMm = transform[6];
        sample.error = transform[7];
        sample.frameNumber = frame;

        // Marker-level counting is not used as a blocking condition here.
        sample.visibleMarkerCount = REQUIRED_VISIBLE_MARKERS;

        const bool finite =
            std::isfinite(sample.q0) &&
            std::isfinite(sample.qx) &&
            std::isfinite(sample.qy) &&
            std::isfinite(sample.qz) &&
            std::isfinite(sample.txMm) &&
            std::isfinite(sample.tyMm) &&
            std::isfinite(sample.tzMm) &&
            std::isfinite(sample.error);

        if (!finite || sample.error > MAX_NDI_ERROR) {
            ++invalidCount;

            if (printErrors) {
                std::cerr
                    << "NDI " << toolName
                    << " BX quality rejected: error="
                    << sample.error
                    << ", frame=" << frame << ".\n";
            }
            return false;
        }

        return true;
    }

    AveragedNdiPose averageSamples(
        const std::vector<NdiPoseSample>& accepted
    ) {
        if (accepted.empty()) {
            throw std::runtime_error(
                "Cannot average an empty NDI sample set."
            );
        }

        NdiPoseSample mean;
        const NdiPoseSample& reference = accepted.front();

        for (NdiPoseSample sample : accepted) {
            const double dot =
                sample.q0 * reference.q0 +
                sample.qx * reference.qx +
                sample.qy * reference.qy +
                sample.qz * reference.qz;

            if (dot < 0.0) {
                sample.q0 = -sample.q0;
                sample.qx = -sample.qx;
                sample.qy = -sample.qy;
                sample.qz = -sample.qz;
            }

            mean.q0 += sample.q0;
            mean.qx += sample.qx;
            mean.qy += sample.qy;
            mean.qz += sample.qz;
            mean.txMm += sample.txMm;
            mean.tyMm += sample.tyMm;
            mean.tzMm += sample.tzMm;
            mean.error += sample.error;
            mean.frameNumber = sample.frameNumber;
            mean.visibleMarkerCount += sample.visibleMarkerCount;
        }

        const double count = static_cast<double>(accepted.size());
        mean.q0 /= count;
        mean.qx /= count;
        mean.qy /= count;
        mean.qz /= count;
        mean.txMm /= count;
        mean.tyMm /= count;
        mean.tzMm /= count;
        mean.error /= count;
        mean.visibleMarkerCount = static_cast<int>(
            std::lround(mean.visibleMarkerCount / count)
        );

        normalizeQuaternion(mean);

        return {mean, static_cast<int>(accepted.size())};
    }

    void releaseTool(int& handle) noexcept {
        if (handle > 0) {
            ndiPDIS(tracker_, handle);
            ndiPHF(tracker_, handle);
            handle = 0;
        }
    }

    void requireNoNdiError(const char* operation) {
        const int error = ndiGetError(tracker_);
        if (error != NDI_OKAY) {
            throw std::runtime_error(
                std::string("NDI ") + operation +
                " failed with error code " +
                std::to_string(error)
            );
        }
    }

    const char* device_;
    const char* movingRomPath_;
    const char* fixedRomPath_;
    ndicapi* tracker_ = nullptr;
    int movingToolHandle_ = 0;
    int fixedToolHandle_ = 0;
    bool tracking_ = false;
};

void writePoseFields(
    std::ofstream& csv,
    const std::string& prefix
) {
    csv
        << ',' << prefix << "_q0"
        << ',' << prefix << "_qx"
        << ',' << prefix << "_qy"
        << ',' << prefix << "_qz"
        << ',' << prefix << "_tx_mm"
        << ',' << prefix << "_ty_mm"
        << ',' << prefix << "_tz_mm"
        << ',' << prefix << "_error"
        << ',' << prefix << "_frame"
        << ',' << prefix << "_visible_markers";
}

void writeCsvHeader(std::ofstream& csv) {
    csv << "pose_id,timestamp_ms";

    for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
        csv << ",target_tick_" << i;
    }
    for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
        csv << ",actual_tick_" << i;
    }
    for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
        csv << ",actual_rad_" << i;
    }

    writePoseFields(csv, "moving_camera");
    csv << ",moving_accepted_samples";

    writePoseFields(csv, "fixed_camera");
    csv << ",fixed_accepted_samples";

    writePoseFields(csv, "moving_relative_fixed");
    csv << '\n';
}

void appendPoseFields(
    std::ofstream& csv,
    const NdiPoseSample& pose
) {
    csv
        << ',' << pose.q0
        << ',' << pose.qx
        << ',' << pose.qy
        << ',' << pose.qz
        << ',' << pose.txMm
        << ',' << pose.tyMm
        << ',' << pose.tzMm
        << ',' << pose.error
        << ',' << pose.frameNumber
        << ',' << pose.visibleMarkerCount;
}

void appendCsvRow(
    std::ofstream& csv,
    std::size_t poseIndex,
    const std::array<uint16_t, JOINT_COUNT>& targetTicks,
    const std::vector<uint16_t>& actualTicks,
    const std::vector<double>& actualRadians,
    const DualToolCapture& ndi
) {
    const auto timestampMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();

    csv << poseIndex + 1 << ',' << timestampMs;

    for (uint16_t tick : targetTicks) {
        csv << ',' << tick;
    }
    for (uint16_t tick : actualTicks) {
        csv << ',' << tick;
    }
    for (double radians : actualRadians) {
        csv << ',' << std::setprecision(12) << radians;
    }

    appendPoseFields(csv, ndi.movingInCamera.pose);
    csv << ',' << ndi.movingInCamera.acceptedSamples;

    appendPoseFields(csv, ndi.fixedInCamera.pose);
    csv << ',' << ndi.fixedInCamera.acceptedSamples;

    appendPoseFields(csv, ndi.movingRelativeToFixed);
    csv << '\n';

    csv.flush();
}

void printPose(
    const std::string& label,
    const NdiPoseSample& pose
) {
    std::cout
        << label << " translation: ["
        << pose.txMm << ", "
        << pose.tyMm << ", "
        << pose.tzMm << "] mm\n"
        << label << " quaternion: ["
        << pose.q0 << ", "
        << pose.qx << ", "
        << pose.qy << ", "
        << pose.qz << "]\n"
        << label << " error: " << pose.error << '\n'
        << label << " visible markers: "
        << pose.visibleMarkerCount << '\n';
}

void printCapturedPose(
    std::size_t poseIndex,
    const std::vector<uint16_t>& actualTicks,
    const std::vector<double>& actualRadians,
    const DualToolCapture& ndi
) {
    std::cout << "\nCaptured pose " << poseIndex + 1 << ":\n";

    for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
        std::cout
            << "Motor " << i
            << " | " << actualTicks[i] << " ticks"
            << " | " << std::fixed << std::setprecision(6)
            << actualRadians[i] << " rad\n";
    }

    printPose("Moving tool in camera frame", ndi.movingInCamera.pose);
    std::cout
        << "Moving accepted samples: "
        << ndi.movingInCamera.acceptedSamples << '\n';

    printPose("Fixed tool in camera frame", ndi.fixedInCamera.pose);
    std::cout
        << "Fixed accepted samples: "
        << ndi.fixedInCamera.acceptedSamples << '\n';

    printPose(
        "Moving tool relative to fixed tool",
        ndi.movingRelativeToFixed
    );
}

}  // namespace

int main() {
    const std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};

    DynamixelMotor motor(
        CYTON_DEVICE,
        CYTON_BAUD_RATE,
        CYTON_PROTOCOL_VERSION
    );

    try {
        for (const auto& pose : TARGET_POSES) {
            if (!targetPoseIsSafe(pose)) {
                throw std::runtime_error(
                    "At least one predefined target pose is unsafe."
                );
            }
        }

        std::ofstream csv(
            OUTPUT_CSV,
            std::ios::out | std::ios::trunc
        );
        if (!csv) {
            throw std::runtime_error(
                std::string("Could not open CSV: ") + OUTPUT_CSV
            );
        }
        writeCsvHeader(csv);

        if (!motor.connect()) {
            throw std::runtime_error(
                "Could not connect to the Cyton motors."
            );
        }

        for (int id : motorIds) {
            if (!motor.pingMotor(id)) {
                throw std::runtime_error(
                    "Could not ping motor " + std::to_string(id)
                );
            }
        }

        NdiTracker ndi(
            NDI_DEVICE,
            MOVING_TOOL_ROM,
            FIXED_TOOL_ROM
        );
        ndi.initialize();

        std::cout
            << "\nFive-pose Cyton + dual-tool NDI capture test\n"
            << "Output: " << OUTPUT_CSV << "\n"
            << "Moving marker: rigid body 2\n"
            << "Fixed marker: rigid body 3\n"
            << "Each Enter press commands exactly one pose.\n"
            << "While waiting on NDI tracking: press 'p' to pause/resume, "
            << "'s' to skip the current pose.\n"
            << "Press Ctrl+C or use your hardware emergency stop "
            << "if needed.\n";

        for (std::size_t poseIndex = 0;
             poseIndex < TARGET_POSES.size();
             ++poseIndex) {
            waitForEnter(
                "\nPress Enter to move to pose " +
                std::to_string(poseIndex + 1) + " of " +
                std::to_string(POSE_COUNT) + "..."
            );

            const auto& targetArray = TARGET_POSES[poseIndex];
            const std::vector<uint16_t> targetTicks(
                targetArray.begin(),
                targetArray.end()
            );

            if (!motor.moveJointsSafely(
                    motorIds,
                    targetTicks,
                    MOVING_SPEED,
                    MOTOR_TOLERANCE_TICKS,
                    MOVE_TIMEOUT_SECONDS,
                    true,
                    STALL_REPEATS_TO_DETECT,
                    STALL_GRACE_SECONDS
                )) {
                std::cout
                    << "\nWarning: pose " << poseIndex + 1
                    << " did not fully reach its target within the "
                    << "timeout. Re-enabling torque and continuing with "
                    << "the actual position reached.\n";

                for (int id : motorIds) {
                    motor.enableTorque(id);
                }
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(SETTLING_TIME_MS)
            );

            const std::vector<uint16_t> actualTicks =
                readActualTicks(motor, motorIds);

            const std::vector<double> actualRadians =
                ticksToRadiansVector(actualTicks);

            try {
                const DualToolCapture ndiCapture =
                    ndi.collectBothTools();

                appendCsvRow(
                    csv,
                    poseIndex,
                    targetArray,
                    actualTicks,
                    actualRadians,
                    ndiCapture
                );

                printCapturedPose(
                    poseIndex,
                    actualTicks,
                    actualRadians,
                    ndiCapture
                );
            } catch (const PoseSkippedByUser&) {
                std::cout
                    << "\nPose " << poseIndex + 1
                    << " skipped by user. No NDI data recorded for "
                    << "this pose.\n";
            }
        }

        std::cout
            << "\nAll five poses were captured successfully.\n"
            << "Saved data to " << OUTPUT_CSV << '\n';

        disableAll(motor, motorIds);
        motor.disconnect();
        return 0;
    } catch (const PoseSkippedByUser&) {
        std::cout
            << "\nSkipped during initial NDI setup (before any pose "
            << "was captured). Exiting.\n";
        disableAll(motor, motorIds);
        motor.disconnect();
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "\nERROR: " << error.what() << '\n';
        disableAll(motor, motorIds);
        motor.disconnect();
        return 1;
    }
}