#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <conio.h>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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

// Prints the manual/auto mode banner after a spacebar toggle.
void printModeToggle(bool manualModeEnabled) {
    std::cout
        << (manualModeEnabled
                ? "\nManual mode ENABLED -- Enter will be required before "
                  "each future pose.\n"
                : "\nManual mode DISABLED -- poses will auto-advance "
                  "again.\n");
}

// Non-blocking check for the mode-toggle spacebar. Usable at points where
// pause/skip wouldn't make sense (e.g. between poses, before a new one has
// started) -- does nothing for any other key.
void checkForModeToggle(bool& manualModeEnabled) {
    if (!_kbhit()) {
        return;
    }
    if (_getch() == ' ') {
        manualModeEnabled = !manualModeEnabled;
        printModeToggle(manualModeEnabled);
    }
}

// Non-blocking check for the pause ('p') / skip ('s') / manual-mode-toggle
// (' ') hotkeys. Call once per polling iteration. Throws PoseSkippedByUser
// if skip is pressed (either directly or while paused). Returns how long
// this call spent paused, so callers can shift their own elapsed-time
// budgets forward by that amount.
std::chrono::milliseconds handleUserControls(bool& manualModeEnabled) {
    if (!_kbhit()) {
        return std::chrono::milliseconds(0);
    }

    const int key = _getch();

    if (key == 's' || key == 'S') {
        throw PoseSkippedByUser{};
    }

    if (key == ' ') {
        manualModeEnabled = !manualModeEnabled;
        printModeToggle(manualModeEnabled);
        return std::chrono::milliseconds(0);
    }

    if (key == 'p' || key == 'P') {
        const auto pauseStart = std::chrono::steady_clock::now();
        std::cout
            << "\nProcess paused. Press 'p' to resume, 's' to skip this "
            << "pose, or space to toggle manual mode.\n";

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
                if (resumeKey == ' ') {
                    manualModeEnabled = !manualModeEnabled;
                    printModeToggle(manualModeEnabled);
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

// Stability-based settling wait (currently used only by --quick-test, see
// NdiTracker::waitForPoseStability()): instead of assuming SETTLING_TIME_MS
// is always enough, poll the moving tool until its last
// STABILITY_WINDOW_SIZE valid readings all agree within
// STABILITY_TOLERANCE_MM of each other. STABILITY_TOLERANCE_MM is set just
// above the ~0.3-0.4mm noise floor measured by --settling-diagnostic on
// these same poses. STABILITY_MAX_WAIT_MS caps how long this can run so a
// pose that never stabilizes doesn't block forever.
constexpr std::size_t STABILITY_WINDOW_SIZE = 5;
constexpr double STABILITY_TOLERANCE_MM = 0.4;
constexpr long STABILITY_MAX_WAIT_MS = 4000;

// If the combined position error across all joints reads exactly the same
// value for this many consecutive ~100ms checks, the arm has effectively
// stopped moving (stalled short of tolerance, e.g. friction/backlash) --
// no point waiting out the full MOVE_TIMEOUT_SECONDS in that case, so the
// current position is accepted immediately once this triggers.
constexpr int STALL_REPEATS_TO_DETECT = 3;

constexpr int NDI_REQUIRED_VALID_SAMPLES = 30;
constexpr int NDI_SAMPLE_INTERVAL_MS = 20;
constexpr int REQUIRED_VISIBLE_MARKERS = 4;
constexpr double MAX_NDI_ERROR = 0.50;

// --settling-diagnostic mode: reuses the first N (already hand-verified
// safe) poses from TARGET_POSES below, but instead of the normal
// SETTLING_TIME_MS sleep + 30-sample average, logs every single BX poll for
// SETTLING_DIAGNOSTIC_DURATION_MS starting the instant the move reports
// done, to see how much the tracked pose is still changing early on.
constexpr int SETTLING_DIAGNOSTIC_POSE_COUNT = 15;
constexpr int SETTLING_DIAGNOSTIC_DURATION_MS = 4000;
constexpr const char* SETTLING_DIAGNOSTIC_CSV = "settling_diagnostic.csv";

// --quick-test mode: runs the SAME real capture pipeline (stability-based
// settle via waitForPoseStability() + full 30-sample average per tool via
// collectBothTools()) as the normal 100-pose run, but in one short sitting
// -- to test whether collecting the dataset without the multi-hour
// session-length drift found in the original (superseded) 200-pose data
// gets the fitted calibration error down near the arm's ~0.5mm repeatability
// floor. Writes to its own CSV so it never touches five_pose_ndi_capture.csv
// or its resume state.
//
// QUICK_TEST_POSE_INDICES covers all 100 poses in this hand-recorded set:
// unlike the old 200-pose dataset's first-N slices, this dataset was
// hand-recorded with per-joint range coverage already in mind and
// independently verified to cover ~90-100% of every joint's safe range
// (see calibration/diag_new_dataset_diversity.py), so no subset curation
// is needed here.
constexpr std::array<int, 100> QUICK_TEST_POSE_INDICES = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99
};
constexpr const char* QUICK_TEST_CSV = "quick_calibration_test.csv";

// Hand-recorded via tests/record_hand_poses.cpp: the arm was physically
// moved by hand (torque off) into each pose and the resulting joint ticks
// captured, so every pose here is a real, human-verified configuration --
// not randomly sampled -- meaning it's known in advance to be physically
// reachable, collision-free, and (assuming it was checked live against the
// "Moving tool: DETECTED/MISSING" status while posing) marker-visible.
const std::array<std::array<uint16_t, JOINT_COUNT>, POSE_COUNT> TARGET_POSES = {{
    {{336, 852, 2084, 3266, 2067, 2067, 3753}},
    {{612, 852, 2083, 3267, 2095, 2053, 3460}},
    {{832, 852, 2106, 3265, 2090, 2085, 3106}},
    {{1130, 852, 2165, 3267, 2112, 2058, 2695}},
    {{1527, 852, 2167, 3248, 2121, 2087, 2439}},
    {{1780, 852, 2167, 3244, 2119, 2109, 2160}},
    {{2103, 852, 2165, 3267, 2140, 2143, 1735}},
    {{2399, 851, 2180, 3266, 2153, 2088, 1541}},
    {{2813, 852, 2180, 3260, 2147, 2122, 1170}},
    {{3259, 852, 2180, 3260, 2149, 2128, 661}},
    {{3263, 851, 2182, 3260, 2150, 2129, 662}},
    {{3589, 852, 2180, 3239, 2152, 2068, 438}},
    {{3740, 852, 2180, 3249, 2154, 2039, 336}},
    {{3498, 852, 2180, 3244, 2154, 2047, 337}},
    {{3777, 2067, 928, 1062, 3035, 1937, 3190}},
    {{3788, 2180, 1229, 1054, 3029, 2207, 3293}},
    {{3788, 2182, 1605, 1062, 3027, 2529, 3294}},
    {{3751, 2202, 1845, 1053, 2700, 2819, 3688}},
    {{3787, 2196, 2254, 1055, 2802, 3081, 3516}},
    {{3718, 2251, 2793, 1054, 1216, 2329, 1241}},
    {{3718, 2253, 3047, 1061, 1219, 2152, 1266}},
    {{3716, 2247, 3325, 1060, 1183, 1854, 1457}},
    {{3692, 3222, 3306, 2043, 2127, 1176, 1614}},
    {{3786, 3225, 1379, 2012, 1598, 968, 3444}},
    {{3776, 3227, 1214, 1869, 1366, 1206, 2907}},
    {{3776, 3228, 1216, 1943, 1044, 1449, 2722}},
    {{3776, 3228, 1216, 2215, 1041, 1430, 2677}},
    {{3776, 3228, 1238, 2217, 1178, 1833, 2677}},
    {{3776, 3228, 1378, 2217, 1332, 1995, 2678}},
    {{3274, 3225, 1671, 1918, 954, 1238, 3669}},
    {{3291, 2169, 1179, 1341, 983, 1867, 3753}},
    {{3597, 1780, 1462, 1267, 1140, 1584, 353}},
    {{3785, 1979, 1462, 1057, 1150, 1639, 720}},
    {{3785, 1937, 1889, 1086, 1149, 1243, 802}},
    {{3745, 1986, 2209, 1203, 1027, 1216, 745}},
    {{3746, 1987, 2512, 1159, 1026, 1217, 872}},
    {{3746, 1987, 2738, 1152, 1024, 1429, 936}},
    {{3746, 2063, 3075, 1193, 1040, 1496, 937}},
    {{3746, 2105, 3207, 1297, 1039, 2067, 952}},
    {{3758, 2109, 3219, 1453, 1042, 2763, 953}},
    {{3787, 2166, 3319, 1585, 1355, 2778, 689}},
    {{3673, 2004, 2721, 2892, 2917, 2029, 501}},
    {{3674, 2151, 2807, 2863, 2965, 2345, 499}},
    {{3761, 2286, 2807, 2889, 2844, 2493, 561}},
    {{3761, 2340, 2861, 2893, 3163, 2727, 508}},
    {{3767, 2499, 2858, 2839, 2519, 2621, 843}},
    {{3787, 2659, 2859, 2867, 2432, 3008, 910}},
    {{3785, 1945, 1739, 3192, 1044, 2616, 2769}},
    {{3786, 1911, 1740, 3192, 1011, 2228, 2854}},
    {{3785, 1864, 1175, 2312, 955, 2098, 3747}},
    {{3785, 1863, 1192, 2314, 1125, 1719, 3746}},
    {{3775, 1859, 1194, 2388, 1639, 1276, 3743}},
    {{3761, 1469, 1271, 2898, 2339, 1704, 863}},
    {{3762, 1457, 1272, 3061, 2353, 2013, 912}},
    {{3761, 1453, 1271, 3104, 2349, 2318, 1010}},
    {{3762, 1451, 1272, 3210, 2349, 2553, 1017}},
    {{3494, 2042, 2128, 2547, 2349, 2274, 591}},
    {{3493, 2172, 2307, 2604, 2349, 2453, 592}},
    {{3486, 2366, 2352, 2605, 2349, 2535, 592}},
    {{3475, 2518, 2350, 2603, 2346, 3076, 592}},
    {{3482, 2592, 2349, 2668, 2346, 3162, 591}},
    {{3538, 3023, 2348, 2340, 2032, 1241, 931}},
    {{3538, 2995, 2347, 2034, 2032, 1462, 931}},
    {{3533, 2984, 2349, 1777, 2033, 1622, 930}},
    {{3469, 2986, 2352, 1413, 2036, 1914, 931}},
    {{3222, 3022, 2353, 1328, 2036, 1896, 1210}},
    {{3195, 2889, 2350, 1108, 1949, 2177, 1209}},
    {{3197, 2646, 2349, 1056, 1858, 2310, 1209}},
    {{3204, 2336, 2350, 962, 1819, 2572, 1442}},
    {{3196, 2007, 2348, 1000, 1592, 2871, 1253}},
    {{3076, 1869, 2140, 854, 1564, 3030, 1518}},
    {{2958, 1610, 2065, 845, 1564, 3126, 1639}},
    {{2958, 1371, 2066, 868, 1523, 3102, 1637}},
    {{2958, 1160, 2066, 852, 1520, 3163, 1636}},
    {{3297, 3226, 1791, 1991, 1941, 1140, 505}},
    {{3296, 3132, 1792, 1581, 1942, 1607, 507}},
    {{3342, 2866, 1792, 1343, 2017, 2028, 550}},
    {{3343, 2621, 1793, 1347, 2028, 2182, 549}},
    {{3339, 2337, 1979, 1693, 2028, 2166, 782}},
    {{3342, 2089, 2043, 2028, 2027, 2155, 775}},
    {{3339, 1960, 1964, 1923, 2027, 2510, 871}},
    {{3319, 1828, 1932, 1690, 2017, 2898, 871}},
    {{3231, 1711, 1885, 1603, 2018, 3010, 874}},
    {{3028, 1446, 1811, 1560, 2021, 3185, 1238}},
    {{2943, 1262, 1773, 1552, 2021, 3291, 1513}},
    {{2927, 850, 2081, 2965, 1982, 3243, 1224}},
    {{3045, 852, 2080, 3274, 2278, 2866, 973}},
    {{3408, 851, 2078, 3274, 2691, 2600, 415}},
    {{3775, 851, 2134, 3276, 2848, 2322, 334}},
    {{3775, 852, 3006, 3256, 2927, 1578, 3752}},
    {{3778, 1585, 3008, 2970, 2021, 1698, 3752}},
    {{3776, 1831, 3315, 2993, 2002, 1522, 3753}},
    {{3775, 1860, 3317, 2990, 2685, 1444, 412}},
    {{3756, 1979, 3303, 2987, 2684, 1961, 677}},
    {{3775, 2013, 3309, 3094, 2957, 3156, 1263}},
    {{3774, 2071, 3307, 3006, 2522, 1342, 477}},
    {{3769, 1995, 2940, 2748, 2923, 2204, 339}},
    {{3770, 1976, 2618, 2787, 3246, 2532, 339}},
    {{3773, 1915, 2036, 3174, 3090, 2945, 853}},
    {{3773, 1914, 2036, 3238, 2909, 2259, 1010}},
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

// One instantaneous (non-averaged) moving-relative-to-fixed sample, tagged
// with elapsed time since collectRawSettlingSeries() started -- used only by
// the --settling-diagnostic mode to see how the pose changes right after a
// move, before the real capture loop's 30-sample average would begin.
struct SettlingSample {
    int poseIndex = 0;
    long elapsedMs = 0;
    bool valid = false;
    double txMm = 0.0;
    double tyMm = 0.0;
    double tzMm = 0.0;
    double error = 0.0;
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

    // No isPositionSafe() gate here: every TARGET_POSES entry was hand-
    // verified by physically moving the arm there (via
    // record_hand_poses.cpp), so a motor reading slightly outside
    // jointCalibrations here reflects settling/backlash noise around an
    // already-confirmed-safe position, not a real safety concern.
    for (int id : motorIds) {
        uint16_t position = 0;
        if (!motor.readPosition(id, position)) {
            throw std::runtime_error(
                "Failed to read motor " + std::to_string(id)
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

    void initialize(bool& manualModeEnabled) {
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

        waitForBothToolsVisible(manualModeEnabled);
    }

    DualToolCapture collectBothTools(bool& manualModeEnabled) {
        waitForBothToolsVisible(manualModeEnabled);

        std::cout
            << "Collecting " << NDI_REQUIRED_VALID_SAMPLES
            << " synchronized samples. Press 'p' to pause, 's' to skip "
            << "this pose, or space to toggle manual mode.\n";

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

        // No attempt cap here on purpose: a mere visibility/quality
        // timeout must never abort the whole run (see the "'s' to skip"
        // hotkey for the deliberate, user-triggered way to give up on a
        // pose instead). Every WARNING_INTERVAL_ATTEMPTS attempts
        // (~NDI_SAMPLE_INTERVAL_MS * that many ms) it prints a reminder
        // instead of throwing, and keeps going indefinitely.
        constexpr int WARNING_INTERVAL_ATTEMPTS = 1000;

        for (int attempt = 0;
             static_cast<int>(movingAccepted.size()) <
                 NDI_REQUIRED_VALID_SAMPLES;
             ++attempt) {

            handleUserControls(manualModeEnabled);

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

            if (attempt > 0 && attempt % WARNING_INTERVAL_ATTEMPTS == 0) {
                std::cout
                    << "\nWARNING: only " << movingAccepted.size() << "/"
                    << NDI_REQUIRED_VALID_SAMPLES << " valid samples "
                    << "collected after "
                    << (attempt * NDI_SAMPLE_INTERVAL_MS) / 1000
                    << "s (moving invalid: " << movingInvalidCount
                    << ", fixed invalid: " << fixedInvalidCount
                    << ", rejected pairs: " << rejectedPairCount << "). "
                    << "Check that both tools have a clear, unobstructed "
                    << "line of sight to the tracker, or press 's' to "
                    << "skip this pose.\n";
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(NDI_SAMPLE_INTERVAL_MS)
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

    // Diagnostic only: polls BX as fast as NDI_SAMPLE_INTERVAL_MS allows for
    // durationMs, starting the instant this is called (no blind
    // SETTLING_TIME_MS sleep beforehand), logging the single-sample
    // moving-relative-to-fixed pose at every poll -- lets us see the actual
    // settling curve after a move instead of assuming 750ms is enough before
    // the real 30-sample average starts.
    void collectRawSettlingSeries(
        int poseIndex,
        int durationMs,
        bool& manualModeEnabled,
        std::vector<SettlingSample>& out
    ) {
        const auto start = std::chrono::steady_clock::now();

        while (true) {
            const long elapsedMs = std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start
            ).count();
            if (elapsedMs >= durationMs) {
                break;
            }

            handleUserControls(manualModeEnabled);
            requestBxUpdate();

            NdiPoseSample moving;
            NdiPoseSample fixed;
            int movingInvalidCount = 0;
            int fixedInvalidCount = 0;

            const bool movingValid = tryReadToolFromCurrentBx(
                movingToolHandle_, "moving", moving, movingInvalidCount, false
            );
            const bool fixedValid = tryReadToolFromCurrentBx(
                fixedToolHandle_, "fixed", fixed, fixedInvalidCount, false
            );

            SettlingSample sample;
            sample.poseIndex = poseIndex;
            sample.elapsedMs = elapsedMs;

            if (movingValid && fixedValid) {
                const NdiPoseSample relative =
                    computeMovingRelativeToFixed(moving, fixed);
                sample.valid = true;
                sample.txMm = relative.txMm;
                sample.tyMm = relative.tyMm;
                sample.tzMm = relative.tzMm;
                sample.error =
                    (moving.error > fixed.error) ? moving.error : fixed.error;
            }

            out.push_back(sample);

            std::this_thread::sleep_for(
                std::chrono::milliseconds(NDI_SAMPLE_INTERVAL_MS)
            );
        }
    }

    // Waits until the moving tool's position has stopped changing (last
    // STABILITY_WINDOW_SIZE valid samples all within STABILITY_TOLERANCE_MM
    // of each other) instead of assuming a fixed sleep is always enough.
    // Only the moving tool is checked -- the fixed tool never moves, so its
    // stability is irrelevant here. Falls back to proceeding anyway after
    // STABILITY_MAX_WAIT_MS, consistent with this file's "warn, don't
    // abort" philosophy elsewhere (waitForBothToolsVisible/collectBothTools).
    void waitForPoseStability(bool& manualModeEnabled) {
        std::vector<std::array<double, 3>> recent;
        const auto start = std::chrono::steady_clock::now();

        while (true) {
            const long elapsedMs = std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start
            ).count();

            if (elapsedMs >= STABILITY_MAX_WAIT_MS) {
                std::cout
                    << "\nWARNING: moving tool did not stabilize within "
                    << STABILITY_MAX_WAIT_MS
                    << "ms -- proceeding with capture anyway.\n";
                return;
            }

            handleUserControls(manualModeEnabled);
            requestBxUpdate();

            NdiPoseSample moving;
            int invalidCount = 0;
            const bool valid = tryReadToolFromCurrentBx(
                movingToolHandle_, "moving", moving, invalidCount, false
            );

            if (valid) {
                recent.push_back({moving.txMm, moving.tyMm, moving.tzMm});
                if (recent.size() > STABILITY_WINDOW_SIZE) {
                    recent.erase(recent.begin());
                }

                if (recent.size() == STABILITY_WINDOW_SIZE) {
                    double maxSpread = 0.0;
                    for (std::size_t i = 0; i < recent.size(); ++i) {
                        for (std::size_t j = i + 1; j < recent.size(); ++j) {
                            const double dx = recent[i][0] - recent[j][0];
                            const double dy = recent[i][1] - recent[j][1];
                            const double dz = recent[i][2] - recent[j][2];
                            const double dist =
                                std::sqrt(dx * dx + dy * dy + dz * dz);
                            if (dist > maxSpread) {
                                maxSpread = dist;
                            }
                        }
                    }

                    if (maxSpread <= STABILITY_TOLERANCE_MM) {
                        return;
                    }
                }
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(NDI_SAMPLE_INTERVAL_MS)
            );
        }
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

    void waitForBothToolsVisible(bool& manualModeEnabled) {
        // No attempt cap here on purpose -- see the identical note in
        // collectBothTools(). A visibility timeout must never abort the
        // whole run; it just keeps waiting and reminds periodically.
        constexpr int WARNING_INTERVAL_ATTEMPTS = 600;

        std::cout
            << "Waiting for both NDI tools to become visible. Press 'p' "
            << "to pause, 's' to skip this pose, or space to toggle "
            << "manual mode.\n";

        NdiToolStatus lastMovingStatus = NdiToolStatus::Detected;
        NdiToolStatus lastFixedStatus = NdiToolStatus::Detected;
        bool movingStatusPrinted = false;
        bool fixedStatusPrinted = false;

        for (int attempt = 1; ; ++attempt) {
            handleUserControls(manualModeEnabled);

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

            if (attempt > 0 && attempt % WARNING_INTERVAL_ATTEMPTS == 0) {
                std::cout
                    << "\nWARNING: still waiting for both tools to "
                    << "become visible after " << (attempt * 50) / 1000
                    << "s. Check that both tools have a clear, "
                    << "unobstructed line of sight to the tracker, or "
                    << "press 's' to skip this pose.\n";
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(50)
            );
        }
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

// Checks OUTPUT_CSV (if it already exists from a previous, interrupted
// run) for poses already captured, so a fresh run continues from the next
// pose instead of overwriting earlier data. Also sanity-checks that
// TARGET_POSES hasn't changed underneath the already-captured rows.
// Returns the 0-based index of the next pose to attempt (0 if the file
// doesn't exist yet or has no data rows).
std::size_t findResumeStartIndex(const std::string& csvPath) {
    std::ifstream in(csvPath);
    if (!in) {
        return 0;
    }

    std::string line;
    std::getline(in, line);  // header row, discarded

    std::size_t maxPoseId = 0;

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream fields(line);
        std::string field;

        if (!std::getline(fields, field, ',')) {
            continue;
        }
        const std::size_t poseId = static_cast<std::size_t>(std::stoul(field));

        std::getline(fields, field, ',');  // timestamp_ms, discarded

        std::array<uint16_t, JOINT_COUNT> targetTicks{};
        bool parsedOk = true;
        for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
            if (!std::getline(fields, field, ',')) {
                parsedOk = false;
                break;
            }
            targetTicks[i] = static_cast<uint16_t>(std::stoi(field));
        }

        if (parsedOk && poseId >= 1 && poseId <= TARGET_POSES.size()) {
            const auto& expected = TARGET_POSES[poseId - 1];
            if (targetTicks != expected) {
                std::cout
                    << "WARNING: recorded pose " << poseId << " in "
                    << csvPath << " does not match the current "
                    << "TARGET_POSES entry at that index -- the pose "
                    << "list may have changed since that data was "
                    << "captured.\n";
            }
        }

        if (poseId > maxPoseId) {
            maxPoseId = poseId;
        }
    }

    return maxPoseId;
}

// --settling-diagnostic mode: moves through the first
// SETTLING_DIAGNOSTIC_POSE_COUNT poses of TARGET_POSES (already
// hand-verified safe -- same array the real 200-pose capture uses) and,
// instead of the normal SETTLING_TIME_MS sleep + 30-sample average, logs
// every raw BX poll for SETTLING_DIAGNOSTIC_DURATION_MS right after each
// move completes. Lets us see directly whether the tracked pose is still
// moving/settling in the window the real capture loop would otherwise
// silently average over.
int runSettlingDiagnostic() {
    const std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};

    DynamixelMotor motor(
        CYTON_DEVICE,
        CYTON_BAUD_RATE,
        CYTON_PROTOCOL_VERSION
    );

    try {
        std::ofstream csv(SETTLING_DIAGNOSTIC_CSV, std::ios::out | std::ios::trunc);
        if (!csv) {
            throw std::runtime_error(
                std::string("Could not open CSV: ") + SETTLING_DIAGNOSTIC_CSV
            );
        }
        csv << "pose_id,elapsed_ms,valid,tx_mm,ty_mm,tz_mm,error\n";

        if (!motor.connect()) {
            throw std::runtime_error("Could not connect to the Cyton motors.");
        }

        for (int id : motorIds) {
            if (!motor.pingMotor(id)) {
                throw std::runtime_error(
                    "Could not ping motor " + std::to_string(id)
                );
            }
        }

        bool manualModeEnabled = false;

        NdiTracker ndi(NDI_DEVICE, MOVING_TOOL_ROM, FIXED_TOOL_ROM);

        try {
            ndi.initialize(manualModeEnabled);
        } catch (const PoseSkippedByUser&) {
            std::cout
                << "\nSkipped the initial visibility wait. Proceeding -- "
                << "visibility gets re-checked at the start of each pose "
                << "anyway.\n";
        }

        std::cout
            << "\nSettling diagnostic: " << SETTLING_DIAGNOSTIC_POSE_COUNT
            << " poses, " << SETTLING_DIAGNOSTIC_DURATION_MS
            << "ms of raw BX samples logged per pose (no settling sleep).\n"
            << "Output: " << SETTLING_DIAGNOSTIC_CSV << "\n"
            << "'p' pauses/resumes, 's' skips the current pose.\n";

        for (int poseIndex = 0;
             poseIndex < SETTLING_DIAGNOSTIC_POSE_COUNT;
             ++poseIndex) {
            checkForModeToggle(manualModeEnabled);

            std::cout
                << "\nMoving to pose " << poseIndex + 1 << " of "
                << SETTLING_DIAGNOSTIC_POSE_COUNT << "...\n";

            const auto& targetArray = TARGET_POSES[poseIndex];
            const std::vector<uint16_t> targetTicks(
                targetArray.begin(), targetArray.end()
            );

            if (!motor.moveJointsSafely(
                    motorIds,
                    targetTicks,
                    MOVING_SPEED,
                    MOTOR_TOLERANCE_TICKS,
                    MOVE_TIMEOUT_SECONDS,
                    true,
                    STALL_REPEATS_TO_DETECT,
                    false
                )) {
                std::cout
                    << "Warning: pose " << poseIndex + 1
                    << " did not fully reach its target. Logging settling "
                    << "data from the actual position reached.\n";
            }

            try {
                std::vector<SettlingSample> samples;
                ndi.collectRawSettlingSeries(
                    poseIndex,
                    SETTLING_DIAGNOSTIC_DURATION_MS,
                    manualModeEnabled,
                    samples
                );

                for (const auto& s : samples) {
                    csv
                        << s.poseIndex << ','
                        << s.elapsedMs << ','
                        << (s.valid ? 1 : 0) << ','
                        << s.txMm << ','
                        << s.tyMm << ','
                        << s.tzMm << ','
                        << s.error << '\n';
                }
                csv.flush();

                std::cout
                    << "  logged " << samples.size() << " samples ("
                    << std::count_if(
                           samples.begin(), samples.end(),
                           [](const SettlingSample& s) { return s.valid; }
                       )
                    << " valid).\n";
            } catch (const PoseSkippedByUser&) {
                std::cout
                    << "\nPose " << poseIndex + 1
                    << " skipped by user. No settling data recorded for "
                    << "this pose.\n";
            }
        }

        std::cout
            << "\nSettling diagnostic complete. Saved data to "
            << SETTLING_DIAGNOSTIC_CSV << '\n';

        disableAll(motor, motorIds);
        motor.disconnect();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\nERROR: " << error.what() << '\n';
        disableAll(motor, motorIds);
        motor.disconnect();
        return 1;
    }
}

// Real (30-sample-averaged) capture, limited to the first
// QUICK_TEST_POSE_COUNT poses, written to its own CSV. See the constant
// comment above for why this exists.
int runQuickCalibrationTest() {
    const std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};

    DynamixelMotor motor(
        CYTON_DEVICE,
        CYTON_BAUD_RATE,
        CYTON_PROTOCOL_VERSION
    );

    try {
        std::ofstream csv(QUICK_TEST_CSV, std::ios::out | std::ios::trunc);
        if (!csv) {
            throw std::runtime_error(
                std::string("Could not open CSV: ") + QUICK_TEST_CSV
            );
        }
        writeCsvHeader(csv);

        if (!motor.connect()) {
            throw std::runtime_error("Could not connect to the Cyton motors.");
        }

        for (int id : motorIds) {
            if (!motor.pingMotor(id)) {
                throw std::runtime_error(
                    "Could not ping motor " + std::to_string(id)
                );
            }
        }

        bool manualModeEnabled = false;

        NdiTracker ndi(NDI_DEVICE, MOVING_TOOL_ROM, FIXED_TOOL_ROM);

        try {
            ndi.initialize(manualModeEnabled);
        } catch (const PoseSkippedByUser&) {
            std::cout
                << "\nSkipped the initial visibility wait. Proceeding -- "
                << "visibility gets re-checked at the start of each pose "
                << "anyway.\n";
        }

        const std::size_t quickTestPoseCount = QUICK_TEST_POSE_INDICES.size();

        std::cout
            << "\nQuick calibration test: " << quickTestPoseCount
            << " diverse poses, full real capture pipeline (stability-based "
            << "settle + 30-sample average per tool), collected in one "
            << "short sitting.\n"
            << "Output: " << QUICK_TEST_CSV << "\n"
            << "Poses auto-advance. 'p' pauses/resumes, 's' skips the "
            << "current pose, space toggles manual mode.\n";

        for (std::size_t i = 0; i < quickTestPoseCount; ++i) {
            const std::size_t poseIndex =
                static_cast<std::size_t>(QUICK_TEST_POSE_INDICES[i]);

            checkForModeToggle(manualModeEnabled);

            if (manualModeEnabled) {
                waitForEnter(
                    "\nPress Enter to move to pose " +
                    std::to_string(i + 1) + " of " +
                    std::to_string(quickTestPoseCount) + " (TARGET_POSES[" +
                    std::to_string(poseIndex) + "])..."
                );
            } else {
                std::cout
                    << "\nAuto-advancing to pose " << i + 1
                    << " of " << quickTestPoseCount << " (TARGET_POSES["
                    << poseIndex << "])...\n";
            }

            const auto& targetArray = TARGET_POSES[poseIndex];
            const std::vector<uint16_t> targetTicks(
                targetArray.begin(), targetArray.end()
            );

            if (!motor.moveJointsSafely(
                    motorIds,
                    targetTicks,
                    MOVING_SPEED,
                    MOTOR_TOLERANCE_TICKS,
                    MOVE_TIMEOUT_SECONDS,
                    true,
                    STALL_REPEATS_TO_DETECT,
                    false
                )) {
                std::cout
                    << "\nWarning: pose " << i + 1
                    << " did not fully reach its target. Continuing with "
                    << "the actual position reached.\n";

                for (int id : motorIds) {
                    if (!motor.enableTorque(id)) {
                        std::cerr
                            << "WARNING: could not confirm torque is "
                            << "enabled on motor " << id << ".\n";
                    }
                }
            }

            try {
                ndi.waitForPoseStability(manualModeEnabled);

                const std::vector<uint16_t> actualTicks =
                    readActualTicks(motor, motorIds);
                const std::vector<double> actualRadians =
                    ticksToRadiansVector(actualTicks);
                const DualToolCapture ndiCapture =
                    ndi.collectBothTools(manualModeEnabled);

                appendCsvRow(
                    csv,
                    poseIndex,
                    targetArray,
                    actualTicks,
                    actualRadians,
                    ndiCapture
                );
                printCapturedPose(
                    poseIndex, actualTicks, actualRadians, ndiCapture
                );
            } catch (const PoseSkippedByUser&) {
                std::cout
                    << "\nPose " << i + 1
                    << " skipped by user. No NDI data recorded for this "
                    << "pose.\n";
            }
        }

        std::cout
            << "\nQuick calibration test complete. Saved data to "
            << QUICK_TEST_CSV << '\n';

        disableAll(motor, motorIds);
        motor.disconnect();
        return 0;
    } catch (const PoseSkippedByUser&) {
        std::cout
            << "\nSkipped during initial NDI setup (before any pose was "
            << "captured). Exiting.\n";
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

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--settling-diagnostic") {
        return runSettlingDiagnostic();
    }
    if (argc > 1 && std::string(argv[1]) == "--quick-test") {
        return runQuickCalibrationTest();
    }

    const std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};

    DynamixelMotor motor(
        CYTON_DEVICE,
        CYTON_BAUD_RATE,
        CYTON_PROTOCOL_VERSION
    );

    try {
        const std::size_t resumeStartIndex =
            findResumeStartIndex(OUTPUT_CSV);
        const bool resuming = resumeStartIndex > 0;

        std::ofstream csv(
            OUTPUT_CSV,
            std::ios::out | (resuming ? std::ios::app : std::ios::trunc)
        );
        if (!csv) {
            throw std::runtime_error(
                std::string("Could not open CSV: ") + OUTPUT_CSV
            );
        }
        if (!resuming) {
            writeCsvHeader(csv);
        } else {
            std::cout
                << "Resuming: " << resumeStartIndex << " pose(s) already "
                << "captured in " << OUTPUT_CSV << ". Continuing from "
                << "pose " << resumeStartIndex + 1 << ".\n";
        }

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

        bool manualModeEnabled = false;

        NdiTracker ndi(
            NDI_DEVICE,
            MOVING_TOOL_ROM,
            FIXED_TOOL_ROM
        );

        try {
            ndi.initialize(manualModeEnabled);
        } catch (const PoseSkippedByUser&) {
            std::cout
                << "\nSkipped the initial visibility wait. Proceeding to "
                << "poses -- visibility gets re-checked at the start of "
                << "each pose anyway.\n";
        }

        std::cout
            << "\nFive-pose Cyton + dual-tool NDI capture test\n"
            << "Output: " << OUTPUT_CSV << "\n"
            << "Moving marker: rigid body 2\n"
            << "Fixed marker: rigid body 3\n"
            << "Poses auto-advance by default -- no Enter needed.\n"
            << "At any time: 'p' pauses/resumes, 's' skips the current "
            << "pose, and space toggles manual mode (requires Enter "
            << "before each future pose until toggled off again).\n"
            << "Press Ctrl+C or use your hardware emergency stop "
            << "if needed.\n";

        for (std::size_t poseIndex = resumeStartIndex;
             poseIndex < TARGET_POSES.size();
             ++poseIndex) {
            checkForModeToggle(manualModeEnabled);

            if (manualModeEnabled) {
                waitForEnter(
                    "\nPress Enter to move to pose " +
                    std::to_string(poseIndex + 1) + " of " +
                    std::to_string(POSE_COUNT) + "..."
                );
            } else {
                std::cout
                    << "\nAuto-advancing to pose " << poseIndex + 1
                    << " of " << POSE_COUNT << "...\n";
            }

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
                    false
                )) {
                std::cout
                    << "\nWarning: pose " << poseIndex + 1
                    << " did not fully reach its target (see the message "
                    << "above for why). Continuing with the actual "
                    << "position reached.\n";

                for (int id : motorIds) {
                    if (!motor.enableTorque(id)) {
                        std::cerr
                            << "WARNING: could not confirm torque is "
                            << "enabled on motor " << id << " -- it may "
                            << "be holding position on its own or may "
                            << "have gone limp. Check the arm physically "
                            << "before continuing.\n";
                    }
                }
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(SETTLING_TIME_MS)
            );

            try {
                const std::vector<uint16_t> actualTicks =
                    readActualTicks(motor, motorIds);

                const std::vector<double> actualRadians =
                    ticksToRadiansVector(actualTicks);

                const DualToolCapture ndiCapture =
                    ndi.collectBothTools(manualModeEnabled);

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