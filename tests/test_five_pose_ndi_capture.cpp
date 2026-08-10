#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <conio.h>
#else
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#endif

#include "dynamixel_motor.h"
#include "robot_calibration.h"
#include "ndicapi.h"

namespace {

#ifndef _WIN32
// Linux stand-ins for Windows' <conio.h> _kbhit()/_getch(), ported 2026-08-07
// so this file can build on the Linux dev machine. Puts stdin into raw,
// non-blocking mode (no line buffering, no echo) for the process lifetime --
// constructed once at the top of main() and restored via RAII on the way
// out. Same non-blocking single-key-peek semantics as the Windows originals:
// _kbhit() returns whether a key is waiting without consuming it, _getch()
// consumes and returns it.
class LinuxRawStdin {
public:
    LinuxRawStdin() {
        if (tcgetattr(STDIN_FILENO, &oldTermios_) == 0) {
            valid_ = true;
            termios raw = oldTermios_;
            raw.c_lflag &= ~(ICANON | ECHO);
            raw.c_cc[VMIN] = 0;
            raw.c_cc[VTIME] = 0;
            tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        }
        const int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        if (flags != -1) {
            fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
        }
    }

    ~LinuxRawStdin() {
        if (valid_) {
            tcsetattr(STDIN_FILENO, TCSANOW, &oldTermios_);
        }
    }

    LinuxRawStdin(const LinuxRawStdin&) = delete;
    LinuxRawStdin& operator=(const LinuxRawStdin&) = delete;

private:
    termios oldTermios_{};
    bool valid_ = false;
};

bool g_hasPendingChar = false;
unsigned char g_pendingChar = 0;

int _kbhit() {
    if (g_hasPendingChar) {
        return 1;
    }
    unsigned char ch = 0;
    const ssize_t n = read(STDIN_FILENO, &ch, 1);
    if (n == 1) {
        g_pendingChar = ch;
        g_hasPendingChar = true;
        return 1;
    }
    return 0;
}

int _getch() {
    if (g_hasPendingChar) {
        g_hasPendingChar = false;
        return g_pendingChar;
    }
    unsigned char ch = 0;
    const ssize_t n = read(STDIN_FILENO, &ch, 1);
    return n == 1 ? ch : -1;
}
#endif  // !_WIN32

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
constexpr std::size_t POSE_COUNT = 1619;

// Ported to Linux 2026-08-07 (see CLAUDE.md) -- device paths and ROM paths
// updated to this machine's actual values (matches cyton_ndi_capture's/
// cyton_hardware's already-established assignment: arm on ttyUSB0, tracker
// on ttyUSB1).
constexpr const char* CYTON_DEVICE = "/dev/ttyUSB0";
constexpr int CYTON_BAUD_RATE = 1000000;
constexpr float CYTON_PROTOCOL_VERSION = 1.0F;

constexpr const char* NDI_DEVICE = "/dev/ttyUSB1";

constexpr const char* MOVING_TOOL_ROM =
    "/home/temp/Downloads/8700339- Polaris Passive 4-Marker Rigid Body 2(1).rom";

constexpr const char* FIXED_TOOL_ROM =
    "/home/temp/Downloads/8700449- Polaris Passive 4-Marker Rigid Body 3(1).rom";

constexpr const char* OUTPUT_CSV = "five_pose_ndi_capture.csv";

// Restored to 40 (2026-07-30) for the normal/production 51-pose backlash-
// margin re-test -- it had been dropped to 15 (2026-07-29) specifically for
// the IK round-trip validation test (poses generated by numerically
// inverting the fitted calibration model, never physically hand-verified
// the way every other pose in this file was), which needed extra caution
// for genuinely new, unverified joint combinations. That caution doesn't
// apply to the hand-verified 51-pose set this run is using.
constexpr uint16_t MOVING_SPEED = 40;
constexpr int MOTOR_TOLERANCE_TICKS = 5;
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
// collectBothTools()) as the normal 291-pose run, but in one short sitting
// -- to test whether collecting the dataset without the multi-hour
// session-length drift found in the original (superseded) 200-pose data
// gets the fitted calibration error down near the arm's ~0.5mm repeatability
// floor. Writes to its own CSV so it never touches five_pose_ndi_capture.csv
// or its resume state.
//
// QUICK_TEST_POSE_INDICES currently targets a 51-pose evenly-spaced subset
// (every 6th index) of the ORIGINAL 0-303 calibration pose range -- NOT
// the 304-307 backlash-test poses. Purpose (2026-07-29): validate the
// dynamixel_motor.cpp backlash-compensation fix (see CLAUDE.md's
// kinematic-calibration section) via a direct, apples-to-apples RMS
// comparison against the existing calibration datasets -- same poses,
// same fitting methodology, the only difference being that
// moveJointsSafely() now compensates for backlash by default during this
// recollection. If the fix is working, refitting the current best model
// on this new data should show a real RMS drop versus the ~8.257mm
// baseline on quick_calibration_test_combined_298.csv. Evenly spacing by
// index (rather than taking a contiguous block) preserves the original
// dataset's joint-range diversity in a smaller, faster-to-collect sample.
// Real-world IK round-trip validation (2026-07-29): poses 316-318, see
// the comment on those TARGET_POSES entries below. MOVING_SPEED reduced
// to 15 for this run (see that constant). Change back to covering the
// full TARGET_POSES range, the 51-pose backlash-fix validation subset, or
// a single-joint backlash-test pose set for other purposes.
// Restored 2026-07-30 to the 51-pose evenly-spaced subset (every 6th index
// of the original 0-303 calibration pose range) for re-testing the
// BACKLASH_OVERSHOOT_TICKS margin (20->35) against the same old-vs-new
// comparison methodology used to validate the original fix -- this had been
// overwritten to {316,317,318} for the separate IK round-trip experiment.
// Restored 2026-07-31 (again) to target the realm-restricted calibration
// poses (TARGET_POSES index 354-482, 129 hand-posed configurations within
// the user's actual reduced operating envelope -- see CLAUDE.md's
// kinematic-calibration section). Change back to the 51-pose evenly-spaced
// subset (0,6,12,...,300) for the backlash-margin comparison, TARGET_POSES
// index 319-346 for the multi-joint backlash NDI capture, 347-353 for the
// displacement-from-home test, or {316,317,318} for the IK round-trip
// experiment.
// Repointed (2026-07-31) to test ONLY the new recorded_hand_poses (3).csv
// batch (index 657-856), in isolation from every prior quick-test dataset.
// QUICK_TEST_CSV below was also changed to a new filename so this run
// cannot overwrite the existing 218-pose quick_calibration_test.csv.
// Previous quick-test pose sets (realm-restricted 354-482, widened-range
// 483-656) are preserved in TARGET_POSES above and can be restored here
// if needed again later.
// Repointed (2026-08-11) to test ONLY the genuinely new poses from the
// ongoing recording session on recorded_hand_poses_fixed_elbow_yaw(3).csv
// (index 1398-1618, 221 poses -- rows 34-254 of that file; extends the
// earlier 141-pose version, 80 more poses added in a later sitting -- same
// session, legitimately resumed this time). The file's OUTPUT_CSV path is
// hardcoded in record_hand_poses_fixed_elbow_yaw.cpp, so rows 1-33 are a
// genuinely unrelated 2026-08-07 session that got silently appended to by
// the 2026-08-10 session via the tool's own resume-detection logic --
// caught when the user correctly didn't recognize that data as what they'd
// just recorded. Those old 33 rows remain excluded here. The standalone
// old-33 entry is NOT separately preserved elsewhere in TARGET_POSES; it
// would need to be re-added from that file's first 33 rows if ever wanted
// again. Every prior quick-test pose set (realm-restricted 354-482,
// widened-range 483-656, batch3 657-856, first elbow_yaw-fixed batch
// 857-1238, second elbow_yaw-fixed batch 1239-1397) is preserved in
// TARGET_POSES and can be restored here if needed again.
constexpr std::array<int, 221> QUICK_TEST_POSE_INDICES = {
    1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411,
    1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425,
    1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439,
    1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453,
    1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467,
    1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481,
    1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495,
    1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509,
    1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523,
    1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537,
    1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551,
    1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565,
    1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579,
    1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593,
    1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607,
    1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618,
};
constexpr const char* QUICK_TEST_CSV = "quick_calibration_test_fixed_elbow_yaw_batch3_newonly.csv";

// Hand-recorded via tests/record_hand_poses.cpp: the arm was physically
// moved by hand (torque off) into each pose and the resulting joint ticks
// captured, so every pose here is a real, human-verified configuration --
// not randomly sampled -- meaning it's known in advance to be physically
// reachable, collision-free, and (assuming it was checked live against the
// "Moving tool: DETECTED/MISSING" status while posing) marker-visible.
const std::array<std::array<uint16_t, JOINT_COUNT>, POSE_COUNT> TARGET_POSES = {{
    {{2706, 3224, 931, 2200, 956, 1172, 3368}},
    {{2328, 3224, 932, 2192, 955, 1116, 3604}},
    {{2247, 3224, 1108, 2191, 955, 1096, 3754}},
    {{2194, 3224, 1224, 2182, 956, 1082, 3754}},
    {{1954, 3222, 1331, 2125, 955, 1216, 412}},
    {{1569, 3225, 1199, 2114, 952, 1243, 415}},
    {{1196, 3225, 1203, 2064, 956, 1252, 911}},
    {{933, 3226, 1201, 2074, 955, 1186, 912}},
    {{722, 3226, 1201, 2115, 950, 1275, 1291}},
    {{2866, 3231, 930, 1004, 1844, 2084, 337}},
    {{2584, 3224, 931, 926, 1981, 2162, 395}},
    {{2271, 3226, 931, 877, 1988, 2257, 575}},
    {{1908, 3225, 931, 830, 1943, 2314, 1120}},
    {{1610, 3223, 931, 831, 2031, 2293, 1463}},
    {{1092, 3225, 937, 1040, 1878, 2066, 1935}},
    {{766, 3227, 937, 957, 1897, 2150, 2224}},
    {{756, 3227, 937, 829, 1932, 2259, 2189}},
    {{829, 3228, 936, 831, 1117, 2149, 2188}},
    {{1384, 3221, 934, 956, 1043, 2555, 2029}},
    {{1812, 3222, 935, 1139, 1127, 2948, 2027}},
    {{2139, 3223, 933, 1242, 1124, 3178, 2030}},
    {{2357, 3222, 928, 1226, 1066, 3224, 2143}},
    {{2384, 3225, 936, 1159, 1129, 1414, 337}},
    {{2384, 3226, 1137, 1223, 1271, 1126, 338}},
    {{1838, 3228, 1138, 1318, 1365, 882, 568}},
    {{1386, 3226, 932, 1274, 1517, 755, 428}},
    {{655, 3226, 934, 1071, 2096, 1914, 2043}},
    {{625, 3227, 936, 1032, 2240, 2061, 2039}},
    {{627, 3230, 934, 989, 2573, 2081, 2040}},
    {{922, 3230, 1407, 829, 2149, 2369, 2451}},
    {{946, 3019, 1405, 1035, 1957, 2118, 2418}},
    {{1091, 3007, 1290, 981, 1856, 2173, 2173}},
    {{1102, 3095, 1241, 897, 1634, 2224, 2148}},
    {{1219, 3103, 1045, 897, 1455, 2441, 2079}},
    {{1374, 3081, 998, 1024, 1763, 2366, 1638}},
    {{1399, 2865, 996, 1070, 2092, 2201, 1522}},
    {{1494, 2842, 924, 1233, 1937, 2102, 1520}},
    {{1527, 2849, 924, 1460, 1905, 1881, 1519}},
    {{1526, 3140, 924, 1399, 1739, 1869, 1399}},
    {{1676, 3222, 925, 1347, 1617, 2416, 1400}},
    {{1944, 3222, 927, 1337, 1515, 2854, 1621}},
    {{2115, 3221, 929, 1353, 1348, 3102, 1823}},
    {{2300, 3222, 927, 1424, 1205, 3186, 2266}},
    {{2361, 3075, 925, 1821, 962, 2869, 2612}},
    {{2307, 2139, 922, 1283, 954, 1208, 2903}},
    {{2200, 2141, 929, 1371, 954, 967, 2907}},
    {{2167, 2138, 927, 1661, 963, 872, 2907}},
    {{2247, 3230, 931, 2219, 1072, 1041, 3689}},
    {{2179, 3224, 931, 1263, 1942, 1929, 871}},
    {{2178, 2853, 932, 1051, 2256, 2100, 874}},
    {{2179, 2678, 931, 1057, 2520, 2104, 942}},
    {{2241, 2218, 1006, 1055, 2874, 2077, 851}},
    {{2203, 2268, 1233, 1733, 2790, 1968, 1437}},
    {{2189, 2288, 1458, 1812, 2613, 1932, 1581}},
    {{2188, 2419, 1611, 1843, 2397, 1868, 1583}},
    {{2188, 2571, 1644, 1846, 2395, 1749, 1582}},
    {{2061, 2543, 1841, 1688, 2365, 1926, 1730}},
    {{1905, 2554, 1986, 1687, 2256, 1920, 2020}},
    {{1748, 2629, 2100, 1686, 2212, 1853, 2187}},
    {{1739, 2872, 2109, 1375, 2212, 1920, 2280}},
    {{1740, 2976, 2106, 1029, 1963, 2114, 2277}},
    {{1852, 2982, 2103, 911, 1829, 2220, 2277}},
    {{2256, 3155, 2082, 829, 1720, 2308, 2040}},
    {{2324, 3229, 2081, 829, 1491, 2332, 2041}},
    {{2370, 3106, 2081, 829, 1951, 2228, 2155}},
    {{2368, 2895, 2083, 888, 2023, 2380, 1423}},
    {{2368, 2456, 2084, 1145, 2031, 2471, 2050}},
    {{2368, 2341, 2110, 1259, 2031, 2438, 1695}},
    {{2369, 2112, 2113, 1632, 2025, 2369, 1508}},
    {{2369, 2127, 2051, 1897, 2026, 2110, 1936}},
    {{2367, 2142, 2031, 2037, 2024, 1987, 1346}},
    {{2368, 1839, 2070, 1945, 2025, 2445, 1387}},
    {{2339, 1464, 1902, 1908, 2092, 2903, 1699}},
    {{2023, 1437, 1863, 2057, 2106, 2824, 2010}},
    {{1739, 1387, 1802, 2172, 2124, 2674, 2525}},
    {{1238, 1240, 1685, 2601, 2245, 2418, 3197}},
    {{1078, 1191, 1578, 2876, 2194, 2192, 3528}},
    {{1065, 1016, 1545, 3128, 1826, 2138, 3586}},
    {{1041, 877, 1549, 3081, 2053, 2081, 3588}},
    {{1043, 1102, 1548, 3025, 2255, 1924, 3572}},
    {{1042, 1331, 1595, 2757, 2403, 1925, 3452}},
    {{1027, 1181, 1600, 2598, 2282, 2267, 3456}},
    {{768, 1161, 1601, 2601, 2285, 2327, 3604}},
    {{592, 1094, 1561, 2805, 2279, 2237, 3755}},
    {{571, 1300, 1559, 2986, 2051, 2016, 3754}},
    {{566, 851, 1559, 3184, 2040, 2221, 336}},
    {{563, 1136, 1561, 3180, 2036, 1907, 3755}},
    {{561, 1333, 1560, 3066, 2074, 1941, 3755}},
    {{561, 1511, 1558, 3066, 1944, 2020, 3754}},
    {{561, 1512, 1539, 3156, 1508, 2038, 3754}},
    {{543, 1752, 1541, 3148, 1523, 2180, 3754}},
    {{543, 1837, 1451, 3273, 1424, 2158, 3583}},
    {{542, 1843, 1191, 3272, 1171, 2093, 3582}},
    {{542, 1837, 1179, 3270, 1538, 1740, 3585}},
    {{542, 1843, 1179, 3069, 2141, 1808, 3736}},
    {{545, 1828, 1196, 2758, 2286, 1832, 3755}},
    {{533, 1541, 1078, 2791, 2336, 2114, 421}},
    {{533, 1434, 1165, 2797, 2343, 2136, 338}},
    {{536, 1260, 1175, 2983, 2355, 2153, 395}},
    {{560, 1120, 1187, 3198, 2357, 2063, 503}},
    {{637, 852, 1188, 3247, 2389, 2095, 347}},
    {{964, 852, 1188, 3236, 2652, 2066, 344}},
    {{1268, 852, 1193, 3272, 2791, 1767, 337}},
    {{1379, 851, 1194, 3272, 3109, 2272, 337}},
    {{1363, 852, 1233, 3270, 2798, 2034, 337}},
    {{872, 1355, 1758, 2662, 2123, 1720, 3008}},
    {{821, 1133, 1759, 2680, 2145, 2137, 3402}},
    {{741, 1118, 1756, 2866, 1977, 2241, 3658}},
    {{568, 1336, 1754, 2893, 2021, 2083, 3660}},
    {{370, 1131, 1688, 2893, 1864, 2142, 3755}},
    {{347, 894, 1688, 3189, 1807, 2140, 3753}},
    {{274, 994, 1662, 3040, 1910, 2190, 335}},
    {{274, 1127, 1396, 2965, 1911, 2113, 338}},
    {{274, 1397, 1385, 2931, 2032, 2037, 338}},
    {{274, 1752, 1389, 2753, 2454, 2005, 335}},
    {{276, 1740, 1050, 2808, 2459, 2290, 513}},
    {{275, 1827, 1034, 3065, 2537, 2131, 784}},
    {{275, 1836, 919, 3147, 2685, 2262, 907}},
    {{275, 2471, 986, 2512, 3249, 1938, 335}},
    {{277, 3223, 933, 2109, 3218, 915, 3754}},
    {{278, 3228, 1305, 2120, 3214, 951, 3754}},
    {{278, 2553, 1301, 2027, 2960, 1464, 3754}},
    {{278, 2355, 1362, 1965, 2958, 1746, 3753}},
    {{278, 2107, 1445, 1916, 2897, 1998, 3569}},
    {{278, 1836, 1437, 1904, 2834, 2332, 3512}},
    {{283, 1610, 1440, 1885, 3039, 2493, 3453}},
    {{512, 1560, 1548, 1874, 3039, 2439, 3254}},
    {{882, 1356, 1592, 1812, 3038, 2820, 2819}},
    {{1087, 1284, 1636, 1814, 3040, 2843, 2685}},
    {{1230, 1084, 1663, 1900, 3039, 3049, 2633}},
    {{1457, 899, 1661, 2109, 3037, 3291, 2118}},
    {{1714, 853, 1662, 2539, 3037, 2968, 1733}},
    {{1805, 851, 1663, 2886, 3046, 2547, 1884}},
    {{1807, 1391, 1661, 3064, 3180, 2386, 2231}},
    {{1739, 1694, 1660, 3042, 3249, 2851, 2468}},
    {{1524, 1909, 1660, 2890, 3234, 2836, 2374}},
    {{1525, 1908, 1533, 2642, 3034, 3036, 2668}},
    {{1011, 1982, 1780, 1335, 1932, 1066, 2821}},
    {{826, 1881, 1780, 1360, 2021, 1093, 2819}},
    {{683, 1789, 1922, 1252, 2350, 1061, 2817}},
    {{770, 1768, 2119, 1155, 2344, 1205, 2818}},
    {{652, 1626, 2127, 1153, 2229, 1252, 2817}},
    {{364, 1669, 2127, 1202, 2413, 1192, 2816}},
    {{647, 1647, 2233, 1218, 2563, 1321, 2816}},
    {{923, 1589, 2281, 1092, 2369, 1506, 2817}},
    {{725, 1611, 2274, 1104, 2348, 1532, 2816}},
    {{540, 1769, 2268, 1121, 2346, 1368, 2816}},
    {{486, 2081, 2252, 1053, 2346, 995, 2816}},
    {{347, 2216, 1801, 1206, 2116, 1006, 2816}},
    {{347, 2431, 1531, 1013, 2121, 1682, 2814}},
    {{348, 2431, 1531, 1192, 2132, 1831, 2820}},
    {{348, 2512, 1494, 1270, 2270, 1788, 2974}},
    {{347, 2697, 1416, 1292, 2338, 1796, 3031}},
    {{353, 3008, 1377, 1264, 2248, 1885, 3115}},
    {{423, 3010, 1269, 1163, 1973, 1746, 2777}},
    {{441, 3146, 1076, 887, 1838, 2020, 2669}},
    {{441, 3111, 1077, 1202, 1993, 1842, 2623}},
    {{656, 3106, 1075, 1211, 1888, 1827, 2349}},
    {{689, 3232, 1079, 1320, 1628, 1570, 2179}},
    {{647, 3228, 1078, 829, 1802, 2156, 2435}},
    {{749, 3223, 1075, 881, 1814, 2227, 2237}},
    {{901, 3014, 1075, 1023, 1773, 2099, 2208}},
    {{908, 2971, 1074, 1018, 1488, 2079, 2134}},
    {{1232, 2966, 940, 1134, 1987, 1931, 1676}},
    {{1393, 2761, 949, 1089, 2243, 1964, 1667}},
    {{1393, 2138, 2029, 1986, 2027, 1926, 2516}},
    {{1393, 2127, 1912, 1979, 2036, 1925, 2515}},
    {{1393, 2110, 1784, 1923, 2370, 2009, 2527}},
    {{1393, 2059, 1669, 1871, 2481, 2066, 2525}},
    {{1391, 2128, 1992, 2042, 2277, 1934, 2527}},
    {{1391, 2206, 2211, 2059, 2018, 1705, 2748}},
    {{1380, 2273, 2333, 2036, 1930, 1604, 2859}},
    {{1293, 2271, 2460, 1994, 1721, 1670, 2854}},
    {{1274, 2250, 2733, 1872, 1432, 1776, 2972}},
    {{1255, 2059, 2630, 1809, 1457, 2070, 2991}},
    {{1231, 1998, 2370, 1795, 1590, 2083, 2856}},
    {{1208, 1947, 2157, 1798, 2010, 2280, 2842}},
    {{1132, 1946, 1928, 1847, 2298, 2267, 2838}},
    {{1110, 1868, 1762, 1848, 2549, 2359, 2835}},
    {{1092, 1737, 1420, 1844, 3031, 2430, 2737}},
    {{1090, 1595, 1311, 1820, 3215, 2460, 2695}},
    {{1090, 1392, 1303, 1826, 3246, 2708, 2755}},
    {{1252, 1090, 1306, 2095, 3248, 3004, 2671}},
    {{1255, 1422, 1305, 2111, 3249, 2719, 2735}},
    {{1255, 1727, 1370, 2195, 3245, 2576, 2735}},
    {{1254, 1874, 1496, 2199, 2905, 2194, 2883}},
    {{1253, 2031, 1753, 2206, 2554, 1911, 2951}},
    {{1253, 2036, 1983, 2205, 2288, 1834, 2949}},
    {{1253, 2047, 2121, 2204, 2128, 1741, 2893}},
    {{1253, 2074, 2403, 2203, 2011, 1714, 2931}},
    {{1252, 2077, 2696, 2203, 1581, 1718, 2931}},
    {{1254, 2078, 2939, 2203, 1312, 1857, 2932}},
    {{1190, 1931, 3019, 2103, 1247, 2048, 2930}},
    {{1148, 1828, 3106, 2057, 1105, 2192, 2908}},
    {{1048, 1429, 3108, 2072, 953, 2443, 2927}},
    {{918, 1270, 3109, 2091, 953, 2655, 3193}},
    {{785, 1120, 3100, 2090, 955, 2766, 3426}},
    {{587, 1037, 3096, 2076, 955, 2852, 3670}},
    {{337, 1029, 3094, 2091, 1007, 2864, 3754}},
    {{336, 1186, 3090, 2232, 1177, 2747, 3752}},
    {{283, 1648, 2727, 2258, 1526, 2388, 3636}},
    {{283, 1763, 2463, 2345, 1697, 2211, 3550}},
    {{283, 1844, 2317, 2490, 1809, 1999, 3549}},
    {{285, 1813, 2066, 2903, 1811, 1638, 3648}},
    {{285, 1661, 1875, 3192, 1812, 1375, 3752}},
    {{284, 1519, 1636, 3178, 2198, 1777, 335}},
    {{284, 1738, 1606, 3102, 2240, 1611, 336}},
    {{283, 1861, 1373, 2794, 2671, 2014, 336}},
    {{284, 1980, 1942, 3276, 1590, 2901, 2862}},
    {{616, 2047, 1934, 3269, 1659, 2932, 3221}},
    {{617, 2037, 1932, 3244, 1353, 2904, 3585}},
    {{861, 2038, 1731, 3236, 1332, 2591, 3714}},
    {{1954, 1929, 1840, 3121, 956, 2513, 957}},
    {{2372, 1930, 1804, 3224, 1002, 2627, 1247}},
    {{2629, 2302, 1718, 2998, 1222, 2715, 1460}},
    {{2707, 2483, 1444, 2890, 1350, 2302, 1769}},
    {{3129, 2367, 2460, 2140, 1546, 1731, 1035}},
    {{3128, 2534, 2500, 2141, 1528, 1549, 1039}},
    {{3128, 2680, 2499, 2141, 1526, 1414, 1038}},
    {{3148, 2836, 2494, 2137, 1425, 1274, 1035}},
    {{3274, 2957, 2469, 2117, 1393, 1226, 929}},
    {{3416, 3015, 2438, 2115, 1377, 1194, 680}},
    {{3594, 3076, 2437, 2043, 1373, 1123, 426}},
    {{3612, 2726, 2439, 1905, 1377, 1493, 415}},
    {{3640, 2547, 2435, 1875, 1608, 1892, 608}},
    {{3645, 2411, 2409, 1688, 1615, 2381, 914}},
    {{3645, 2321, 2254, 1388, 1616, 2784, 916}},
    {{3645, 2324, 2178, 1170, 1614, 2992, 999}},
    {{3641, 2353, 2028, 1211, 1795, 2852, 856}},
    {{3599, 2051, 1140, 1942, 2919, 2095, 337}},
    {{3578, 2001, 1010, 1946, 3058, 2286, 583}},
    {{3550, 1461, 1022, 2316, 3056, 2798, 582}},
    {{3254, 1018, 1006, 2294, 3056, 3040, 796}},
    {{3243, 910, 1006, 2092, 3055, 3139, 797}},
    {{3575, 851, 1104, 2218, 3056, 3279, 531}},
    {{3618, 888, 1135, 2119, 3052, 3235, 336}},
    {{3615, 1094, 1136, 2267, 3043, 3096, 337}},
    {{3614, 1289, 1136, 2446, 3012, 2935, 337}},
    {{3617, 1575, 1139, 2845, 2674, 2586, 869}},
    {{3615, 1592, 1520, 3121, 2512, 2257, 984}},
    {{3639, 1711, 2214, 2884, 2411, 2006, 338}},
    {{3636, 1894, 2217, 2897, 2659, 2287, 706}},
    {{3631, 2090, 2251, 2897, 2752, 2611, 680}},
    {{3628, 2309, 2259, 2874, 2742, 2525, 779}},
    {{3630, 2511, 2260, 2865, 2725, 3095, 778}},
    {{3633, 2528, 2257, 2560, 2683, 2562, 778}},
    {{3605, 2654, 2260, 2413, 2501, 1589, 934}},
    {{3416, 2673, 2260, 2094, 2337, 1525, 932}},
    {{3077, 2650, 2262, 1774, 2001, 1890, 935}},
    {{3069, 2515, 2262, 1589, 1952, 2070, 1015}},
    {{3069, 2318, 2255, 1469, 1873, 2207, 975}},
    {{3069, 2138, 2210, 1450, 1866, 2411, 968}},
    {{3069, 1899, 2156, 1405, 1860, 2609, 902}},
    {{2949, 1760, 2004, 1395, 1861, 3003, 1304}},
    {{2750, 1705, 1843, 1421, 1861, 3122, 1644}},
    {{2642, 1660, 1662, 1549, 1865, 3171, 1905}},
    {{2657, 1883, 1665, 1929, 1858, 2861, 1795}},
    {{2657, 1993, 1513, 2096, 1858, 2701, 1852}},
    {{2654, 2059, 1280, 2162, 1859, 2551, 2064}},
    {{2653, 2141, 1108, 2205, 1857, 2454, 2064}},
    {{2641, 2143, 920, 2357, 2426, 2198, 1930}},
    {{2596, 2143, 930, 2410, 2788, 2144, 1930}},
    {{2523, 2143, 1121, 2407, 2889, 1971, 1925}},
    {{2520, 2145, 1347, 2410, 2639, 1821, 1741}},
    {{2365, 2341, 1366, 2407, 2634, 1604, 1762}},
    {{2322, 2422, 1526, 2400, 2622, 1570, 1856}},
    {{2235, 2336, 1665, 2240, 2685, 1492, 1948}},
    {{2234, 2119, 1712, 2133, 2684, 1747, 1887}},
    {{2234, 1996, 1959, 2113, 2256, 1960, 1817}},
    {{2235, 1984, 2008, 2115, 2103, 1985, 1821}},
    {{2236, 1754, 2078, 2138, 1881, 2251, 1865}},
    {{2226, 1676, 2043, 2139, 1871, 2310, 1992}},
    {{2242, 1576, 2077, 2147, 1873, 2416, 1990}},
    {{2249, 1529, 2180, 2148, 1872, 2439, 1924}},
    {{2310, 1438, 2276, 2149, 1871, 2574, 1749}},
    {{2326, 1483, 2280, 2428, 1792, 2165, 1656}},
    {{2310, 1488, 2214, 2678, 1742, 2019, 1657}},
    {{2214, 1490, 2140, 2900, 1742, 1775, 1660}},
    {{1978, 1483, 1953, 2873, 1807, 1754, 2082}},
    {{1793, 1449, 1922, 2778, 2079, 1821, 2425}},
    {{1645, 1308, 1877, 2726, 2101, 2127, 2684}},
    {{1588, 1142, 1875, 2723, 2101, 2296, 2779}},
    {{1382, 1104, 1875, 2727, 2103, 2334, 3007}},
    {{1055, 1057, 1873, 2802, 2133, 2312, 3354}},
    {{1059, 1094, 1873, 2997, 2133, 2016, 3354}},
    {{1056, 1282, 1874, 3026, 2210, 1810, 3349}},
    {{906, 1109, 1557, 3093, 2153, 2117, 3753}},
    {{896, 851, 1341, 3192, 1719, 2169, 3753}},
    {{835, 852, 1236, 3192, 1360, 2190, 3754}},
    {{430, 851, 1266, 3273, 1880, 2068, 348}},
    {{414, 1036, 1263, 3236, 1939, 1891, 481}},
    // Poses 291-303 (0-based): 13 new hand-recorded poses specifically
    // targeting ORIENTATION diversity (not just joint-range coverage) --
    // varying wrist_pitch/wrist_roll in combination rather than each joint
    // individually. Independently verified (calibration/
    // diag_check_new13_orientation.py) to have nearest-neighbor orientation
    // distance ~24.3 deg vs ~8.3 deg for the rest of this dataset, and
    // pairwise mean ~103.5 deg, approaching the ~120 deg random-orientation
    // benchmark.
    {{287, 852, 1265, 3034, 1920, 3345, 1710}},
    {{287, 852, 1266, 3088, 1765, 2502, 1493}},
    {{285, 852, 1265, 3087, 2263, 2477, 998}},
    {{286, 852, 1265, 3051, 2264, 2522, 604}},
    {{286, 852, 1265, 3055, 2262, 2540, 337}},
    {{337, 851, 1265, 3271, 2022, 1921, 3753}},
    {{337, 852, 1265, 3273, 2019, 1781, 3753}},
    {{340, 852, 1265, 3275, 1607, 1709, 3557}},
    {{339, 852, 1265, 3274, 1511, 1583, 3557}},
    {{339, 852, 1265, 3274, 1296, 1513, 3554}},
    {{340, 852, 1265, 3274, 1016, 1903, 3319}},
    {{354, 852, 1265, 3273, 2342, 2972, 1558}},
    {{351, 852, 1265, 3274, 2341, 3344, 1672}},

    // Poses 304-307 (0-based): backlash test for wrist_pitch (motor 5),
    // from recorded_hand_poses.csv rows 217-220. Other 6 joints locked
    // (torqued) during recording via record_hand_poses.cpp so they
    // couldn't drift across the four. ALL FOUR must be visited in this
    // exact order in the automated re-run too -- not just the two target
    // poses -- otherwise the arm (which ends this sequence sitting right
    // next to both targets already) would never actually approach either
    // target from the intended direction, defeating the whole test:
    //   304 (row 217): wrist_pitch BELOW the target (staging pose, not compared)
    //   305 (row 218): wrist_pitch AT the target, arriving from below (tick 2058) <- COMPARE
    //   306 (row 219): wrist_pitch ABOVE the target (staging pose, not compared)
    //   307 (row 220): wrist_pitch AT the target again, arriving from above
    //                  (tick 2059, an automatic servo-driven move back to
    //                  305's exact tick -- within the 15-tick settling
    //                  tolerance, i.e. effectively the same commanded
    //                  target) <- COMPARE
    // Compare the NDI-measured position of poses 305 vs 307: a gap
    // meaningfully larger than the ~0.5mm repeatability floor means real
    // backlash in wrist_pitch; a gap within that noise means it isn't a
    // meaningful contributor. See CLAUDE.md's kinematic-calibration
    // section for full context.
    {{1024, 2072, 2096, 2112, 2049, 840, 3020}},
    {{1026, 2077, 2096, 2112, 2049, 2058, 3020}},
    {{1026, 2079, 2096, 2116, 2051, 3345, 3022}},
    {{1026, 2078, 2096, 2116, 2050, 2059, 3022}},

    // Poses 308-311 (0-based): backlash test for elbow_pitch (motor 3),
    // from recorded_hand_poses.csv rows 221-224. Same structure/rationale
    // as poses 304-307 above -- all four must be visited in order:
    //   308 (row 221): elbow_pitch BELOW the target (staging, not compared)
    //   309 (row 222): elbow_pitch AT the target, arriving from below (tick 2079) <- COMPARE
    //   310 (row 223): elbow_pitch ABOVE the target (staging, not compared)
    //   311 (row 224): elbow_pitch AT the target again, arriving from above
    //                  (tick 2090, auto-driven back toward 309's tick --
    //                  landed 11 ticks off, within the 15-tick settling
    //                  tolerance) <- COMPARE
    // Compare NDI-measured position of poses 309 vs 311, same interpretation
    // as the wrist_pitch test (gap beyond ~0.5mm = real backlash).
    {{1102, 2055, 2059, 829, 2095, 2055, 3108}},
    {{1103, 2061, 2057, 2079, 2095, 2060, 3108}},
    {{1104, 2068, 2057, 3275, 2097, 2070, 3110}},
    {{1105, 2063, 2058, 2090, 2095, 2061, 3110}},

    // Poses 312-315 (0-based): REDO of the elbow_pitch backlash test above,
    // from recorded_hand_poses.csv rows 225-228, after tightening
    // record_hand_poses.cpp's AUTO_DRIVE_TOLERANCE_TICKS 15->4 -- the
    // first attempt's two "target" poses landed 11 ticks apart (2079 vs
    // 2090), too loose given elbow_pitch's longer lever arm to the
    // marker. This time: 313 (row 226) targets tick 2077, 315 (row 228,
    // auto-driven) landed at 2081 -- only 4 ticks apart.
    //   312 (row 225): elbow_pitch BELOW the target (staging, not compared)
    //   313 (row 226): elbow_pitch AT the target, arriving from below (tick 2077) <- COMPARE
    //   314 (row 227): elbow_pitch ABOVE the target (staging, not compared)
    //   315 (row 228): elbow_pitch AT the target again, arriving from above
    //                  (tick 2081) <- COMPARE
    {{1102, 2055, 2056, 829, 2089, 2048, 3101}},
    {{1102, 2064, 2055, 2077, 2090, 2055, 3102}},
    {{1103, 2069, 2054, 3275, 2092, 2066, 3104}},
    {{1103, 2064, 2054, 2081, 2091, 2056, 3104}},

    // Poses 316-318 (0-based): real-world IK round-trip validation
    // (2026-07-29, calibration/diag_ik_validation_setup.py). Each pose's
    // ticks were found by numerically inverting the FULLY corrected model
    // (offset+scale+tilt+origin+shoulder_pitch Fourier term+tool+base
    // frame, RMS 8.187mm on the 298-pose training set) to reach a chosen
    // target position, starting the solve from a DIFFERENT initial guess
    // than whatever generated the target (residual ~0.0000mm -- confirms
    // the solve genuinely converged, not just returned its start point).
    // Compare each pose's NDI-measured moving_relative_fixed position
    // against its intended target (see the script's printed output) --
    // this tests whether the offline-validated ~8mm accuracy figure
    // actually holds when going the OTHER direction: position -> commanded
    // joint angles -> real measured result, the practical direction any
    // real IK/MoveIt usage would need. MOVING_SPEED reduced to 15 for
    // this run (see that constant's comment) since these exact
    // combinations were never hand-verified the way every other pose in
    // this file was -- individually within jointCalibrations' safe range
    // with margin, but not cross-checked for self-collision.
    //   Target 1: [218.48, 265.65, -375.24] mm
    //   Target 2: [219.13, -556.65, -98.84] mm
    //   Target 3: [228.88, -427.39, -365.99] mm
    {{1722, 1184, 1815, 2011, 1919, 2160, 1401}},
    {{2289, 2058, 2512, 1858, 2246, 2023, 2215}},
    {{2209, 2225, 2191, 1959, 1985, 1670, 2227}},

    // Multi-joint backlash test (2026-07-30, record_hand_poses.cpp,
    // generalized version -- see CLAUDE.md's kinematic-calibration
    // section). Each joint gets 4 poses in order: below target, at target
    // (arriving from below), above target, at target again (auto-driven
    // back to the exact tick recorded for pose 2, arriving from above).
    // All 4 per joint must be visited in that order -- comparing only the
    // 2 "at target" poses without also passing through "below"/"above"
    // first wouldn't actually approach from the intended direction.
    // motor 0 backlash test (index 319-322)
    {{1418, 2056, 2048, 2062, 2124, 2105, 2976}},
    {{1032, 2055, 2048, 2062, 2124, 2105, 2977}},
    {{819, 2056, 2048, 2062, 2124, 2105, 2977}},
    {{1028, 2056, 2048, 2062, 2124, 2105, 2977}},
    // motor 1 backlash test (index 323-326)
    {{1029, 1698, 2049, 2062, 2123, 2101, 2977}},
    {{1026, 2020, 2048, 2062, 2124, 2105, 2976}},
    {{1024, 2320, 2050, 2067, 2125, 2112, 2977}},
    {{1026, 2022, 2048, 2062, 2124, 2103, 2977}},
    // motor 2 backlash test (index 327-330)
    {{1029, 2026, 2343, 2065, 2125, 2105, 2977}},
    {{1027, 2020, 2067, 2063, 2125, 2105, 2977}},
    {{1029, 2020, 1816, 2063, 2123, 2105, 2977}},
    {{1028, 2020, 2064, 2063, 2125, 2104, 2977}},
    // motor 3 backlash test (index 331-334)
    {{945, 2075, 2041, 1435, 2060, 2062, 2975}},
    {{946, 2073, 2041, 2022, 2059, 2063, 2977}},
    {{943, 2068, 2044, 2501, 2059, 2069, 2978}},
    {{946, 2073, 2041, 2025, 2058, 2062, 2978}},
    // motor 4 backlash test (index 335-338)
    {{946, 2070, 2044, 2023, 1710, 2062, 2979}},
    {{946, 2071, 2042, 2023, 2048, 2062, 2979}},
    {{946, 2074, 2043, 2024, 2282, 2067, 2978}},
    {{945, 2067, 2043, 2021, 2053, 2053, 2979}},
    // motor 5 backlash test (index 339-342)
    {{946, 2072, 2041, 2025, 2054, 1234, 2979}},
    {{944, 2072, 2041, 2024, 2053, 2109, 2979}},
    {{945, 2068, 2043, 2022, 2054, 2905, 2979}},
    {{946, 2073, 2041, 2025, 2053, 2108, 2979}},
    // motor 6 backlash test (index 343-346)
    {{946, 2070, 2041, 2024, 2053, 2104, 2735}},
    {{946, 2069, 2041, 2023, 2053, 2102, 3081}},
    {{946, 2069, 2043, 2023, 2053, 2103, 3403}},
    {{946, 2070, 2040, 2023, 2053, 2104, 3085}},

    // Displacement-from-home test (2026-07-30): home position + increasing
    // Cartesian offsets along +Y (0/5/10/15/20/25/30mm), numerically
    // inverted via the current-best fitted FK model (calibrate_kinematics.py
    // + gen_displacement_from_home_poses.py -- see CLAUDE.md's kinematic-
    // calibration section). Checks whether position error grows roughly
    // proportionally with commanded displacement, the signature of a
    // residual joint gear-ratio scale error -- distinct from the Jacobian/
    // manipulability and travel-distance hypotheses already ruled out on
    // the existing calibration dataset. Individually within jointCalibrations'
    // safe range, but like the earlier IK round-trip poses, NEVER hand-
    // verified for self-collision (numerically inverted, not physically
    // posed) -- MOVING_SPEED reduced for this run, same precaution as before.
    // index 347-353
    {{1545, 2044, 1625, 2101, 2049, 2120, 2090}},  // home + 0mm
    {{1541, 2052, 1620, 2102, 2006, 2092, 2318}},  // home + 5mm
    {{1571, 2077, 1618, 2092, 1999, 2082, 2308}},  // home + 10mm
    {{1571, 2080, 1618, 2086, 1989, 2068, 2303}},  // home + 15mm
    {{1571, 2084, 1617, 2081, 1981, 2056, 2300}},  // home + 20mm
    {{1568, 2084, 1615, 2075, 1974, 2045, 2297}},  // home + 25mm
    {{1569, 2085, 1614, 2071, 1967, 2035, 2292}},  // home + 30mm

    // Realm-restricted calibration poses (2026-07-31): 129 hand-posed
    // configurations within the user's actual reduced operating envelope
    // ("bent arm grabbing something," never fully extended), recorded via
    // record_hand_poses (1).csv (200 poses collected, 71 excluded here for
    // falling outside jointCalibrations' safe range on joint 0 and/or joint
    // 4 -- see CLAUDE.md's kinematic-calibration section). Purpose: test
    // whether restricting both calibration data collection AND the fitted
    // model to this narrower realm improves accuracy there, following the
    // in-sample finding that excluding just the 37-pose extreme-config
    // cluster from the existing dataset already dropped bulk RMS from
    // 5.38mm to 4.74mm with no new data at all.
    // index 354-482
    {{488, 1862, 1323, 3274, 952, 2321, 3383}},
    {{561, 1861, 1322, 3274, 953, 2304, 3447}},
    {{629, 1862, 1323, 3264, 952, 2279, 3491}},
    {{698, 1863, 1323, 3179, 953, 2275, 3652}},
    {{759, 1865, 1322, 3167, 952, 2257, 3738}},
    {{769, 1866, 1244, 3124, 968, 2164, 3754}},
    {{766, 1861, 1264, 3129, 1002, 2253, 3732}},
    {{691, 1874, 1266, 3191, 997, 2266, 3575}},
    {{606, 1878, 1265, 3191, 979, 2304, 3507}},
    {{518, 1879, 1265, 3191, 993, 2304, 3435}},
    {{417, 1879, 1265, 3191, 974, 2313, 3370}},
    {{395, 1948, 1275, 3190, 1021, 2257, 3299}},
    {{474, 1949, 1274, 3190, 1016, 2250, 3362}},
    {{551, 1950, 1270, 3190, 1014, 2201, 3449}},
    {{625, 1946, 1266, 3190, 1012, 2214, 3502}},
    {{690, 1947, 1266, 3188, 1016, 2192, 3554}},
    {{745, 1945, 1222, 3188, 1014, 2152, 3655}},
    {{614, 1957, 1281, 3122, 1023, 2246, 3615}},
    {{547, 1958, 1278, 3122, 1023, 2268, 3565}},
    {{480, 1958, 1278, 3122, 1020, 2284, 3502}},
    {{414, 1961, 1278, 3124, 1021, 2280, 3450}},
    {{414, 1936, 1221, 3145, 1022, 2210, 3392}},
    {{454, 1935, 1208, 3145, 1022, 2186, 3440}},
    {{513, 1938, 1208, 3145, 1023, 2206, 3495}},
    {{564, 1937, 1208, 3145, 1023, 2202, 3555}},
    {{616, 1934, 1209, 3145, 1023, 2212, 3589}},
    {{685, 1934, 1211, 3140, 1024, 2227, 3666}},
    {{738, 1936, 1210, 3141, 1023, 2215, 3702}},
    {{781, 1946, 1211, 3138, 1027, 2211, 3753}},
    {{763, 1963, 1231, 3138, 1058, 2217, 3754}},
    {{756, 1985, 1232, 3137, 1103, 2254, 3754}},
    {{693, 1985, 1231, 3139, 1120, 2269, 3693}},
    {{643, 1986, 1231, 3141, 1118, 2279, 3634}},
    {{593, 1988, 1217, 3141, 1113, 2269, 3583}},
    {{521, 1983, 1214, 3140, 1091, 2291, 3467}},
    {{455, 1984, 1213, 3140, 1097, 2304, 3407}},
    {{393, 1984, 1213, 3139, 1091, 2321, 3359}},
    {{396, 1988, 1198, 3141, 1066, 2198, 3392}},
    {{472, 1927, 1198, 3140, 1022, 2206, 3462}},
    {{552, 1927, 1199, 3140, 1023, 2199, 3543}},
    {{618, 1925, 1198, 3137, 996, 2192, 3575}},
    {{675, 1926, 1198, 3130, 999, 2176, 3651}},
    {{746, 1928, 1198, 3129, 997, 2169, 3703}},
    {{813, 1932, 1182, 3128, 997, 2178, 3753}},
    {{704, 1954, 1183, 3093, 1054, 2174, 3746}},
    {{622, 1953, 1200, 3090, 1067, 2239, 3658}},
    {{519, 1955, 1200, 3091, 1056, 2259, 3593}},
    {{469, 1954, 1201, 3090, 1077, 2277, 3533}},
    {{429, 1954, 1200, 3090, 1057, 2293, 3476}},
    {{719, 1825, 1144, 3010, 993, 2087, 3753}},
    {{719, 1824, 1143, 3010, 1014, 2136, 3753}},
    {{719, 1800, 1101, 3010, 1017, 2129, 3753}},
    {{719, 1866, 1134, 3013, 1062, 2209, 3753}},
    {{671, 1921, 1147, 3019, 1074, 2207, 3753}},
    {{662, 1967, 1161, 3019, 1094, 2272, 3753}},
    {{600, 1985, 1196, 3021, 1094, 2271, 3753}},
    {{539, 1986, 1204, 3010, 1095, 2280, 3706}},
    {{464, 1995, 1204, 3010, 1095, 2253, 3608}},
    {{407, 2025, 1217, 2999, 1102, 2197, 3492}},
    {{474, 2024, 1217, 2993, 1054, 2188, 3595}},
    {{563, 1992, 1216, 2991, 1038, 2213, 3718}},
    {{619, 1957, 1197, 2991, 991, 2171, 3754}},
    {{617, 1955, 1192, 2993, 1002, 2207, 3694}},
    {{620, 1932, 1190, 2991, 956, 2145, 3754}},
    {{718, 1927, 1182, 2989, 957, 2135, 3755}},
    {{716, 1926, 1133, 2993, 991, 2172, 3755}},
    {{714, 1915, 1104, 2994, 993, 2182, 3754}},
    {{707, 1966, 1130, 3026, 1072, 2242, 3754}},
    {{588, 1988, 1129, 3056, 1072, 2232, 3753}},
    {{588, 1989, 1127, 3056, 1115, 2212, 3753}},
    {{590, 1988, 1127, 3055, 1051, 2139, 3711}},
    {{607, 1988, 1128, 3000, 1045, 2120, 3715}},
    {{613, 1989, 1129, 2995, 1008, 2084, 3716}},
    {{620, 1956, 1133, 2932, 966, 2110, 3755}},
    {{619, 1896, 1098, 2921, 957, 2135, 3753}},
    {{618, 1895, 1097, 2921, 1015, 2127, 3753}},
    {{618, 1895, 1073, 2923, 1017, 2143, 3713}},
    {{582, 1980, 1073, 2932, 1016, 2256, 3713}},
    {{581, 1979, 1040, 2941, 1052, 2239, 3709}},
    {{583, 1979, 1040, 2940, 1055, 2139, 3710}},
    {{584, 1981, 1041, 2938, 1047, 2048, 3714}},
    {{617, 1973, 1065, 2937, 1013, 2003, 3753}},
    {{683, 1971, 1037, 2917, 1012, 1975, 3755}},
    {{551, 1932, 1109, 2918, 1009, 2136, 3755}},
    {{550, 1920, 1110, 2993, 1009, 2157, 3754}},
    {{551, 1847, 1080, 3145, 1007, 2078, 3637}},
    {{555, 1845, 1084, 3137, 953, 2075, 3572}},
    {{662, 1846, 1166, 3126, 953, 2163, 3678}},
    {{802, 1848, 1191, 3089, 952, 2164, 3755}},
    {{800, 1848, 1190, 3087, 995, 2098, 3755}},
    {{799, 1949, 1220, 3087, 1013, 2190, 3754}},
    {{716, 1972, 1220, 3088, 1083, 2275, 3754}},
    {{609, 1982, 1219, 3090, 1105, 2280, 3667}},
    {{519, 1985, 1219, 3090, 1093, 2303, 3584}},
    {{441, 1984, 1205, 3090, 1070, 2288, 3508}},
    {{440, 1893, 1139, 3091, 989, 2206, 3438}},
    {{737, 1944, 1360, 3008, 955, 2315, 3753}},
    {{703, 1946, 1325, 3010, 978, 2311, 3753}},
    {{652, 1950, 1318, 3022, 995, 2304, 3753}},
    {{572, 1950, 1305, 3032, 997, 2301, 3753}},
    {{534, 1993, 1269, 2990, 1000, 2294, 3753}},
    {{463, 1994, 1271, 2981, 999, 2291, 3753}},
    {{379, 2020, 1268, 2907, 998, 2278, 3753}},
    {{709, 1897, 1090, 2990, 999, 2043, 3753}},
    {{695, 1789, 1083, 3185, 995, 2060, 3649}},
    {{696, 1909, 1140, 3185, 1085, 2227, 3660}},
    {{693, 1913, 1058, 3213, 1137, 2112, 3661}},
    {{695, 1906, 1060, 3215, 1012, 2042, 3662}},
    {{695, 1893, 1063, 3212, 1006, 1979, 3720}},
    {{696, 1869, 1065, 3205, 954, 1950, 3719}},
    {{789, 1863, 1028, 3116, 998, 1962, 3753}},
    {{789, 1865, 1008, 3115, 1066, 1992, 3753}},
    {{785, 1865, 952, 3116, 1084, 1977, 3753}},
    {{788, 1777, 970, 3116, 955, 1870, 3754}},
    {{810, 1848, 929, 2992, 1005, 1989, 3751}},
    {{761, 1853, 927, 2991, 1049, 2011, 3753}},
    {{733, 1852, 925, 2991, 1051, 2028, 3751}},
    {{653, 1853, 923, 3062, 1046, 1978, 3628}},
    {{599, 1855, 922, 3168, 1045, 1982, 3462}},
    {{527, 1876, 922, 3187, 1093, 1911, 3435}},
    {{485, 1888, 922, 3189, 1086, 1972, 3357}},
    {{435, 1904, 922, 3189, 1127, 1896, 3357}},
    {{424, 1911, 923, 3190, 1039, 1965, 3309}},
    {{426, 1907, 925, 3190, 961, 1885, 3330}},
    {{428, 1861, 933, 3187, 955, 1866, 3340}},
    {{428, 1861, 934, 3170, 955, 1830, 3374}},
    {{533, 1860, 934, 3107, 954, 1884, 3469}},
    {{550, 1860, 935, 3087, 953, 1842, 3582}},
    {{603, 1859, 943, 3087, 954, 1866, 3614}},

    // Widened-range calibration poses (2026-07-31): hand-posed
    // configurations previously excluded from the realm-restricted set
    // above because they fell outside jointCalibrations' OLD safe range
    // on joint 0, 3, 4, and/or 6 -- now included since those bounds were
    // widened (robot_calibration.cpp) to cover the full extent of both
    // recorded_hand_poses (1)/(2).csv batches. First 70 from batch 1
    // (excluded there for joint 0/4), remaining 104 from batch 2
    // (excluded for joint 0/3/4/6).
    // index 483-656
    {{276, 2036, 1392, 3275, 1236, 2395, 3051}},
    {{276, 2009, 1392, 3276, 1162, 2402, 3090}},
    {{276, 1971, 1324, 3276, 1082, 2342, 3119}},
    {{276, 1964, 1323, 3276, 1047, 2322, 3145}},
    {{276, 1927, 1307, 3275, 1022, 2292, 3149}},
    {{276, 1868, 1301, 3276, 955, 2295, 3184}},
    {{391, 1864, 1255, 3276, 950, 2236, 3275}},
    {{395, 1862, 1324, 3274, 950, 2304, 3328}},
    {{770, 1866, 1268, 3163, 951, 2176, 3754}},
    {{340, 1879, 1265, 3191, 987, 2309, 3291}},
    {{284, 1878, 1259, 3191, 1001, 2329, 3224}},
    {{281, 1876, 1219, 3192, 1022, 2266, 3180}},
    {{282, 1944, 1276, 3191, 1071, 2285, 3179}},
    {{327, 1946, 1275, 3191, 1033, 2263, 3235}},
    {{363, 1960, 1278, 3124, 1021, 2288, 3400}},
    {{314, 1961, 1276, 3125, 1022, 2293, 3347}},
    {{287, 1958, 1270, 3128, 1027, 2290, 3279}},
    {{282, 1933, 1247, 3156, 1049, 2304, 3255}},
    {{325, 1935, 1233, 3156, 1044, 2244, 3299}},
    {{366, 1936, 1232, 3149, 1024, 2219, 3338}},
    {{301, 1984, 1211, 3142, 1085, 2301, 3307}},
    {{283, 1985, 1209, 3142, 1099, 2257, 3255}},
    {{324, 1988, 1206, 3142, 1068, 2226, 3331}},
    {{343, 1957, 1200, 3093, 1073, 2264, 3382}},
    {{294, 1954, 1171, 3093, 1062, 2261, 3309}},
    {{290, 1934, 1118, 3093, 1051, 2193, 3309}},
    {{343, 1934, 1120, 3090, 1056, 2172, 3372}},
    {{344, 1897, 1140, 3090, 1009, 2180, 3416}},
    {{344, 1831, 1146, 3091, 952, 2127, 3418}},
    {{426, 1830, 1145, 3092, 948, 2110, 3543}},
    {{492, 1830, 1145, 3092, 948, 2129, 3574}},
    {{575, 1829, 1147, 3088, 949, 2118, 3682}},
    {{622, 1827, 1146, 3089, 950, 2061, 3683}},
    {{623, 1827, 1147, 3088, 949, 2037, 3651}},
    {{676, 1826, 1146, 3013, 951, 2102, 3753}},
    {{721, 1827, 1146, 3011, 950, 2113, 3753}},
    {{365, 1996, 1204, 3010, 1095, 2231, 3547}},
    {{305, 1997, 1204, 3010, 1097, 2216, 3500}},
    {{287, 2040, 1189, 3012, 1166, 2214, 3441}},
    {{338, 2040, 1158, 3011, 1142, 2141, 3492}},
    {{556, 1847, 1115, 3126, 951, 2081, 3572}},
    {{556, 1847, 1160, 3126, 951, 2147, 3572}},
    {{717, 1849, 1192, 3087, 951, 2168, 3755}},
    {{451, 1814, 1142, 3089, 951, 2168, 3486}},
    {{520, 1811, 1141, 3089, 949, 2142, 3573}},
    {{593, 1811, 1140, 3086, 949, 2126, 3650}},
    {{673, 1811, 1140, 3086, 948, 2136, 3728}},
    {{752, 1809, 1142, 3086, 950, 2114, 3754}},
    {{295, 2021, 1235, 2866, 998, 2266, 3752}},
    {{291, 2067, 1247, 2792, 1008, 2243, 3753}},
    {{294, 2021, 1247, 2792, 958, 2207, 3753}},
    {{281, 2049, 1197, 2878, 1042, 2199, 3753}},
    {{282, 1982, 1178, 2992, 1040, 2343, 3529}},
    {{282, 1982, 1166, 2992, 1055, 2300, 3529}},
    {{282, 1981, 1164, 2992, 1062, 2224, 3541}},
    {{282, 1946, 1167, 2992, 1056, 2187, 3555}},
    {{285, 1939, 1167, 2993, 973, 2156, 3550}},
    {{286, 1903, 1166, 2992, 955, 2183, 3554}},
    {{285, 1899, 1166, 2992, 952, 2221, 3555}},
    {{339, 1901, 1149, 2990, 944, 2228, 3555}},
    {{407, 1903, 1146, 2990, 945, 2147, 3648}},
    {{458, 1902, 1146, 2989, 944, 2137, 3658}},
    {{521, 1903, 1145, 2989, 944, 2141, 3717}},
    {{551, 1901, 1145, 2989, 947, 2048, 3753}},
    {{614, 1904, 1144, 2990, 945, 2057, 3755}},
    {{707, 1903, 1144, 2990, 945, 2084, 3755}},
    {{702, 1865, 1066, 3177, 951, 1918, 3753}},
    {{786, 1864, 1064, 3115, 951, 1968, 3754}},
    {{820, 1775, 970, 3111, 948, 1834, 3754}},
    {{822, 1652, 946, 2959, 950, 1913, 3753}},
    {{619, 2022, 1145, 3275, 1306, 2408, 3465}},
    {{770, 1822, 1062, 3023, 948, 1923, 3755}},
    {{769, 1820, 1001, 3030, 950, 1919, 3754}},
    {{683, 1796, 1009, 2974, 951, 1877, 3754}},
    {{530, 1711, 932, 2865, 1040, 1719, 3756}},
    {{576, 1691, 933, 2815, 1010, 1685, 3756}},
    {{616, 1664, 933, 2808, 949, 1667, 3756}},
    {{615, 1661, 927, 2829, 948, 1815, 3755}},
    {{528, 1661, 919, 2878, 951, 1855, 3754}},
    {{357, 1662, 916, 2894, 951, 1892, 3632}},
    {{289, 1694, 914, 2904, 951, 1934, 3606}},
    {{285, 1775, 914, 2960, 1009, 1984, 3492}},
    {{286, 1798, 914, 2962, 1078, 1941, 3401}},
    {{286, 1806, 916, 2957, 1121, 1818, 3401}},
    {{287, 1806, 918, 2955, 1122, 1709, 3402}},
    {{290, 1802, 930, 2928, 1052, 1631, 3403}},
    {{295, 1785, 931, 2783, 1046, 1635, 3554}},
    {{363, 1705, 932, 2755, 1044, 1624, 3725}},
    {{420, 1669, 932, 2725, 951, 1548, 3755}},
    {{564, 1988, 1141, 2795, 947, 2040, 3755}},
    {{586, 1989, 1141, 2795, 945, 2036, 3755}},
    {{682, 1988, 1118, 2824, 944, 2066, 3755}},
    {{689, 1984, 1113, 2977, 947, 2096, 3754}},
    {{700, 1916, 1084, 2981, 948, 1848, 3755}},
    {{771, 1918, 1079, 2981, 946, 1952, 3755}},
    {{841, 1915, 1078, 2987, 950, 2034, 3755}},
    {{580, 2486, 1006, 2780, 1572, 2074, 3756}},
    {{580, 2559, 1005, 2711, 1652, 2115, 3756}},
    {{583, 2601, 1006, 2621, 1645, 2207, 3756}},
    {{583, 2759, 1005, 2473, 1730, 2513, 3758}},
    {{581, 2654, 974, 2570, 1775, 2464, 3757}},
    {{340, 2659, 970, 2604, 1841, 2496, 3678}},
    {{297, 2694, 971, 2604, 1862, 2519, 3594}},
    {{295, 2759, 971, 2600, 1963, 2520, 3594}},
    {{300, 2801, 966, 2429, 1958, 2727, 3617}},
    {{309, 2938, 944, 2285, 1956, 2905, 3669}},
    {{399, 2748, 929, 2357, 1803, 2739, 3756}},
    {{399, 2826, 929, 2301, 1802, 2836, 3756}},
    {{464, 2842, 931, 2300, 1890, 2852, 3757}},
    {{466, 2846, 929, 2275, 1893, 2854, 3757}},
    {{465, 2857, 930, 2252, 1894, 2866, 3756}},
    {{359, 2861, 933, 2219, 1916, 2903, 3754}},
    {{300, 2862, 933, 2209, 1916, 2981, 3752}},
    {{302, 3060, 923, 2026, 1897, 3303, 3754}},
    {{419, 2391, 1206, 2610, 1225, 2500, 3757}},
    {{419, 2383, 1207, 2612, 1140, 2495, 3757}},
    {{419, 2377, 1207, 2614, 1052, 2443, 3757}},
    {{419, 2312, 1206, 2623, 1024, 2331, 3760}},
    {{478, 2258, 1206, 2674, 1024, 2303, 3760}},
    {{559, 2193, 1204, 2729, 1028, 2240, 3760}},
    {{569, 2186, 1206, 2739, 1059, 2215, 3760}},
    {{639, 2147, 1204, 2775, 1069, 2220, 3760}},
    {{644, 2147, 1204, 2797, 1146, 2211, 3760}},
    {{649, 2150, 1165, 2796, 1225, 2173, 3759}},
    {{624, 2151, 1156, 2796, 1281, 2221, 3757}},
    {{555, 2152, 1133, 2796, 1320, 2179, 3756}},
    {{491, 2592, 1005, 2737, 1749, 2195, 3757}},
    {{492, 2693, 1007, 2630, 1792, 2342, 3757}},
    {{359, 2979, 1103, 2355, 1798, 2882, 3755}},
    {{386, 2758, 931, 2409, 1784, 2717, 3757}},
    {{339, 2954, 935, 2366, 1988, 2773, 3677}},
    {{329, 2956, 934, 2436, 2119, 2704, 3540}},
    {{310, 2955, 936, 2493, 2127, 2695, 3468}},
    {{290, 2954, 936, 2591, 2131, 2569, 3465}},
    {{291, 2956, 934, 2701, 2184, 2423, 3465}},
    {{289, 2963, 934, 2798, 2225, 2326, 3470}},
    {{291, 2981, 935, 2690, 2184, 2473, 3542}},
    {{288, 3082, 937, 2501, 2182, 2695, 3675}},
    {{299, 3104, 933, 2492, 2181, 2729, 3754}},
    {{342, 3108, 932, 2481, 2177, 2689, 3665}},
    {{374, 3105, 933, 2413, 2339, 2676, 3496}},
    {{303, 3104, 936, 2416, 2346, 2688, 3401}},
    {{287, 3105, 948, 2444, 2350, 2652, 3269}},
    {{284, 3103, 1005, 2464, 2351, 2686, 3136}},
    {{279, 3106, 1042, 2469, 2421, 2682, 3008}},
    {{279, 3115, 1048, 2478, 2457, 2752, 3010}},
    {{281, 3133, 1043, 2405, 2455, 2933, 3052}},
    {{281, 3134, 1031, 2314, 2455, 3060, 3098}},
    {{281, 3132, 1031, 2313, 2455, 2903, 3144}},
    {{281, 3140, 980, 2310, 2456, 2982, 3193}},
    {{286, 3167, 933, 2286, 2454, 2939, 3192}},
    {{293, 3169, 925, 2235, 2449, 3116, 3205}},
    {{286, 3163, 930, 2237, 2459, 2913, 3234}},
    {{321, 3171, 925, 2222, 2454, 3037, 3253}},
    {{363, 3170, 931, 2236, 2461, 2782, 3418}},
    {{310, 3170, 931, 2283, 2481, 2775, 3358}},
    {{292, 3170, 932, 2326, 2555, 2774, 3281}},
    {{288, 3174, 934, 2356, 2680, 2782, 3090}},
    {{285, 3175, 972, 2409, 2682, 2785, 2974}},
    {{322, 3228, 937, 2316, 2682, 2856, 3107}},
    {{362, 3221, 946, 2404, 2597, 2708, 3217}},
    {{317, 3221, 952, 2443, 2600, 2671, 3182}},
    {{315, 3216, 950, 2371, 2603, 2831, 3180}},
    {{316, 3216, 952, 2352, 2606, 2809, 3181}},
    {{299, 3218, 971, 2420, 2624, 2748, 3151}},
    {{291, 3225, 1017, 2488, 2683, 2637, 3076}},
    {{290, 3225, 1020, 2468, 2684, 2540, 3107}},
    {{292, 3225, 1015, 2408, 2524, 2593, 3340}},
    {{334, 3226, 1002, 2371, 2506, 2651, 3444}},
    {{359, 3200, 952, 2510, 2556, 2628, 3329}},
    {{297, 3202, 972, 2515, 2615, 2601, 3189}},
    {{291, 3218, 1017, 2450, 2657, 2674, 3192}},
    {{289, 3226, 1074, 2455, 2601, 2606, 3192}},
    {{333, 3226, 1062, 2415, 2518, 2596, 3319}},

    // recorded_hand_poses (3).csv batch (2026-07-31): 200 new hand-posed
    // configurations, collected after the joint 0/3/4/6 widening. All fall
    // within the current jointCalibrations bounds except joint 6's max,
    // which needed one more tick (3760 -> 3761) to fit this batch's max
    // observed value.
    // index 657-856
    {{456, 2407, 1076, 2740, 1425, 2275, 3701}},
    {{480, 2412, 1076, 2729, 1411, 2284, 3754}},
    {{513, 2464, 1077, 2730, 1407, 2302, 3757}},
    {{529, 2473, 1108, 2692, 1406, 2318, 3758}},
    {{554, 2512, 1108, 2691, 1405, 2330, 3760}},
    {{490, 2480, 1092, 2756, 1466, 2306, 3756}},
    {{570, 2505, 1093, 2753, 1466, 2319, 3758}},
    {{617, 2511, 1094, 2732, 1468, 2310, 3758}},
    {{642, 2569, 1096, 2704, 1470, 2336, 3761}},
    {{698, 2573, 1137, 2670, 1475, 2368, 3761}},
    {{668, 2573, 1139, 2710, 1559, 2375, 3759}},
    {{613, 2572, 1139, 2724, 1560, 2385, 3757}},
    {{558, 2573, 1139, 2731, 1567, 2389, 3757}},
    {{505, 2573, 1140, 2736, 1568, 2404, 3756}},
    {{484, 2632, 1138, 2737, 1676, 2392, 3757}},
    {{484, 2751, 1139, 2679, 1699, 2510, 3753}},
    {{484, 2761, 1136, 2540, 1703, 2596, 3753}},
    {{524, 2797, 1106, 2510, 1702, 2601, 3756}},
    {{559, 2844, 1098, 2476, 1702, 2632, 3757}},
    {{606, 2865, 1094, 2409, 1704, 2708, 3758}},
    {{620, 2917, 1072, 2406, 1707, 2720, 3757}},
    {{589, 2923, 1071, 2411, 1879, 2743, 3754}},
    {{581, 2941, 1072, 2411, 1891, 2713, 3754}},
    {{583, 2950, 1072, 2400, 1890, 2673, 3754}},
    {{572, 2951, 1074, 2405, 1892, 2748, 3738}},
    {{554, 2951, 1075, 2413, 1895, 2756, 3730}},
    {{533, 2955, 1074, 2422, 1895, 2753, 3730}},
    {{520, 2962, 1074, 2447, 1895, 2757, 3730}},
    {{512, 2964, 1075, 2469, 1895, 2732, 3730}},
    {{509, 2971, 1075, 2488, 1896, 2703, 3730}},
    {{509, 2981, 1075, 2502, 1896, 2683, 3730}},
    {{509, 2990, 1076, 2506, 1896, 2641, 3730}},
    {{509, 3066, 1075, 2504, 2075, 2670, 3656}},
    {{453, 3073, 1075, 2505, 2133, 2680, 3559}},
    {{383, 3091, 1075, 2507, 2134, 2686, 3485}},
    {{437, 3095, 1075, 2478, 2135, 2625, 3486}},
    {{501, 3095, 1075, 2449, 2137, 2634, 3491}},
    {{564, 3096, 1074, 2449, 2135, 2662, 3536}},
    {{670, 3095, 1075, 2401, 2136, 2711, 3575}},
    {{554, 3093, 1075, 2465, 2136, 2640, 3532}},
    {{487, 3094, 1076, 2551, 2195, 2558, 3464}},
    {{474, 3097, 1077, 2628, 2265, 2485, 3401}},
    {{496, 3143, 1076, 2596, 2256, 2531, 3447}},
    {{515, 3154, 1067, 2524, 2261, 2615, 3450}},
    {{544, 3168, 1034, 2505, 2262, 2616, 3480}},
    {{589, 3168, 1012, 2487, 2262, 2622, 3527}},
    {{645, 3168, 1012, 2439, 2266, 2654, 3533}},
    {{648, 3165, 1032, 2415, 2355, 2620, 3451}},
    {{595, 3166, 1095, 2412, 2352, 2640, 3434}},
    {{534, 3169, 1131, 2417, 2354, 2658, 3344}},
    {{458, 3170, 1133, 2458, 2355, 2608, 3265}},
    {{399, 3170, 1133, 2498, 2355, 2583, 3205}},
    {{396, 3188, 1077, 2522, 2353, 2641, 3210}},
    {{400, 3216, 1009, 2525, 2353, 2675, 3275}},
    {{401, 3220, 950, 2457, 2351, 2708, 3325}},
    {{441, 3219, 941, 2374, 2352, 2753, 3368}},
    {{534, 3217, 943, 2358, 2352, 2735, 3369}},
    {{609, 3218, 944, 2368, 2353, 2733, 3429}},
    {{550, 3214, 955, 2436, 2354, 2704, 3375}},
    {{458, 3215, 960, 2504, 2358, 2625, 3338}},
    {{394, 3213, 971, 2536, 2418, 2580, 3271}},
    {{401, 3221, 1000, 2432, 2423, 2625, 3272}},
    {{448, 3222, 999, 2358, 2423, 2711, 3275}},
    {{514, 3222, 999, 2308, 2427, 2731, 3276}},
    {{568, 3222, 998, 2289, 2434, 2717, 3329}},
    {{627, 3223, 998, 2274, 2435, 2758, 3331}},
    {{637, 3222, 983, 2318, 2440, 2835, 3337}},
    {{581, 3222, 981, 2387, 2466, 2778, 3331}},
    {{493, 3221, 982, 2410, 2470, 2806, 3273}},
    {{407, 3221, 983, 2413, 2472, 2842, 3176}},
    {{324, 3220, 983, 2412, 2480, 2877, 3112}},
    {{304, 3222, 1007, 2387, 2505, 2626, 3105}},
    {{306, 3224, 1006, 2355, 2504, 2591, 3105}},
    {{323, 3223, 986, 2326, 2504, 2534, 3105}},
    {{324, 3222, 937, 2310, 2504, 2460, 3104}},
    {{326, 3223, 930, 2315, 2504, 2532, 3105}},
    {{339, 3224, 929, 2317, 2504, 2638, 3105}},
    {{364, 3224, 929, 2317, 2503, 2767, 3105}},
    {{412, 3225, 927, 2318, 2504, 2873, 3106}},
    {{420, 3225, 927, 2314, 2503, 2665, 3106}},
    {{428, 3225, 928, 2312, 2502, 2548, 3105}},
    {{471, 3225, 927, 2310, 2503, 2473, 3105}},
    {{509, 3225, 928, 2310, 2502, 2425, 3189}},
    {{548, 3225, 928, 2315, 2502, 2656, 3278}},
    {{552, 3225, 928, 2329, 2502, 2743, 3278}},
    {{551, 3225, 929, 2379, 2504, 2783, 3278}},
    {{535, 3223, 932, 2458, 2505, 2754, 3278}},
    {{494, 3223, 932, 2505, 2525, 2712, 3273}},
    {{460, 3220, 936, 2521, 2582, 2593, 3273}},
    {{428, 3220, 939, 2538, 2647, 2480, 3274}},
    {{442, 3222, 937, 2503, 2627, 2441, 3278}},
    {{477, 3222, 930, 2459, 2619, 2395, 3306}},
    {{552, 3223, 928, 2413, 2601, 2405, 3380}},
    {{606, 3223, 928, 2358, 2596, 2517, 3381}},
    {{653, 3223, 929, 2321, 2580, 2625, 3381}},
    {{659, 3225, 929, 2317, 2566, 2763, 3381}},
    {{644, 3225, 931, 2354, 2567, 2805, 3380}},
    {{588, 3224, 932, 2401, 2574, 2807, 3376}},
    {{522, 3224, 934, 2452, 2576, 2788, 3374}},
    {{474, 3224, 933, 2563, 2576, 2687, 3375}},
    {{474, 3224, 934, 2564, 2577, 2531, 3378}},
    {{480, 3225, 932, 2512, 2574, 2499, 3380}},
    {{529, 3226, 930, 2430, 2567, 2552, 3381}},
    {{569, 3225, 930, 2384, 2506, 2550, 3517}},
    {{582, 3224, 931, 2278, 2496, 2688, 3517}},
    {{608, 3224, 930, 2225, 2473, 2724, 3515}},
    {{619, 3224, 932, 2228, 2467, 2913, 3514}},
    {{563, 3224, 936, 2270, 2472, 2938, 3513}},
    {{495, 3224, 961, 2337, 2493, 2856, 3513}},
    {{444, 3224, 1036, 2368, 2502, 2833, 3513}},
    {{446, 3225, 1040, 2372, 2501, 2788, 3504}},
    {{445, 3224, 1040, 2375, 2501, 2762, 3405}},
    {{442, 3224, 1042, 2381, 2503, 2746, 3295}},
    {{435, 3223, 1046, 2382, 2504, 2737, 3240}},
    {{423, 3224, 1053, 2384, 2504, 2735, 3216}},
    {{411, 3223, 1060, 2386, 2505, 2737, 3197}},
    {{404, 3223, 1065, 2394, 2505, 2758, 3197}},
    {{401, 3225, 1063, 2456, 2504, 2804, 3197}},
    {{404, 3224, 1063, 2445, 2500, 2603, 3197}},
    {{415, 3224, 1062, 2435, 2494, 2374, 3197}},
    {{423, 3225, 1055, 2432, 2468, 2542, 3200}},
    {{425, 3226, 1047, 2439, 2459, 2694, 3200}},
    {{440, 3225, 1013, 2435, 2451, 2735, 3202}},
    {{481, 3225, 1007, 2411, 2448, 2786, 3203}},
    {{540, 3225, 1007, 2408, 2425, 2764, 3204}},
    {{479, 3224, 1008, 2412, 2542, 2733, 3198}},
    {{526, 3170, 1016, 2366, 2882, 2841, 2951}},
    {{539, 3180, 1013, 2344, 2880, 2880, 3000}},
    {{566, 3216, 1011, 2288, 2878, 2946, 3002}},
    {{619, 3222, 1003, 2213, 2878, 3013, 3002}},
    {{701, 3221, 962, 2147, 2876, 2947, 3116}},
    {{762, 3148, 924, 2080, 2917, 2965, 3191}},
    {{765, 3189, 927, 2126, 2918, 2854, 3194}},
    {{817, 3227, 927, 2131, 2906, 2774, 3281}},
    {{781, 3228, 968, 2210, 2906, 2752, 3282}},
    {{751, 3225, 991, 2347, 2915, 2649, 3250}},
    {{711, 3225, 998, 2428, 2941, 2701, 3254}},
    {{710, 3225, 1002, 2399, 2788, 2647, 3416}},
    {{629, 3224, 1003, 2372, 2776, 2699, 3412}},
    {{515, 3223, 1002, 2322, 2755, 2788, 3377}},
    {{507, 3222, 1001, 2340, 2721, 2852, 3210}},
    {{512, 3227, 1001, 2255, 2704, 2746, 3210}},
    {{572, 3227, 999, 2275, 2696, 2740, 3241}},
    {{635, 3227, 998, 2293, 2629, 2718, 3377}},
    {{697, 3227, 996, 2321, 2585, 2702, 3457}},
    {{661, 3225, 1001, 2432, 2611, 2682, 3385}},
    {{584, 3224, 1001, 2478, 2611, 2607, 3305}},
    {{559, 3225, 1010, 2467, 2585, 2453, 3310}},
    {{593, 3225, 1002, 2440, 2534, 2479, 3441}},
    {{650, 3225, 993, 2396, 2506, 2511, 3444}},
    {{720, 3223, 985, 2349, 2505, 2647, 3487}},
    {{688, 3221, 993, 2390, 2613, 2696, 3398}},
    {{597, 3223, 996, 2414, 2668, 2676, 3330}},
    {{516, 3223, 998, 2414, 2661, 2647, 3296}},
    {{406, 3223, 1001, 2415, 2616, 2629, 3217}},
    {{308, 3224, 1006, 2415, 2565, 2595, 3145}},
    {{336, 3224, 1003, 2406, 2434, 2616, 3147}},
    {{399, 3224, 999, 2390, 2424, 2581, 3210}},
    {{411, 3218, 951, 2271, 2426, 2752, 3209}},
    {{479, 3219, 947, 2221, 2400, 2736, 3209}},
    {{549, 3218, 947, 2154, 2371, 2655, 3209}},
    {{375, 3219, 961, 2136, 2371, 2568, 3209}},
    {{284, 3219, 993, 2146, 2372, 2909, 3210}},
    {{280, 3219, 1005, 2161, 2372, 3097, 3210}},
    {{280, 3219, 1107, 2161, 2371, 3057, 3210}},
    {{507, 3152, 947, 2135, 2523, 3017, 3207}},
    {{537, 3192, 945, 2137, 2524, 3009, 3208}},
    {{568, 3226, 944, 2102, 2521, 3043, 3207}},
    {{519, 3221, 948, 2110, 2522, 3115, 3205}},
    {{366, 3221, 967, 2182, 2523, 3066, 3130}},
    {{296, 3221, 990, 2239, 2524, 2868, 3023}},
    {{295, 3222, 991, 2244, 2477, 2726, 2967}},
    {{356, 3223, 963, 2243, 2468, 2728, 2970}},
    {{305, 3221, 1000, 2287, 2475, 2651, 2965}},
    {{285, 3221, 1009, 2340, 2536, 2748, 2934}},
    {{282, 3221, 1009, 2432, 2673, 2741, 2845}},
    {{292, 3223, 996, 2361, 2583, 2696, 2850}},
    {{353, 3223, 963, 2341, 2534, 2607, 2935}},
    {{431, 3223, 953, 2318, 2501, 2634, 3096}},
    {{499, 3223, 948, 2247, 2445, 2684, 3244}},
    {{522, 3223, 946, 2162, 2439, 2821, 3246}},
    {{537, 3225, 948, 2241, 2441, 2801, 3246}},
    {{540, 3225, 949, 2340, 2443, 2753, 3246}},
    {{480, 3223, 982, 2406, 2457, 2713, 3240}},
    {{362, 3224, 1002, 2418, 2458, 2691, 3147}},
    {{307, 3224, 1002, 2423, 2458, 2687, 3096}},
    {{312, 3221, 1002, 2335, 2450, 2617, 3094}},
    {{413, 3222, 1001, 2251, 2414, 2692, 3173}},
    {{464, 3222, 993, 2176, 2402, 2833, 3230}},
    {{520, 3222, 979, 2191, 2401, 2974, 3281}},
    {{455, 3221, 988, 2265, 2434, 2976, 3276}},
    {{409, 3220, 993, 2302, 2464, 2775, 3279}},
    {{359, 3216, 996, 2301, 2464, 2688, 3248}},
    {{308, 3216, 1010, 2304, 2466, 2716, 3084}},
    {{286, 3216, 1029, 2310, 2530, 2786, 2967}},
    {{284, 3216, 1041, 2376, 2580, 2790, 2892}},
    {{285, 3219, 1039, 2433, 2585, 2784, 2892}},
    {{288, 3220, 963, 2432, 2584, 2833, 2897}},
    {{318, 3224, 932, 2414, 2582, 2765, 2936}},
    {{372, 3223, 928, 2383, 2581, 2792, 2999}},

    // Reduced-DOF collection (2026-08-04): elbow_yaw (motor 4) held
    // fixed at its calibrated midpoint (2095) for every pose in this
    // batch, only the other 6 joints hand-posed -- see CLAUDE.md's
    // kinematic-calibration section. From recorded_hand_poses_fixed_
    // elbow_yaw.csv.
    // index 857-1238
    {{740, 2009, 2012, 3276, 2093, 2870, 2775}},
    {{807, 2004, 2013, 3276, 2087, 2882, 2888}},
    {{908, 1980, 2013, 3276, 2087, 2912, 2957}},
    {{1000, 1982, 2012, 3276, 2086, 2905, 3045}},
    {{1103, 1980, 2012, 3276, 2087, 2917, 3140}},
    {{1193, 1988, 2013, 3276, 2085, 2887, 3254}},
    {{1302, 2004, 2012, 3276, 2085, 2879, 3355}},
    {{1380, 2013, 2012, 3276, 2084, 2834, 3470}},
    {{1475, 2021, 2012, 3276, 2086, 2813, 3531}},
    {{1557, 2092, 2012, 3260, 2086, 2781, 3609}},
    {{1677, 2112, 2012, 3190, 2087, 2837, 3718}},
    {{1603, 2111, 2011, 3214, 2094, 2779, 3632}},
    {{1543, 2106, 2011, 3240, 2095, 2795, 3575}},
    {{1461, 2082, 2012, 3274, 2095, 2800, 3487}},
    {{1371, 2052, 2012, 3275, 2096, 2829, 3392}},
    {{1249, 2018, 2013, 3276, 2095, 2873, 3275}},
    {{1133, 2000, 2014, 3277, 2095, 2900, 3165}},
    {{966, 1998, 2014, 3277, 2095, 2909, 2980}},
    {{834, 1999, 2015, 3277, 2094, 2901, 2839}},
    {{673, 2034, 2015, 3276, 2094, 2847, 2720}},
    {{760, 2061, 2015, 3237, 2084, 2847, 2819}},
    {{821, 2058, 2015, 3259, 2084, 2846, 2818}},
    {{903, 2057, 2015, 3260, 2084, 2827, 2909}},
    {{984, 2051, 2014, 3261, 2086, 2851, 2978}},
    {{1066, 2050, 2015, 3262, 2084, 2844, 3088}},
    {{1137, 2050, 2015, 3257, 2083, 2839, 3191}},
    {{1239, 2051, 2015, 3251, 2085, 2849, 3287}},
    {{1325, 2055, 2015, 3247, 2085, 2837, 3381}},
    {{1389, 2057, 2014, 3245, 2084, 2829, 3447}},
    {{1441, 2059, 2014, 3244, 2084, 2834, 3504}},
    {{1488, 2116, 2012, 3213, 2088, 2816, 3553}},
    {{1402, 2112, 2012, 3214, 2095, 2802, 3450}},
    {{1296, 2112, 2012, 3216, 2095, 2819, 3351}},
    {{1199, 2113, 2014, 3219, 2095, 2813, 3253}},
    {{1125, 2113, 2014, 3221, 2094, 2806, 3195}},
    {{1065, 2112, 2013, 3226, 2095, 2804, 3134}},
    {{988, 2112, 2013, 3228, 2096, 2810, 3035}},
    {{898, 2115, 2014, 3228, 2094, 2796, 2965}},
    {{836, 2116, 2014, 3229, 2095, 2810, 2888}},
    {{783, 2115, 2014, 3229, 2096, 2799, 2832}},
    {{722, 2115, 2014, 3230, 2095, 2806, 2773}},
    {{679, 2112, 2015, 3230, 2093, 2827, 2750}},
    {{614, 2101, 2015, 3231, 2094, 2849, 2649}},
    {{578, 2035, 2015, 3276, 2091, 2871, 2587}},
    {{586, 2009, 2022, 3276, 2087, 2910, 2593}},
    {{633, 1981, 2055, 3276, 2085, 2927, 2648}},
    {{748, 1957, 2062, 3276, 2085, 2949, 2777}},
    {{839, 1947, 2061, 3276, 2085, 2951, 2844}},
    {{919, 1922, 2060, 3276, 2085, 2958, 2935}},
    {{1025, 1920, 2059, 3276, 2085, 2958, 3042}},
    {{1111, 1920, 2058, 3276, 2087, 2956, 3126}},
    {{1194, 1923, 2058, 3276, 2086, 2945, 3203}},
    {{1268, 1924, 2058, 3276, 2085, 2929, 3271}},
    {{1333, 1925, 2058, 3276, 2085, 2934, 3331}},
    {{1383, 1926, 2058, 3276, 2085, 2909, 3384}},
    {{1450, 1949, 2055, 3276, 2085, 2896, 3435}},
    {{1563, 2014, 2011, 3273, 2085, 2841, 3575}},
    {{1581, 2018, 2011, 3262, 2084, 2807, 3633}},
    {{1497, 2023, 2010, 3266, 2094, 2850, 3499}},
    {{1407, 2011, 2009, 3269, 2095, 2874, 3412}},
    {{1338, 1979, 2009, 3276, 2097, 2896, 3335}},
    {{1259, 1978, 2009, 3276, 2096, 2910, 3280}},
    {{1153, 1978, 2010, 3276, 2097, 2922, 3175}},
    {{1098, 1978, 2009, 3276, 2096, 2915, 3141}},
    {{1029, 1979, 2009, 3276, 2095, 2901, 3091}},
    {{947, 1979, 2010, 3276, 2097, 2905, 2993}},
    {{866, 1981, 2011, 3276, 2096, 2896, 2931}},
    {{768, 1992, 2012, 3276, 2097, 2866, 2835}},
    {{731, 2042, 2011, 3275, 2095, 2796, 2797}},
    {{743, 2065, 2013, 3274, 2086, 2775, 2847}},
    {{835, 2065, 2013, 3274, 2086, 2779, 2929}},
    {{905, 2067, 2013, 3274, 2085, 2774, 2980}},
    {{962, 2068, 2012, 3274, 2085, 2751, 3050}},
    {{1050, 2089, 2012, 3274, 2085, 2758, 3106}},
    {{1121, 2089, 2012, 3272, 2085, 2777, 3172}},
    {{1160, 2090, 2012, 3266, 2084, 2772, 3230}},
    {{1225, 2091, 2011, 3254, 2085, 2776, 3302}},
    {{1274, 2120, 2010, 3216, 2084, 2810, 3365}},
    {{1348, 2137, 2010, 3201, 2085, 2765, 3456}},
    {{1415, 2138, 2009, 3182, 2085, 2789, 3507}},
    {{1471, 2172, 2010, 3165, 2086, 2802, 3562}},
    {{1567, 2164, 2010, 3276, 2089, 2752, 3563}},
    {{1465, 2163, 2009, 3276, 2096, 2777, 3486}},
    {{1364, 2159, 2009, 3276, 2096, 2808, 3399}},
    {{1299, 2159, 2009, 3276, 2096, 2809, 3350}},
    {{1217, 2159, 2009, 3276, 2096, 2782, 3302}},
    {{1177, 1994, 2012, 3276, 2085, 2849, 3292}},
    {{1122, 2038, 2009, 3276, 2095, 2844, 3197}},
    {{1000, 2040, 2010, 3276, 2095, 2845, 3064}},
    {{896, 2086, 2010, 3277, 2095, 2788, 3004}},
    {{830, 2120, 2010, 3277, 2094, 2736, 2950}},
    {{755, 2128, 2010, 3276, 2096, 2721, 2836}},
    {{670, 2190, 2011, 3276, 2092, 2655, 2780}},
    {{670, 2251, 2012, 3246, 2087, 2628, 2781}},
    {{728, 2249, 2012, 3239, 2085, 2678, 2787}},
    {{772, 2213, 2012, 3274, 2084, 2674, 2845}},
    {{856, 2186, 2012, 3276, 2085, 2697, 2908}},
    {{914, 2186, 2012, 3276, 2085, 2696, 2973}},
    {{993, 2187, 2010, 3276, 2087, 2704, 3024}},
    {{1058, 2185, 2010, 3276, 2087, 2730, 3084}},
    {{1101, 2187, 2010, 3276, 2087, 2725, 3131}},
    {{1165, 2186, 2010, 3276, 2086, 2716, 3204}},
    {{1233, 2186, 2010, 3276, 2086, 2715, 3250}},
    {{1321, 2188, 2009, 3276, 2086, 2708, 3340}},
    {{1377, 2188, 2009, 3275, 2086, 2679, 3411}},
    {{1373, 2247, 2009, 3258, 2095, 2639, 3411}},
    {{1308, 2271, 2009, 3200, 2093, 2647, 3407}},
    {{1225, 2271, 2010, 3177, 2093, 2641, 3367}},
    {{1152, 2271, 2011, 3172, 2093, 2642, 3277}},
    {{1115, 2271, 2011, 3172, 2093, 2645, 3241}},
    {{1064, 2272, 2011, 3172, 2094, 2646, 3159}},
    {{1005, 2271, 2011, 3174, 2093, 2661, 3095}},
    {{943, 2273, 2011, 3176, 2092, 2661, 3048}},
    {{889, 2282, 2012, 3177, 2093, 2654, 2990}},
    {{838, 2302, 2011, 3174, 2093, 2633, 2940}},
    {{797, 2304, 2011, 3179, 2093, 2633, 2890}},
    {{730, 2313, 2011, 3179, 2093, 2633, 2791}},
    {{658, 2315, 2012, 3186, 2093, 2656, 2729}},
    {{619, 2312, 2011, 3276, 2093, 2576, 2664}},
    {{606, 2351, 2012, 3276, 2091, 2549, 2668}},
    {{677, 2375, 2012, 3276, 2082, 2486, 2795}},
    {{771, 2403, 2012, 3266, 2083, 2497, 2863}},
    {{854, 2403, 2011, 3273, 2085, 2470, 2958}},
    {{919, 2403, 2012, 3273, 2082, 2469, 3036}},
    {{1010, 2401, 2012, 3265, 2083, 2488, 3119}},
    {{1114, 2402, 2010, 3265, 2085, 2483, 3222}},
    {{1154, 2399, 2010, 3262, 2086, 2504, 3225}},
    {{1173, 2400, 2010, 3261, 2085, 2506, 3225}},
    {{1224, 2401, 2010, 3255, 2086, 2511, 3285}},
    {{1264, 2402, 2010, 3247, 2085, 2516, 3336}},
    {{1307, 2402, 2010, 3244, 2086, 2550, 3339}},
    {{1364, 2403, 2010, 3235, 2087, 2573, 3394}},
    {{1414, 2403, 2010, 3228, 2086, 2583, 3443}},
    {{1486, 2403, 2009, 3239, 2088, 2589, 3511}},
    {{1383, 2398, 2009, 3276, 2099, 2569, 3413}},
    {{1279, 2399, 2009, 3276, 2098, 2573, 3321}},
    {{1215, 2396, 2009, 3276, 2099, 2675, 3288}},
    {{1140, 2397, 2009, 3276, 2098, 2742, 3247}},
    {{1013, 2398, 2008, 3275, 2097, 2807, 3197}},
    {{896, 2397, 2008, 3275, 2098, 2766, 3027}},
    {{811, 2399, 2008, 3276, 2099, 2693, 2925}},
    {{752, 2399, 2009, 3275, 2099, 2559, 2869}},
    {{746, 2400, 2010, 3274, 2096, 2493, 2865}},
    {{747, 2403, 2010, 3273, 2089, 2405, 2866}},
    {{783, 2405, 2012, 3271, 2082, 2377, 2898}},
    {{889, 2404, 2012, 3271, 2084, 2351, 3020}},
    {{938, 2403, 2011, 3271, 2085, 2363, 3034}},
    {{963, 2404, 2012, 3271, 2085, 2361, 3033}},
    {{1038, 2403, 2010, 3271, 2084, 2392, 3103}},
    {{1089, 2403, 2010, 3271, 2086, 2409, 3143}},
    {{1130, 2403, 2009, 3271, 2087, 2433, 3167}},
    {{1162, 2403, 2010, 3270, 2087, 2444, 3178}},
    {{1217, 2403, 2009, 3267, 2086, 2463, 3253}},
    {{1271, 2403, 2010, 3258, 2088, 2471, 3308}},
    {{1314, 2403, 2010, 3254, 2086, 2492, 3373}},
    {{1369, 2405, 2010, 3252, 2086, 2487, 3465}},
    {{1405, 2405, 2011, 3252, 2086, 2595, 3465}},
    {{1423, 2405, 2011, 3257, 2087, 2693, 3461}},
    {{1421, 2402, 2009, 3275, 2093, 2761, 3429}},
    {{1396, 2401, 2007, 3276, 2096, 2785, 3410}},
    {{1274, 2402, 1982, 3276, 2099, 2749, 3305}},
    {{1223, 2402, 1906, 3276, 2098, 2723, 3303}},
    {{1178, 2406, 1873, 3277, 2097, 2679, 3304}},
    {{1168, 2411, 1874, 3276, 2094, 2638, 3309}},
    {{1172, 2428, 1875, 3276, 2086, 2560, 3311}},
    {{1203, 2502, 1875, 3275, 2082, 2475, 3386}},
    {{1279, 2497, 1875, 3274, 2081, 2474, 3464}},
    {{1351, 2499, 1875, 3273, 2079, 2428, 3574}},
    {{1376, 2514, 1875, 3273, 2081, 2374, 3577}},
    {{1375, 2596, 1874, 3251, 2090, 2306, 3575}},
    {{1362, 2609, 1873, 3275, 2096, 2299, 3511}},
    {{1314, 2616, 1872, 3275, 2097, 2289, 3465}},
    {{1267, 2616, 1872, 3276, 2097, 2287, 3424}},
    {{1215, 2618, 1872, 3276, 2096, 2302, 3380}},
    {{1158, 2619, 1872, 3276, 2097, 2287, 3353}},
    {{1095, 2625, 1872, 3276, 2095, 2288, 3310}},
    {{1039, 2640, 1872, 3276, 2096, 2276, 3236}},
    {{975, 2641, 1872, 3276, 2097, 2269, 3175}},
    {{908, 2643, 1872, 3276, 2095, 2245, 3133}},
    {{835, 2669, 1873, 3275, 2095, 2182, 3060}},
    {{828, 2695, 1874, 3274, 2091, 2187, 3062}},
    {{871, 2713, 1874, 3273, 2085, 2175, 3108}},
    {{941, 2710, 1875, 3273, 2086, 2184, 3161}},
    {{989, 2709, 1875, 3273, 2086, 2212, 3162}},
    {{1042, 2709, 1874, 3271, 2085, 2208, 3219}},
    {{1091, 2709, 1874, 3271, 2086, 2223, 3237}},
    {{1133, 2709, 1874, 3267, 2085, 2219, 3302}},
    {{1179, 2709, 1875, 3260, 2085, 2236, 3332}},
    {{1230, 2711, 1874, 3242, 2085, 2260, 3382}},
    {{1261, 2711, 1875, 3228, 2085, 2285, 3383}},
    {{1317, 2715, 1874, 3186, 2085, 2325, 3484}},
    {{1383, 2714, 1874, 3131, 2085, 2365, 3571}},
    {{1396, 2712, 1874, 3083, 2090, 2395, 3586}},
    {{1332, 2720, 1874, 3081, 2093, 2404, 3527}},
    {{1285, 2719, 1873, 3086, 2094, 2399, 3462}},
    {{1239, 2718, 1874, 3087, 2094, 2409, 3405}},
    {{1188, 2719, 1874, 3094, 2094, 2407, 3354}},
    {{1147, 2719, 1873, 3127, 2096, 2369, 3307}},
    {{1106, 2718, 1873, 3163, 2096, 2307, 3268}},
    {{1060, 2719, 1873, 3171, 2095, 2297, 3238}},
    {{985, 2719, 1874, 3175, 2095, 2264, 3180}},
    {{928, 2718, 1873, 3167, 2096, 2262, 3147}},
    {{903, 2746, 1872, 3118, 2089, 2284, 3148}},
    {{949, 2757, 1873, 3053, 2088, 2386, 3191}},
    {{1025, 2756, 1873, 3050, 2089, 2410, 3235}},
    {{1076, 2756, 1873, 3044, 2086, 2416, 3288}},
    {{1134, 2754, 1872, 3039, 2086, 2417, 3339}},
    {{1174, 2753, 1873, 3013, 2085, 2454, 3368}},
    {{1221, 2755, 1873, 2982, 2086, 2491, 3405}},
    {{1289, 2756, 1873, 2973, 2086, 2503, 3470}},
    {{1344, 2758, 1873, 2941, 2087, 2511, 3540}},
    {{1301, 2755, 1873, 2893, 2094, 2535, 3490}},
    {{1241, 2765, 1873, 2891, 2095, 2532, 3451}},
    {{1170, 2770, 1873, 2890, 2095, 2534, 3393}},
    {{1114, 2771, 1873, 2890, 2094, 2527, 3349}},
    {{1060, 2771, 1874, 2891, 2095, 2550, 3286}},
    {{1010, 2771, 1874, 2893, 2095, 2559, 3242}},
    {{959, 2772, 1874, 2892, 2095, 2548, 3205}},
    {{912, 2771, 1873, 2893, 2094, 2534, 3169}},
    {{855, 2772, 1873, 2892, 2095, 2520, 3103}},
    {{806, 2782, 1874, 2892, 2095, 2506, 3056}},
    {{864, 2630, 1875, 2890, 2088, 2688, 3062}},
    {{861, 2637, 1875, 2894, 2088, 2655, 3061}},
    {{861, 2685, 1875, 2894, 2087, 2651, 3061}},
    {{863, 2723, 1875, 2894, 2087, 2610, 3061}},
    {{862, 2753, 1874, 2893, 2087, 2558, 3061}},
    {{862, 2806, 1875, 2893, 2088, 2542, 3061}},
    {{862, 2831, 1874, 2891, 2087, 2505, 3061}},
    {{862, 2868, 1875, 2891, 2088, 2472, 3061}},
    {{862, 2906, 1875, 2892, 2088, 2432, 3061}},
    {{861, 2941, 1874, 2892, 2089, 2406, 3059}},
    {{926, 2929, 1873, 2889, 2087, 2432, 3143}},
    {{965, 2926, 1874, 2889, 2088, 2428, 3149}},
    {{1006, 2927, 1873, 2887, 2087, 2420, 3202}},
    {{1052, 2927, 1873, 2888, 2087, 2404, 3214}},
    {{1094, 2926, 1873, 2887, 2087, 2391, 3261}},
    {{1144, 2927, 1873, 2887, 2087, 2384, 3289}},
    {{1201, 2927, 1873, 2887, 2087, 2371, 3350}},
    {{1233, 2931, 1873, 2851, 2087, 2419, 3395}},
    {{1257, 2982, 1872, 2731, 2087, 2503, 3442}},
    {{1309, 3005, 1873, 2699, 2087, 2556, 3469}},
    {{1374, 2898, 1873, 2895, 2085, 2455, 3525}},
    {{1426, 2887, 1874, 3032, 2087, 2325, 3525}},
    {{1396, 2847, 1873, 3149, 2089, 2214, 3518}},
    {{1346, 2848, 1873, 3187, 2096, 2186, 3478}},
    {{1254, 2849, 1873, 3187, 2094, 2190, 3385}},
    {{1178, 2850, 1873, 3190, 2095, 2204, 3301}},
    {{1129, 2850, 1873, 3192, 2095, 2214, 3270}},
    {{1084, 2849, 1873, 3193, 2098, 2222, 3226}},
    {{997, 2851, 1873, 3192, 2096, 2217, 3152}},
    {{937, 2851, 1873, 3191, 2095, 2209, 3105}},
    {{883, 2851, 1873, 3191, 2095, 2192, 3036}},
    {{818, 2855, 1873, 3186, 2093, 2128, 2981}},
    {{871, 2875, 1873, 3186, 2086, 2163, 3014}},
    {{920, 2877, 1874, 3189, 2084, 2162, 3022}},
    {{975, 2876, 1874, 3185, 2084, 2165, 3087}},
    {{1063, 2875, 1874, 3186, 2085, 2177, 3153}},
    {{1156, 2875, 1874, 3186, 2085, 2158, 3247}},
    {{1200, 2873, 1874, 3173, 2082, 2165, 3297}},
    {{1251, 2873, 1874, 3098, 2083, 2245, 3402}},
    {{1312, 2876, 1872, 3051, 2082, 2275, 3469}},
    {{1383, 2874, 1873, 3039, 2083, 2284, 3524}},
    {{1430, 2874, 1874, 2956, 2085, 2386, 3566}},
    {{1401, 2899, 1874, 2855, 2092, 2437, 3564}},
    {{1365, 2928, 1874, 2812, 2095, 2488, 3560}},
    {{1275, 2930, 1874, 2797, 2094, 2496, 3497}},
    {{1202, 2930, 1874, 2797, 2094, 2515, 3421}},
    {{1146, 2930, 1874, 2797, 2094, 2516, 3387}},
    {{1065, 2931, 1874, 2798, 2094, 2510, 3284}},
    {{1002, 2930, 1874, 2797, 2093, 2499, 3242}},
    {{951, 2930, 1873, 2797, 2093, 2497, 3196}},
    {{900, 2931, 1874, 2800, 2094, 2494, 3121}},
    {{826, 2934, 1874, 2831, 2095, 2494, 3015}},
    {{786, 2902, 1873, 3001, 2087, 2385, 2963}},
    {{841, 2830, 1874, 3187, 2086, 2199, 2965}},
    {{911, 2828, 1874, 3274, 2082, 2092, 3012}},
    {{1006, 2821, 1875, 3275, 2082, 2114, 3078}},
    {{1078, 2825, 1875, 3276, 2082, 2120, 3140}},
    {{1171, 2828, 1875, 3276, 2081, 2104, 3241}},
    {{1270, 2828, 1875, 3274, 2081, 2085, 3332}},
    {{1365, 2828, 1875, 3267, 2081, 2083, 3452}},
    {{1429, 2832, 1874, 3230, 2084, 2129, 3524}},
    {{1434, 2850, 1874, 3103, 2090, 2287, 3525}},
    {{1382, 2869, 1873, 3049, 2094, 2288, 3518}},
    {{1311, 2870, 1873, 3052, 2096, 2297, 3466}},
    {{1226, 2871, 1873, 3059, 2095, 2293, 3368}},
    {{1165, 2870, 1873, 3083, 2095, 2266, 3323}},
    {{1102, 2870, 1873, 3091, 2097, 2269, 3255}},
    {{1025, 2869, 1872, 3091, 2098, 2260, 3212}},
    {{952, 2870, 1873, 3090, 2097, 2230, 3134}},
    {{884, 2873, 1873, 3085, 2096, 2185, 3058}},
    {{783, 2596, 1874, 3087, 2088, 2476, 3061}},
    {{785, 2596, 1874, 3095, 2085, 2550, 3060}},
    {{824, 2590, 1875, 3159, 2084, 2513, 3063}},
    {{872, 2563, 1875, 3219, 2084, 2466, 3061}},
    {{938, 2559, 1875, 3273, 2084, 2415, 3065}},
    {{1007, 2533, 1875, 3274, 2084, 2465, 3065}},
    {{1122, 2534, 1875, 3275, 2080, 2475, 3203}},
    {{1228, 2537, 1875, 3274, 2081, 2454, 3310}},
    {{1313, 2538, 1876, 3273, 2079, 2417, 3432}},
    {{1409, 2538, 1875, 3268, 2080, 2368, 3565}},
    {{1390, 2581, 1874, 3146, 2092, 2438, 3605}},
    {{1322, 2593, 1873, 3134, 2093, 2476, 3517}},
    {{1243, 2592, 1873, 3162, 2093, 2456, 3440}},
    {{1168, 2592, 1873, 3189, 2093, 2422, 3354}},
    {{1067, 2592, 1874, 3193, 2093, 2402, 3254}},
    {{969, 2592, 1874, 3192, 2093, 2423, 3147}},
    {{892, 2596, 1874, 3187, 2092, 2413, 3068}},
    {{823, 2605, 1875, 3141, 2089, 2353, 3070}},
    {{830, 2634, 1874, 3048, 2084, 2465, 3073}},
    {{927, 2674, 1873, 3018, 2086, 2491, 3168}},
    {{1019, 2674, 1874, 3010, 2085, 2502, 3233}},
    {{1096, 2674, 1873, 2994, 2086, 2526, 3316}},
    {{1159, 2672, 1875, 2988, 2086, 2555, 3352}},
    {{1211, 2673, 1874, 2986, 2086, 2545, 3385}},
    {{1274, 2673, 1874, 2986, 2086, 2547, 3458}},
    {{1338, 2673, 1875, 2987, 2087, 2563, 3501}},
    {{1405, 2677, 1874, 2990, 2086, 2600, 3550}},
    {{1417, 2670, 1874, 3012, 2088, 2613, 3546}},
    {{1355, 2661, 1875, 3085, 2089, 2560, 3487}},
    {{1285, 2629, 1874, 3133, 2088, 2485, 3465}},
    {{1203, 2624, 1874, 3204, 2089, 2437, 3368}},
    {{1115, 2623, 1874, 3252, 2090, 2370, 3272}},
    {{994, 2628, 1874, 3264, 2088, 2308, 3147}},
    {{860, 2631, 1874, 3267, 2089, 2270, 2959}},
    {{818, 2638, 1875, 3160, 2084, 2317, 3009}},
    {{854, 2664, 1875, 3036, 2083, 2517, 3040}},
    {{1006, 2676, 1875, 3021, 2084, 2475, 3227}},
    {{1102, 2675, 1875, 3005, 2084, 2482, 3332}},
    {{1182, 2673, 1874, 2973, 2083, 2529, 3414}},
    {{1311, 2683, 1874, 2946, 2086, 2545, 3511}},
    {{1321, 2760, 1874, 2954, 2087, 2503, 3511}},
    {{1371, 2821, 1873, 2969, 2086, 2481, 3524}},
    {{1374, 2834, 1873, 3065, 2095, 2352, 3524}},
    {{1291, 2854, 1873, 3168, 2098, 2210, 3454}},
    {{1155, 2860, 1873, 3180, 2098, 2163, 3349}},
    {{1057, 2860, 1873, 3180, 2099, 2154, 3233}},
    {{930, 2861, 1873, 3176, 2098, 2143, 3094}},
    {{858, 2863, 1874, 3121, 2096, 2140, 3082}},
    {{870, 2883, 1872, 2967, 2087, 2331, 3138}},
    {{971, 2919, 1872, 2919, 2086, 2364, 3247}},
    {{1076, 2899, 1872, 2906, 2088, 2396, 3338}},
    {{1169, 2898, 1872, 2906, 2089, 2412, 3411}},
    {{1251, 2898, 1872, 2905, 2090, 2411, 3462}},
    {{1324, 2901, 1872, 2906, 2092, 2420, 3533}},
    {{1321, 2949, 1873, 3051, 2102, 2254, 3467}},
    {{1243, 2894, 1873, 3202, 2102, 2153, 3361}},
    {{1088, 2891, 1873, 3273, 2104, 2020, 3242}},
    {{919, 2890, 1873, 3262, 2101, 1978, 3108}},
    {{817, 2881, 1873, 3147, 2097, 2161, 3072}},
    {{824, 2888, 1872, 2995, 2086, 2311, 3077}},
    {{885, 2889, 1872, 2978, 2087, 2323, 3134}},
    {{966, 2887, 1872, 2965, 2086, 2342, 3198}},
    {{1071, 2890, 1873, 2928, 2088, 2371, 3260}},
    {{1149, 2890, 1872, 2917, 2087, 2378, 3365}},
    {{1217, 2892, 1872, 2911, 2086, 2386, 3430}},
    {{1314, 2892, 1871, 2911, 2088, 2406, 3514}},
    {{1369, 2910, 1872, 2924, 2092, 2409, 3512}},
    {{1317, 2931, 1873, 3045, 2100, 2273, 3442}},
    {{1218, 2924, 1873, 3149, 2100, 2152, 3334}},
    {{1042, 2887, 1873, 3214, 2099, 2126, 3194}},
    {{952, 2796, 1873, 3213, 2095, 2218, 3113}},
    {{912, 2699, 1874, 3187, 2091, 2307, 3113}},
    {{934, 2650, 1875, 3123, 2082, 2401, 3117}},
    {{1005, 2652, 1875, 3077, 2084, 2430, 3204}},
    {{1079, 2652, 1874, 3062, 2083, 2448, 3297}},
    {{1168, 2653, 1874, 3057, 2083, 2442, 3393}},
    {{1243, 2654, 1875, 3056, 2082, 2444, 3463}},
    {{1334, 2662, 1874, 3060, 2084, 2433, 3531}},
    {{1371, 2695, 1874, 3060, 2086, 2410, 3532}},
    {{1372, 2752, 1874, 3061, 2086, 2375, 3531}},
    {{1369, 2840, 1873, 3068, 2092, 2321, 3527}},
    {{1361, 2921, 1873, 3113, 2097, 2213, 3525}},
    {{1238, 2929, 1872, 3166, 2102, 2168, 3410}},
    {{1086, 2926, 1873, 3165, 2100, 2144, 3292}},
    {{903, 2749, 1874, 3156, 2094, 2227, 3212}},
    {{905, 2642, 1874, 3154, 2083, 2351, 3212}},
    {{1022, 2608, 1874, 3151, 2084, 2382, 3312}},
    {{1031, 2762, 1874, 3155, 2086, 2271, 3312}},
    {{1027, 2868, 1873, 3152, 2093, 2163, 3307}},
    {{1027, 2944, 1873, 3129, 2098, 2119, 3307}},
    {{1027, 3010, 1873, 3129, 2099, 2050, 3307}},
    // 159 poses (2026-08-07), index 1239-1397: recorded_hand_poses_fixed_
    // elbow_yaw(2).csv -- a second batch of the reduced-DOF (elbow_yaw
    // locked near 2095) hand-posing session, collected for a fresh
    // calibration refit, not an accuracy test. Verified all 159 fall
    // within the current jointCalibrations bounds before wiring in.
    {{3094, 2079, 2073, 828, 2088, 1244, 1062}},
    {{3174, 2047, 2001, 829, 2091, 1295, 1061}},
    {{3238, 2047, 1959, 829, 2092, 1276, 1165}},
    {{3300, 2032, 1899, 828, 2092, 1213, 1164}},
    {{3276, 2033, 1948, 828, 2087, 1048, 1165}},
    {{3273, 2039, 2073, 828, 2082, 1060, 1164}},
    {{3219, 2035, 2187, 829, 2088, 1058, 1144}},
    {{3220, 1978, 2189, 829, 2089, 1086, 1143}},
    {{3221, 1954, 2187, 828, 2092, 1197, 1146}},
    {{3219, 1928, 2189, 828, 2089, 1340, 1145}},
    {{3258, 1896, 2187, 829, 2097, 1395, 1147}},
    {{3337, 1853, 2187, 829, 2097, 1345, 1147}},
    {{3335, 1938, 2186, 829, 2096, 1416, 1311}},
    {{3340, 1879, 2139, 829, 2097, 1329, 1308}},
    {{3335, 1844, 2141, 828, 2082, 1214, 1305}},
    {{3199, 1960, 2141, 828, 2086, 1300, 1229}},
    {{3021, 1960, 2140, 828, 2088, 1352, 1050}},
    {{2902, 1943, 2139, 829, 2090, 1412, 933}},
    {{2869, 1873, 2139, 829, 2092, 1258, 802}},
    {{2978, 1836, 2031, 829, 2090, 1296, 903}},
    {{3190, 1837, 2031, 828, 2092, 1256, 1095}},
    {{3191, 1830, 2032, 829, 2089, 1439, 1157}},
    {{3268, 1827, 2029, 829, 2095, 1368, 1160}},
    {{3319, 1792, 2031, 828, 2090, 1243, 1159}},
    {{3244, 1852, 2031, 828, 2081, 1258, 1155}},
    {{3141, 1852, 2031, 828, 2081, 1355, 1155}},
    {{3096, 1852, 2032, 828, 2082, 1570, 1229}},
    {{3048, 1853, 2134, 828, 2084, 1398, 1062}},
    {{3046, 1849, 2256, 828, 2085, 1297, 1032}},
    {{2978, 1848, 2325, 828, 2086, 1397, 1122}},
    {{3096, 1785, 2323, 829, 2099, 1270, 1122}},
    {{3172, 1810, 2294, 830, 2095, 1151, 1205}},
    {{3270, 1842, 2292, 830, 2097, 1236, 1322}},
    {{3253, 1854, 2050, 828, 2089, 1255, 1315}},
    {{3254, 1874, 1945, 844, 2094, 1305, 1316}},
    {{3255, 1876, 1944, 842, 2093, 1470, 1271}},
    {{3250, 1912, 2066, 843, 2081, 1462, 1272}},
    {{3245, 1910, 2165, 835, 2081, 1421, 1273}},
    {{3146, 1913, 2168, 829, 2082, 1357, 1270}},
    {{3148, 1911, 2178, 829, 2090, 1192, 1199}},
    {{3198, 1910, 2128, 829, 2095, 1203, 1260}},
    {{3282, 1911, 2107, 829, 2096, 1323, 1267}},
    {{3304, 1917, 2100, 830, 2093, 1472, 1373}},
    {{3304, 1912, 2100, 829, 2092, 1201, 1286}},
    {{3301, 1912, 2234, 828, 2087, 1128, 1288}},
    {{3301, 1915, 1910, 829, 2086, 1294, 1313}},
    {{3301, 1916, 1898, 829, 2091, 1161, 1259}},
    {{3297, 1911, 1968, 828, 2078, 1150, 1245}},
    {{3297, 1910, 2067, 828, 2079, 1180, 1245}},
    {{3253, 1910, 2123, 828, 2081, 1286, 1255}},
    {{3204, 1916, 1976, 829, 2090, 1402, 1370}},
    {{3206, 1917, 1888, 830, 2097, 1283, 1224}},
    {{3206, 1916, 1992, 828, 2085, 1212, 1141}},
    {{3204, 1911, 2055, 828, 2082, 1211, 1160}},
    {{3188, 1911, 2186, 828, 2085, 1277, 1200}},
    {{3179, 1911, 2228, 827, 2083, 1355, 1253}},
    {{3087, 1911, 2204, 829, 2087, 1430, 1297}},
    {{3018, 1911, 2205, 829, 2094, 1208, 1036}},
    {{3143, 1856, 2083, 829, 2100, 1232, 1157}},
    {{3193, 1860, 1982, 829, 2098, 1288, 1190}},
    {{3275, 1858, 1982, 832, 2093, 1418, 1274}},
    {{3271, 1856, 1983, 832, 2082, 1512, 1272}},
    {{3160, 1854, 2086, 830, 2082, 1499, 1272}},
    {{3079, 1845, 2213, 828, 2083, 1365, 1271}},
    {{3078, 1749, 2229, 829, 2101, 1220, 1200}},
    {{3170, 1751, 2135, 829, 2100, 1227, 1202}},
    {{3236, 1753, 2066, 830, 2100, 1304, 1203}},
    {{3253, 1826, 1904, 829, 2086, 1514, 1201}},
    {{3144, 1836, 1921, 827, 2077, 1615, 1227}},
    {{3078, 1834, 2029, 828, 2079, 1562, 1227}},
    {{3040, 1830, 2108, 828, 2078, 1497, 1226}},
    {{3005, 1830, 2165, 828, 2080, 1362, 1178}},
    {{3115, 1793, 2012, 829, 2097, 1172, 1013}},
    {{3177, 1848, 1822, 828, 2082, 1463, 1103}},
    {{3120, 1848, 1944, 828, 2080, 1520, 1147}},
    {{3120, 1847, 2095, 828, 2080, 1485, 1180}},
    {{3116, 1846, 2193, 828, 2079, 1432, 1180}},
    {{3083, 1802, 2364, 829, 2092, 1295, 1224}},
    {{3141, 1777, 2248, 830, 2099, 1208, 1161}},
    {{3141, 1778, 2169, 830, 2101, 1197, 1150}},
    {{3141, 1778, 1959, 829, 2099, 1203, 1148}},
    {{3139, 1886, 1960, 827, 2082, 1345, 1121}},
    {{3129, 1884, 2022, 828, 2079, 1323, 1145}},
    {{3037, 1885, 2111, 828, 2081, 1321, 1144}},
    {{3016, 1866, 2240, 828, 2092, 1216, 1145}},
    {{3098, 1809, 2176, 829, 2101, 1156, 1088}},
    {{3148, 1812, 1996, 830, 2098, 1310, 1146}},
    {{3148, 1855, 1945, 830, 2100, 1322, 1145}},
    {{3159, 1986, 1896, 829, 2084, 1393, 1142}},
    {{3145, 1983, 1937, 829, 2077, 1400, 1142}},
    {{3058, 1984, 1996, 829, 2083, 1400, 1141}},
    {{3054, 1983, 1997, 829, 2083, 1398, 1141}},
    {{3049, 1984, 1997, 829, 2083, 1396, 1141}},
    {{3018, 1983, 1998, 829, 2082, 1388, 1141}},
    {{2963, 1983, 2001, 827, 2077, 1337, 1140}},
    {{2943, 1972, 2156, 828, 2088, 1252, 1114}},
    {{3108, 1979, 2083, 829, 2099, 1176, 1119}},
    {{3161, 1979, 2011, 829, 2098, 1301, 1204}},
    {{3181, 1970, 2055, 828, 2077, 1448, 1286}},
    {{3081, 1973, 2219, 828, 2083, 1379, 1203}},
    {{2995, 1976, 2219, 829, 2089, 1222, 1009}},
    {{3281, 1998, 2219, 829, 2100, 1236, 1191}},
    {{3319, 1946, 2218, 829, 2101, 1259, 1192}},
    {{3368, 1945, 2218, 830, 2099, 1322, 1309}},
    {{3364, 1846, 2219, 829, 2090, 1366, 1308}},
    {{3205, 1832, 2220, 829, 2084, 1403, 1276}},
    {{3118, 1872, 2218, 829, 2094, 1143, 1050}},
    {{3158, 1873, 2195, 829, 2097, 1121, 1124}},
    {{3273, 1874, 2055, 829, 2096, 1176, 1123}},
    {{3312, 1876, 2034, 830, 2095, 1264, 1122}},
    {{3270, 1937, 2064, 828, 2077, 1452, 1190}},
    {{3150, 1940, 2130, 828, 2079, 1418, 1189}},
    {{3077, 1939, 2190, 828, 2080, 1335, 1188}},
    {{3049, 1938, 2213, 829, 2093, 1178, 1029}},
    {{3066, 1939, 2210, 829, 2099, 1088, 982}},
    {{3193, 1938, 2198, 829, 2099, 1154, 1127}},
    {{3260, 1938, 2105, 829, 2098, 1219, 1128}},
    {{3272, 1937, 2009, 829, 2098, 1326, 1128}},
    {{3272, 1937, 1985, 829, 2090, 1462, 1222}},
    {{3226, 1934, 1998, 828, 2078, 1463, 1220}},
    {{3138, 1936, 2027, 828, 2080, 1452, 1219}},
    {{3079, 1936, 2065, 828, 2079, 1451, 1219}},
    {{3003, 1939, 2168, 828, 2081, 1345, 1127}},
    {{3005, 1941, 2199, 829, 2091, 1219, 1106}},
    {{3005, 1938, 2132, 829, 2101, 1119, 1098}},
    {{3092, 1920, 2073, 829, 2100, 1118, 1110}},
    {{3317, 1877, 2060, 829, 2098, 1229, 1245}},
    {{3366, 1868, 2062, 851, 2087, 1411, 1331}},
    {{3269, 1825, 2063, 851, 2081, 1433, 1328}},
    {{3146, 1827, 2066, 848, 2080, 1408, 1283}},
    {{3059, 1829, 2066, 832, 2080, 1391, 1191}},
    {{2995, 1847, 2066, 827, 2079, 1348, 1098}},
    {{2957, 1920, 2066, 827, 2080, 1223, 1048}},
    {{2964, 1970, 2066, 828, 2095, 1122, 1049}},
    {{3145, 1966, 2021, 829, 2096, 1270, 1172}},
    {{3143, 1957, 2021, 828, 2075, 1417, 1201}},
    {{3071, 1958, 2183, 828, 2081, 1391, 1201}},
    {{3063, 1958, 2269, 828, 2086, 1303, 1202}},
    {{3065, 1934, 2280, 829, 2096, 1206, 1150}},
    {{3066, 1899, 2209, 829, 2100, 1146, 1149}},
    {{3071, 1899, 2125, 829, 2100, 1128, 1148}},
    {{3135, 1898, 2079, 829, 2099, 1124, 1152}},
    {{3258, 1897, 2009, 829, 2099, 1201, 1219}},
    {{3283, 1892, 1955, 829, 2098, 1250, 1219}},
    {{3280, 1868, 1942, 828, 2075, 1440, 1219}},
    {{3251, 1870, 2026, 828, 2079, 1452, 1219}},
    {{3197, 1871, 2073, 828, 2080, 1447, 1220}},
    {{3125, 1871, 2074, 828, 2079, 1428, 1218}},
    {{2991, 1871, 2162, 827, 2078, 1356, 1127}},
    {{2984, 1874, 2239, 829, 2095, 1258, 1113}},
    {{3056, 1874, 2163, 829, 2100, 1173, 1115}},
    {{3190, 1872, 2048, 829, 2099, 1200, 1115}},
    {{3247, 1871, 2011, 829, 2099, 1245, 1116}},
    {{3295, 1870, 2002, 829, 2096, 1363, 1212}},
    {{3301, 1867, 2003, 829, 2079, 1501, 1261}},
    {{3229, 1867, 2011, 828, 2075, 1515, 1261}},
    {{3143, 1867, 2049, 828, 2075, 1515, 1260}},
    {{3039, 1868, 2155, 828, 2079, 1482, 1255}},
    {{3005, 1868, 2224, 828, 2083, 1420, 1206}},
    // 221 poses (2026-08-11), index 1398-1618: recorded_hand_poses_fixed_
    // elbow_yaw(3).csv -- ONLY the genuinely new poses from the ongoing
    // recording session (rows 34-254 of that file; extends the earlier
    // 141-pose version of this same range with 80 more poses added in a
    // later sitting -- same session, legitimately resumed/appended this
    // time, not the unrelated-session mixup documented in CLAUDE.md). The
    // file's OUTPUT_CSV path is hardcoded, so rows 1-33 (a genuinely
    // unrelated 2026-08-07 session) are still excluded here. Third batch
    // of the reduced-DOF (elbow_yaw locked near 2095) hand-posing session --
    // see CLAUDE.md's kinematic-calibration section.
    {{3122, 1886, 2117, 828, 2093, 1379, 1169}},
    {{3122, 1887, 2117, 828, 2094, 1285, 1167}},
    {{3111, 1884, 2220, 828, 2085, 1269, 1132}},
    {{3037, 1890, 2236, 829, 2090, 1467, 1161}},
    {{3041, 1891, 2168, 829, 2099, 1525, 1193}},
    {{3074, 1892, 2128, 829, 2099, 1525, 1193}},
    {{3143, 1889, 2045, 829, 2100, 1497, 1219}},
    {{3204, 1885, 1977, 829, 2096, 1382, 1169}},
    {{3209, 1880, 1979, 829, 2092, 1295, 1168}},
    {{3207, 1809, 1991, 827, 2078, 1262, 1168}},
    {{3206, 1823, 2112, 828, 2082, 1271, 1169}},
    {{3116, 1864, 2148, 828, 2080, 1320, 1167}},
    {{3073, 1866, 2211, 829, 2090, 1523, 1243}},
    {{3074, 1866, 2211, 828, 2095, 1553, 1259}},
    {{3079, 1861, 2081, 829, 2098, 1552, 1264}},
    {{3133, 1844, 1944, 829, 2093, 1489, 1199}},
    {{3135, 1845, 1944, 829, 2095, 1366, 1199}},
    {{3171, 1815, 2022, 828, 2089, 1303, 1201}},
    {{3143, 1845, 2144, 828, 2090, 1251, 1201}},
    {{3084, 1836, 2277, 829, 2090, 1311, 1101}},
    {{3084, 1842, 2277, 828, 2088, 1445, 1099}},
    {{3085, 1843, 2275, 829, 2090, 1466, 1103}},
    {{3085, 1857, 2148, 829, 2098, 1537, 1105}},
    {{3141, 1856, 2041, 830, 2099, 1488, 1105}},
    {{3197, 1856, 2007, 829, 2099, 1406, 1072}},
    {{3213, 1833, 2009, 828, 2084, 1303, 1009}},
    {{3189, 1836, 2087, 829, 2083, 1286, 1038}},
    {{3102, 1836, 2170, 828, 2084, 1310, 1039}},
    {{3071, 1832, 2275, 829, 2091, 1479, 1152}},
    {{3064, 1833, 2110, 829, 2096, 1537, 1105}},
    {{3066, 1834, 1993, 829, 2096, 1525, 1032}},
    {{3066, 1836, 1980, 829, 2099, 1482, 1023}},
    {{3067, 1836, 1979, 829, 2099, 1479, 1023}},
    {{3067, 1836, 1978, 829, 2099, 1477, 1023}},
    {{3067, 1837, 1976, 829, 2100, 1476, 1023}},
    {{3067, 1837, 1975, 829, 2100, 1474, 1023}},
    {{3067, 1838, 1974, 829, 2100, 1471, 1023}},
    {{3067, 1838, 1972, 829, 2101, 1470, 1022}},
    {{3067, 1838, 1970, 829, 2102, 1466, 1020}},
    {{3067, 1838, 1968, 829, 2102, 1460, 1013}},
    {{3067, 1837, 1966, 829, 2102, 1451, 1005}},
    {{3067, 1837, 1962, 829, 2102, 1442, 998}},
    {{3067, 1836, 1960, 829, 2102, 1436, 997}},
    {{3067, 1836, 1957, 829, 2100, 1432, 997}},
    {{3067, 1835, 1954, 829, 2100, 1430, 996}},
    {{3067, 1836, 1952, 829, 2100, 1427, 996}},
    {{3104, 1836, 1943, 829, 2099, 1399, 998}},
    {{3167, 1831, 2051, 829, 2087, 1303, 999}},
    {{3127, 1830, 2068, 828, 2079, 1302, 1029}},
    {{3071, 1830, 2140, 828, 2082, 1337, 1029}},
    {{3037, 1830, 2219, 827, 2085, 1397, 1002}},
    {{3021, 1831, 2223, 828, 2085, 1545, 1130}},
    {{3021, 1832, 2115, 831, 2098, 1591, 1123}},
    {{3065, 1831, 2022, 830, 2095, 1539, 1031}},
    {{3066, 1830, 2078, 828, 2087, 1424, 1019}},
    {{3066, 1830, 2083, 828, 2089, 1342, 1018}},
    {{3015, 1781, 2084, 827, 2083, 1328, 1015}},
    {{3015, 1781, 2083, 848, 2087, 1392, 1018}},
    {{3015, 1766, 2084, 913, 2088, 1436, 1069}},
    {{3080, 1738, 2083, 926, 2093, 1420, 1101}},
    {{3098, 1750, 2082, 907, 2093, 1389, 1102}},
    {{3100, 1792, 2083, 829, 2093, 1407, 1101}},
    {{3093, 1787, 2083, 828, 2084, 1363, 1072}},
    {{3018, 1785, 2084, 827, 2084, 1380, 1041}},
    {{2964, 1783, 2103, 829, 2084, 1414, 1003}},
    {{2941, 1783, 2110, 855, 2090, 1471, 1008}},
    {{2966, 1759, 2033, 912, 2101, 1483, 1007}},
    {{3131, 1777, 1993, 897, 2098, 1320, 956}},
    {{3133, 1777, 1995, 844, 2091, 1303, 956}},
    {{3132, 1776, 2002, 827, 2076, 1299, 975}},
    {{3105, 1793, 2090, 828, 2083, 1337, 1020}},
    {{3081, 1792, 2194, 828, 2085, 1407, 1098}},
    {{3045, 1840, 2057, 828, 2094, 1527, 1068}},
    {{3045, 1840, 1940, 828, 2096, 1514, 1012}},
    {{3074, 1838, 1873, 829, 2095, 1418, 950}},
    {{3075, 1841, 1942, 828, 2090, 1294, 945}},
    {{3073, 1837, 2028, 828, 2088, 1294, 957}},
    {{3073, 1837, 2104, 828, 2085, 1315, 991}},
    {{3047, 1838, 2169, 828, 2081, 1362, 995}},
    {{3044, 1836, 2252, 828, 2082, 1449, 1089}},
    {{3045, 1837, 2210, 829, 2096, 1543, 1147}},
    {{3046, 1838, 2087, 829, 2098, 1544, 1095}},
    {{3105, 1842, 2034, 829, 2096, 1475, 1061}},
    {{3153, 1842, 2034, 828, 2095, 1413, 1092}},
    {{3160, 1837, 2036, 828, 2091, 1360, 1091}},
    {{3109, 1836, 2076, 829, 2081, 1359, 1051}},
    {{3103, 1837, 2091, 830, 2087, 1443, 1052}},
    {{3072, 1834, 2093, 828, 2082, 1455, 1048}},
    {{2995, 1834, 2094, 827, 2081, 1414, 957}},
    {{3005, 1839, 1980, 829, 2094, 1303, 931}},
    {{3063, 1838, 1980, 828, 2091, 1448, 985}},
    {{3063, 1837, 1979, 828, 2093, 1497, 985}},
    {{3058, 1832, 2045, 828, 2082, 1517, 1004}},
    {{3016, 1833, 2090, 828, 2080, 1448, 1018}},
    {{3008, 1828, 2186, 828, 2097, 1362, 1021}},
    {{3082, 1804, 2098, 829, 2101, 1309, 1021}},
    {{3084, 1807, 2038, 828, 2099, 1333, 1000}},
    {{3165, 1807, 2038, 830, 2095, 1414, 1083}},
    {{3163, 1804, 2045, 830, 2081, 1536, 1110}},
    {{3140, 1803, 2121, 830, 2082, 1563, 1112}},
    {{3043, 1802, 2149, 829, 2082, 1525, 1091}},
    {{3014, 1801, 2149, 828, 2080, 1450, 1074}},
    {{3015, 1802, 2149, 828, 2087, 1365, 1075}},
    {{3060, 1803, 2144, 829, 2101, 1299, 1078}},
    {{3139, 1754, 2057, 831, 2099, 1344, 1077}},
    {{3156, 1754, 2057, 878, 2099, 1369, 1104}},
    {{3161, 1751, 2058, 899, 2087, 1412, 1134}},
    {{3050, 1750, 2112, 916, 2081, 1418, 1121}},
    {{3011, 1767, 2153, 832, 2092, 1395, 1079}},
    {{3012, 1812, 2152, 829, 2095, 1367, 1079}},
    {{3064, 1811, 2151, 829, 2099, 1335, 1080}},
    {{3121, 1807, 2071, 829, 2096, 1354, 1081}},
    {{3122, 1807, 2067, 830, 2090, 1419, 1081}},
    {{3079, 1802, 2077, 840, 2076, 1443, 1077}},
    {{3130, 1780, 1962, 835, 2092, 1385, 1078}},
    {{3132, 1777, 1962, 890, 2095, 1408, 1080}},
    {{3065, 1776, 2110, 859, 2084, 1508, 1078}},
    {{3057, 1776, 2211, 831, 2087, 1452, 1082}},
    {{3059, 1778, 2216, 828, 2084, 1334, 1080}},
    {{3139, 1774, 2214, 829, 2097, 1263, 1083}},
    {{3158, 1773, 2092, 829, 2095, 1347, 1103}},
    {{3158, 1774, 2077, 836, 2097, 1408, 1106}},
    {{3155, 1772, 2078, 866, 2085, 1471, 1104}},
    {{3037, 1771, 2176, 870, 2084, 1489, 1063}},
    {{3037, 1771, 2251, 829, 2087, 1419, 1083}},
    {{3039, 1766, 2172, 829, 2098, 1315, 1049}},
    {{3055, 1770, 2083, 829, 2099, 1318, 1049}},
    {{3095, 1770, 2069, 832, 2093, 1368, 1052}},
    {{3095, 1768, 2069, 850, 2095, 1455, 1081}},
    {{3098, 1770, 2059, 847, 2100, 1457, 1098}},
    {{3098, 1768, 1990, 828, 2091, 1339, 1061}},
    {{3096, 1767, 2090, 828, 2087, 1325, 1062}},
    {{3075, 1767, 2099, 828, 2084, 1425, 1063}},
    {{3054, 1767, 2099, 831, 2087, 1485, 1065}},
    {{3056, 1768, 2032, 832, 2094, 1482, 1093}},
    {{3053, 1766, 2146, 829, 2083, 1469, 1062}},
    {{2998, 1769, 2147, 828, 2083, 1391, 1012}},
    {{3000, 1774, 2146, 828, 2094, 1313, 997}},
    {{3024, 1925, 2144, 829, 2097, 1338, 1002}},
    {{3023, 1993, 2143, 829, 2096, 1270, 1001}},
    {{3022, 2135, 2146, 828, 2092, 1132, 999}},
    {{3142, 1870, 2133, 854, 2089, 1422, 1156}},
    {{3193, 1869, 2132, 890, 2095, 1411, 1188}},
    {{3212, 1875, 2050, 864, 2099, 1423, 1211}},
    {{3263, 1874, 2045, 865, 2099, 1324, 1216}},
    {{3263, 1875, 1989, 855, 2094, 1251, 1210}},
    {{3261, 1874, 1989, 845, 2091, 1206, 1181}},
    {{3260, 1873, 2047, 843, 2084, 1154, 1164}},
    {{3215, 1873, 2111, 844, 2085, 1196, 1166}},
    {{3202, 1871, 2201, 846, 2086, 1196, 1165}},
    {{3151, 1874, 2216, 850, 2089, 1320, 1202}},
    {{3139, 1875, 2212, 851, 2094, 1476, 1229}},
    {{3141, 1875, 2124, 849, 2094, 1506, 1210}},
    {{3142, 1876, 2050, 849, 2100, 1467, 1143}},
    {{3199, 1875, 2032, 850, 2098, 1413, 1138}},
    {{3201, 1874, 2022, 848, 2095, 1343, 1137}},
    {{3196, 1872, 2097, 845, 2083, 1352, 1152}},
    {{3150, 1872, 2114, 846, 2084, 1370, 1156}},
    {{3095, 1872, 2118, 851, 2081, 1353, 1154}},
    {{3078, 1873, 2160, 846, 2085, 1312, 1153}},
    {{3079, 1875, 2142, 851, 2099, 1241, 1146}},
    {{3134, 1872, 2047, 850, 2097, 1241, 1147}},
    {{3144, 1874, 1983, 850, 2099, 1272, 1180}},
    {{3153, 1875, 1967, 849, 2098, 1388, 1164}},
    {{3168, 1874, 1967, 848, 2094, 1441, 1165}},
    {{3168, 1871, 1969, 846, 2091, 1524, 1164}},
    {{3165, 1868, 1988, 843, 2079, 1557, 1163}},
    {{3157, 1870, 2074, 846, 2083, 1518, 1165}},
    {{3113, 1868, 2095, 845, 2082, 1477, 1164}},
    {{3090, 1868, 2117, 845, 2082, 1444, 1165}},
    {{3084, 1870, 2141, 847, 2083, 1416, 1165}},
    {{3084, 1869, 2142, 845, 2083, 1357, 1197}},
    {{3085, 1869, 2142, 846, 2086, 1318, 1204}},
    {{3087, 1874, 2139, 849, 2095, 1301, 1206}},
    {{3132, 1874, 2107, 850, 2099, 1308, 1207}},
    {{3132, 1875, 2073, 849, 2099, 1356, 1188}},
    {{3132, 1871, 2073, 850, 2099, 1395, 1165}},
    {{3121, 1867, 2074, 851, 2086, 1408, 1163}},
    {{3058, 1869, 2075, 850, 2085, 1406, 1126}},
    {{3051, 1873, 2075, 845, 2084, 1383, 1058}},
    {{3054, 1876, 2076, 845, 2088, 1314, 1032}},
    {{3062, 1878, 2073, 848, 2096, 1281, 1049}},
    {{3119, 1875, 2073, 848, 2096, 1281, 1085}},
    {{3143, 1875, 2073, 850, 2099, 1291, 1119}},
    {{3143, 1872, 2072, 850, 2100, 1344, 1147}},
    {{3142, 1871, 2072, 850, 2099, 1415, 1148}},
    {{3142, 1869, 2072, 850, 2098, 1445, 1148}},
    {{3053, 1868, 2075, 846, 2084, 1466, 1059}},
    {{2998, 1904, 2075, 828, 2083, 1412, 975}},
    {{2997, 1981, 2075, 828, 2084, 1339, 991}},
    {{2998, 1937, 2074, 829, 2092, 1396, 999}},
    {{3105, 1936, 2074, 829, 2094, 1386, 1055}},
    {{3150, 1912, 2073, 828, 2097, 1445, 1115}},
    {{3156, 1819, 2073, 875, 2095, 1454, 1115}},
    {{3081, 1797, 2075, 913, 2086, 1429, 1085}},
    {{3044, 1799, 2075, 900, 2084, 1399, 1048}},
    {{3019, 1817, 2075, 877, 2081, 1396, 1019}},
    {{3002, 1858, 2076, 828, 2088, 1395, 1018}},
    {{3095, 1897, 2075, 829, 2094, 1397, 1072}},
    {{3144, 1896, 2075, 828, 2096, 1430, 1113}},
    {{3158, 1836, 2073, 850, 2096, 1456, 1148}},
    {{3141, 1813, 2073, 906, 2084, 1412, 1144}},
    {{3065, 1820, 2074, 894, 2087, 1395, 1077}},
    {{3068, 1867, 2074, 850, 2096, 1388, 1062}},
    {{3166, 1894, 2073, 830, 2096, 1434, 1097}},
    {{3173, 1867, 2073, 867, 2095, 1457, 1125}},
    {{3174, 1835, 2008, 914, 2097, 1385, 1167}},
    {{3173, 1835, 1953, 895, 2093, 1317, 1146}},
    {{3172, 1833, 1954, 890, 2087, 1236, 1143}},
    {{3172, 1833, 1969, 862, 2080, 1223, 1143}},
    {{3160, 1833, 2080, 857, 2082, 1207, 1145}},
    {{3126, 1839, 2173, 849, 2084, 1259, 1143}},
    {{3121, 1838, 2238, 850, 2086, 1397, 1143}},
    {{3096, 1840, 2236, 851, 2089, 1448, 1147}},
    {{3096, 1839, 2213, 854, 2092, 1524, 1149}},
    {{3098, 1841, 2104, 849, 2094, 1570, 1146}},
    {{3098, 1842, 1977, 831, 2098, 1547, 1105}},
    {{3099, 1842, 1939, 830, 2095, 1455, 1081}},
    {{3098, 1842, 2094, 829, 2094, 1411, 1124}},
    {{3097, 2193, 2096, 829, 2092, 1133, 1123}},
    {{3097, 2193, 2096, 829, 2092, 1133, 1123}},
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
// resumeFromSequenceIndex: 0-based position in QUICK_TEST_POSE_INDICES to
// start from. 0 (the default, fresh-start case) truncates and overwrites
// QUICK_TEST_CSV as before. Any nonzero value means "resume" -- the CSV is
// opened in append mode instead, so previously-captured poses from an
// earlier run of this same pose sequence are preserved rather than wiped.
// Deliberately manual (caller/user supplies the pose number to resume from)
// rather than auto-detected from the CSV, since Ctrl+C gives no chance to
// run any auto-detection logic anyway -- the user has to know and report
// where they stopped regardless.
int runQuickCalibrationTest(std::size_t resumeFromSequenceIndex = 0) {
    const std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};

    DynamixelMotor motor(
        CYTON_DEVICE,
        CYTON_BAUD_RATE,
        CYTON_PROTOCOL_VERSION
    );

    // Compliance slope / punch experiment (2026-07-29) was tried here and
    // removed: it caused audible vibration on the real hardware (too
    // aggressive a change to both parameters at once -- see CLAUDE.md's
    // kinematic-calibration section) and was not worth the risk for a
    // modest precision gain. Kept out of this function going forward;
    // DynamixelMotor::readCompliance*/writeCompliance*/readPunch/
    // writePunch() still exist if this is revisited later, more
    // conservatively (one parameter at a time, smaller steps).

    try {
        const bool resuming = resumeFromSequenceIndex > 0;
        std::ofstream csv(
            QUICK_TEST_CSV,
            std::ios::out | (resuming ? std::ios::app : std::ios::trunc)
        );
        if (!csv) {
            throw std::runtime_error(
                std::string("Could not open CSV: ") + QUICK_TEST_CSV
            );
        }
        if (!resuming) {
            writeCsvHeader(csv);
        } else {
            std::cout
                << "Resuming quick-test: appending to " << QUICK_TEST_CSV
                << " starting at pose " << resumeFromSequenceIndex + 1
                << ".\n";
        }

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

        for (std::size_t i = resumeFromSequenceIndex; i < quickTestPoseCount; ++i) {
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

            // compensateBacklash disabled (2026-07-31) for the realm-
            // restricted calibration capture (TARGET_POSES 354-482) --
            // requested uncompensated for this specific run. Restore to
            // true for normal/production runs (this is what every other
            // dataset in this project has used, including the 49/51-pose
            // backlash-margin comparisons).
            if (!motor.moveJointsSafely(
                    motorIds,
                    targetTicks,
                    MOVING_SPEED,
                    MOTOR_TOLERANCE_TICKS,
                    MOVE_TIMEOUT_SECONDS,
                    true,
                    STALL_REPEATS_TO_DETECT,
                    false,
                    false,
                    {}
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

// --validate mode (2026-08-04): moves the arm to a caller-supplied list of
// joint-tick test points and reports, live, how far the NDI-measured actual
// position deviates from a precomputed *predicted* position for each point.
// Predictions are NOT computed here -- this program has no FK/calibration
// model of its own. They must be precomputed offline (see
// calibration/current/gen_validation_predictions.py, which fits the current
// best-fit correction model against a chosen dataset and evaluates it at
// each requested tick target) and supplied via VALIDATION_INPUT_CSV, format:
//   tick_0,tick_1,tick_2,tick_3,tick_4,tick_5,tick_6,predicted_x_mm,predicted_y_mm,predicted_z_mm
// For a meaningful test, the tick targets should be points NOT already in
// whatever dataset the model was fit on -- reusing fit poses here would
// just measure fit quality on training data again, not real generalization.
struct ValidationPoint {
    std::array<uint16_t, JOINT_COUNT> ticks{};
    double predictedXMm = 0.0;
    double predictedYMm = 0.0;
    double predictedZMm = 0.0;
};

std::vector<ValidationPoint> loadValidationPoints(const std::string& csvPath) {
    std::ifstream in(csvPath);
    if (!in) {
        throw std::runtime_error(
            "Could not open validation points CSV: " + csvPath +
            " -- run calibration/current/gen_validation_predictions.py "
            "first to generate it."
        );
    }

    std::string line;
    std::getline(in, line);  // header row, discarded

    std::vector<ValidationPoint> points;

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream fields(line);
        std::string field;

        ValidationPoint point;
        bool parsedOk = true;
        for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
            if (!std::getline(fields, field, ',')) {
                parsedOk = false;
                break;
            }
            point.ticks[i] = static_cast<uint16_t>(std::stoi(field));
        }
        if (parsedOk && std::getline(fields, field, ',')) {
            point.predictedXMm = std::stod(field);
        } else {
            parsedOk = false;
        }
        if (parsedOk && std::getline(fields, field, ',')) {
            point.predictedYMm = std::stod(field);
        } else {
            parsedOk = false;
        }
        if (parsedOk && std::getline(fields, field, ',')) {
            point.predictedZMm = std::stod(field);
        } else {
            parsedOk = false;
        }

        if (!parsedOk) {
            std::cout
                << "WARNING: skipping malformed row in " << csvPath
                << ": " << line << '\n';
            continue;
        }

        points.push_back(point);
    }

    return points;
}

constexpr const char* VALIDATION_RESULTS_CSV = "validation_results.csv";

// Reduced-DOF validation (2026-08-05): if set (>= 0), this joint is
// explicitly moved to LOCKED_VALIDATION_JOINT_TICK and locked (torqued)
// before any validation points run, rather than relying on every row of
// the predictions CSV happening to already carry a tick value near that
// target. Set LOCKED_VALIDATION_JOINT_ID to -1 to skip this and let all 7
// joints move normally for a non-DOF-reduced validation run.
constexpr int LOCKED_VALIDATION_JOINT_ID = 4;
constexpr uint16_t LOCKED_VALIDATION_JOINT_TICK = 2095;
// Deliberately looser than MOTOR_TOLERANCE_TICKS (5) -- that tighter value
// is what governs precision on the actual validation points and shouldn't
// be loosened, but this one locking move stalled in the servo's own
// compliance-margin deadband a few ticks short of target at 5-tick
// tolerance (observed: stuck at 7 ticks error, never closing further).
// A wider margin here is harmless: the data this was fit on saw the same
// lock settle anywhere in a ~25-tick window (2079-2104) and that was fine.
constexpr int LOCKED_VALIDATION_JOINT_TOLERANCE_TICKS = 15;

int runValidationTest(const std::string& inputCsvPath) {
    const std::vector<ValidationPoint> points = loadValidationPoints(inputCsvPath);
    if (points.empty()) {
        std::cerr << "No valid rows found in " << inputCsvPath << ".\n";
        return 1;
    }

    const std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};

    DynamixelMotor motor(
        CYTON_DEVICE,
        CYTON_BAUD_RATE,
        CYTON_PROTOCOL_VERSION
    );

    try {
        std::ofstream csv(VALIDATION_RESULTS_CSV, std::ios::out | std::ios::trunc);
        if (!csv) {
            throw std::runtime_error(
                std::string("Could not open CSV: ") + VALIDATION_RESULTS_CSV
            );
        }
        csv
            << "test_id,tick_0,tick_1,tick_2,tick_3,tick_4,tick_5,tick_6,"
            << "achieved_tick_0,achieved_tick_1,achieved_tick_2,"
            << "achieved_tick_3,achieved_tick_4,achieved_tick_5,"
            << "achieved_tick_6,max_tick_error,"
            << "predicted_x_mm,predicted_y_mm,predicted_z_mm,"
            << "actual_x_mm,actual_y_mm,actual_z_mm,deviation_mm\n";

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

        if (LOCKED_VALIDATION_JOINT_ID >= 0) {
            std::cout
                << "\nMoving motor " << LOCKED_VALIDATION_JOINT_ID
                << " to its locked validation position ("
                << LOCKED_VALIDATION_JOINT_TICK << ") before anything "
                << "else...\n";
            if (!motor.moveJointSafely(
                    LOCKED_VALIDATION_JOINT_ID,
                    LOCKED_VALIDATION_JOINT_TICK,
                    MOVING_SPEED,
                    LOCKED_VALIDATION_JOINT_TOLERANCE_TICKS,
                    MOVE_TIMEOUT_SECONDS,
                    true
                )) {
                throw std::runtime_error(
                    "Motor " + std::to_string(LOCKED_VALIDATION_JOINT_ID) +
                    " did not reach its locked validation target."
                );
            }
            if (!motor.enableTorque(LOCKED_VALIDATION_JOINT_ID)) {
                throw std::runtime_error(
                    "Failed to confirm torque is enabled on motor " +
                    std::to_string(LOCKED_VALIDATION_JOINT_ID) +
                    " to lock it."
                );
            }
            std::cout
                << "Motor " << LOCKED_VALIDATION_JOINT_ID << " locked at "
                << LOCKED_VALIDATION_JOINT_TICK << ".\n";
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
            << "\nCalibration validation test: " << points.size()
            << " test points.\n"
            << "Output: " << VALIDATION_RESULTS_CSV << "\n"
            << "Poses auto-advance. 'p' pauses/resumes, 's' skips the "
            << "current point, space toggles manual mode.\n";

        std::vector<double> deviations;

        for (std::size_t i = 0; i < points.size(); ++i) {
            const ValidationPoint& point = points[i];

            checkForModeToggle(manualModeEnabled);

            if (manualModeEnabled) {
                waitForEnter(
                    "\nPress Enter to move to test point " +
                    std::to_string(i + 1) + " of " +
                    std::to_string(points.size()) + "..."
                );
            } else {
                std::cout
                    << "\nAuto-advancing to test point " << i + 1
                    << " of " << points.size() << "...\n";
            }

            const std::vector<uint16_t> targetTicks(
                point.ticks.begin(), point.ticks.end()
            );

            if (!motor.moveJointsSafely(
                    motorIds,
                    targetTicks,
                    MOVING_SPEED,
                    MOTOR_TOLERANCE_TICKS,
                    MOVE_TIMEOUT_SECONDS,
                    true,
                    STALL_REPEATS_TO_DETECT,
                    true,
                    true,
                    {}
                )) {
                std::cout
                    << "\nWarning: test point " << i + 1
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

                const std::vector<uint16_t> achievedTicks =
                    readActualTicks(motor, motorIds);
                int maxTickError = 0;
                for (std::size_t j = 0; j < JOINT_COUNT; ++j) {
                    const int tickError = std::abs(
                        static_cast<int>(achievedTicks[j]) -
                        static_cast<int>(point.ticks[j])
                    );
                    if (tickError > maxTickError) {
                        maxTickError = tickError;
                    }
                }

                const DualToolCapture ndiCapture =
                    ndi.collectBothTools(manualModeEnabled);

                const double dx =
                    ndiCapture.movingRelativeToFixed.txMm - point.predictedXMm;
                const double dy =
                    ndiCapture.movingRelativeToFixed.tyMm - point.predictedYMm;
                const double dz =
                    ndiCapture.movingRelativeToFixed.tzMm - point.predictedZMm;
                const double deviationMm = std::sqrt(dx * dx + dy * dy + dz * dz);
                deviations.push_back(deviationMm);

                csv
                    << i + 1;
                for (uint16_t tick : point.ticks) {
                    csv << ',' << tick;
                }
                for (uint16_t tick : achievedTicks) {
                    csv << ',' << tick;
                }
                csv
                    << ',' << maxTickError
                    << ',' << std::setprecision(9) << point.predictedXMm
                    << ',' << point.predictedYMm
                    << ',' << point.predictedZMm
                    << ',' << ndiCapture.movingRelativeToFixed.txMm
                    << ',' << ndiCapture.movingRelativeToFixed.tyMm
                    << ',' << ndiCapture.movingRelativeToFixed.tzMm
                    << ',' << deviationMm
                    << '\n';
                csv.flush();

                std::cout
                    << "Test point " << i + 1 << ": predicted=("
                    << point.predictedXMm << ", " << point.predictedYMm
                    << ", " << point.predictedZMm << ") actual=("
                    << ndiCapture.movingRelativeToFixed.txMm << ", "
                    << ndiCapture.movingRelativeToFixed.tyMm << ", "
                    << ndiCapture.movingRelativeToFixed.tzMm
                    << ") deviation=" << deviationMm << "mm (max tick "
                    << "error vs. target: " << maxTickError << ")\n";
            } catch (const PoseSkippedByUser&) {
                std::cout
                    << "\nTest point " << i + 1
                    << " skipped by user. No deviation recorded for this "
                    << "point.\n";
            }
        }

        if (!deviations.empty()) {
            const double sum = std::accumulate(
                deviations.begin(), deviations.end(), 0.0
            );
            const double mean = sum / static_cast<double>(deviations.size());
            const double maxDeviation =
                *std::max_element(deviations.begin(), deviations.end());
            const double sumSq = std::inner_product(
                deviations.begin(), deviations.end(),
                deviations.begin(), 0.0
            );
            const double rms = std::sqrt(
                sumSq / static_cast<double>(deviations.size())
            );

            std::cout
                << "\nValidation complete. " << deviations.size() << " of "
                << points.size() << " test points measured.\n"
                << "Mean deviation: " << mean << "mm\n"
                << "RMS deviation:  " << rms << "mm\n"
                << "Max deviation:  " << maxDeviation << "mm\n"
                << "Saved data to " << VALIDATION_RESULTS_CSV << '\n';
        }

        disableAll(motor, motorIds);
        motor.disconnect();
        return 0;
    } catch (const PoseSkippedByUser&) {
        std::cout
            << "\nSkipped during initial NDI setup (before any point was "
            << "measured). Exiting.\n";
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
#ifndef _WIN32
    // Puts stdin in raw/non-blocking mode for the whole run so the ported
    // _kbhit()/_getch() shims above work; restored automatically on any
    // return path via the destructor.
    static LinuxRawStdin rawStdinGuard;
#endif
    if (argc > 1 && std::string(argv[1]) == "--settling-diagnostic") {
        return runSettlingDiagnostic();
    }
    if (argc > 1 && std::string(argv[1]) == "--quick-test") {
        // Optional third arg: 1-based pose number to resume from, matching
        // the "pose N of TOTAL" numbering already printed to the console --
        // e.g. `--quick-test 151` resumes right after "pose 150" was last
        // seen completed. Omit to start fresh (truncates the CSV as before).
        std::size_t resumeFromSequenceIndex = 0;
        if (argc > 2) {
            const long resumePoseNumber = std::stol(argv[2]);
            if (resumePoseNumber < 1) {
                std::cerr
                    << "Resume pose number must be 1 or greater.\n";
                return 1;
            }
            resumeFromSequenceIndex =
                static_cast<std::size_t>(resumePoseNumber - 1);
        }
        return runQuickCalibrationTest(resumeFromSequenceIndex);
    }
    if (argc > 1 && std::string(argv[1]) == "--validate") {
        // Optional second arg: path to the predictions CSV generated by
        // calibration/current/gen_validation_predictions.py. Defaults to
        // validation_points.csv in the current directory.
        const std::string inputCsvPath =
            (argc > 2) ? std::string(argv[2]) : "validation_points.csv";
        return runValidationTest(inputCsvPath);
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
