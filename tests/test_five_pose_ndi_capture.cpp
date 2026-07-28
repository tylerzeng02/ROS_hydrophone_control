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
constexpr std::size_t POSE_COUNT = 304;

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
// collectBothTools()) as the normal 291-pose run, but in one short sitting
// -- to test whether collecting the dataset without the multi-hour
// session-length drift found in the original (superseded) 200-pose data
// gets the fitted calibration error down near the arm's ~0.5mm repeatability
// floor. Writes to its own CSV so it never touches five_pose_ndi_capture.csv
// or its resume state.
//
// QUICK_TEST_POSE_INDICES currently targets ONLY the 13 new poses (indices
// 291-303) specifically recorded to test orientation diversity -- a small,
// fast, focused capture so it's quick to check both (a) whether these
// poses are genuinely more orientation-diverse in the real NDI-measured
// data too (not just the FK-predicted check already done), and (b) whether
// NDI can actually track the marker reliably at these new, more varied
// orientations. Change back to covering the full TARGET_POSES range for a
// complete recalibration run once these are validated.
constexpr std::array<int, 13> QUICK_TEST_POSE_INDICES = {
    291,292,293,294,295,296,297,298,299,300,301,302,303
};
constexpr const char* QUICK_TEST_CSV = "quick_calibration_test.csv";

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