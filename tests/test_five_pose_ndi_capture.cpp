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
constexpr std::size_t POSE_COUNT = 190;

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
// collectBothTools()) as the normal 190-pose run, but in one short sitting
// -- to test whether collecting the dataset without the multi-hour
// session-length drift found in the original (superseded) 200-pose data
// gets the fitted calibration error down near the arm's ~0.5mm repeatability
// floor. Writes to its own CSV so it never touches five_pose_ndi_capture.csv
// or its resume state.
//
// QUICK_TEST_POSE_INDICES covers all 190 poses in this hand-recorded set:
// independently verified to cover ~96-103% of every joint's safe range
// (see calibration/diag_check_new190.py), so no subset curation is needed
// here.
constexpr std::array<int, 190> QUICK_TEST_POSE_INDICES = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189
};
constexpr const char* QUICK_TEST_CSV = "quick_calibration_test.csv";

// Hand-recorded via tests/record_hand_poses.cpp: the arm was physically
// moved by hand (torque off) into each pose and the resulting joint ticks
// captured, so every pose here is a real, human-verified configuration --
// not randomly sampled -- meaning it's known in advance to be physically
// reachable, collision-free, and (assuming it was checked live against the
// "Moving tool: DETECTED/MISSING" status while posing) marker-visible.
const std::array<std::array<uint16_t, JOINT_COUNT>, POSE_COUNT> TARGET_POSES = {{
    {{3206, 3228, 999, 1061, 1857, 2013, 3755}},
    {{3150, 3230, 1273, 1060, 1858, 1920, 336}},
    {{3093, 3230, 1395, 1061, 2041, 1984, 335}},
    {{2975, 3230, 1464, 1062, 1926, 2019, 432}},
    {{2834, 3231, 1464, 1063, 1948, 2014, 633}},
    {{2620, 3229, 1464, 936, 1902, 2033, 853}},
    {{2413, 3229, 1273, 885, 1886, 2095, 1027}},
    {{2351, 3229, 1079, 829, 1875, 2220, 1015}},
    {{2034, 3227, 1078, 832, 1925, 2155, 1017}},
    {{1652, 3228, 1134, 830, 1868, 2150, 1505}},
    {{1319, 3228, 1006, 830, 1872, 2239, 1705}},
    {{870, 3227, 1007, 831, 1939, 2270, 2033}},
    {{696, 3227, 1006, 830, 1937, 2295, 2345}},
    {{695, 3229, 2518, 830, 2199, 2172, 3617}},
    {{1222, 3230, 2508, 830, 2198, 2171, 3220}},
    {{1540, 3230, 2915, 901, 2250, 2143, 3421}},
    {{1911, 3230, 2913, 1104, 2231, 2011, 2997}},
    {{2267, 3230, 2873, 831, 2183, 2191, 2459}},
    {{2623, 3231, 2074, 833, 2035, 2033, 1510}},
    {{2701, 3231, 1530, 831, 1957, 2094, 898}},
    {{2840, 3230, 1314, 831, 1908, 2164, 499}},
    {{2919, 3230, 1085, 833, 1908, 2198, 337}},
    {{2901, 3045, 1110, 1014, 1014, 1925, 3754}},
    {{2632, 3230, 1190, 1063, 1175, 1516, 338}},
    {{2347, 3228, 1227, 1063, 1368, 1157, 338}},
    {{2343, 3035, 1226, 830, 1694, 1153, 340}},
    {{2129, 2574, 1398, 834, 2378, 1099, 1048}},
    {{2041, 2997, 1540, 830, 2367, 1036, 1048}},
    {{2008, 2845, 1952, 829, 2844, 1348, 1049}},
    {{1976, 3230, 2474, 829, 2569, 2122, 2083}},
    {{1961, 3230, 2913, 834, 3249, 2871, 2082}},
    {{1656, 3228, 2963, 830, 2247, 2203, 3118}},
    {{1710, 3227, 3225, 850, 2147, 2262, 3290}},
    {{1617, 3222, 3309, 1105, 2938, 1460, 3755}},
    {{1617, 3222, 3307, 1634, 3153, 1304, 335}},
    {{1430, 3222, 3307, 1947, 3249, 1562, 544}},
    {{761, 2036, 3318, 1510, 958, 2044, 3753}},
    {{561, 1856, 3314, 1339, 966, 2123, 335}},
    {{327, 1856, 3309, 1219, 973, 2027, 336}},
    {{334, 1686, 3312, 1252, 955, 2157, 335}},
    {{296, 1696, 2976, 1252, 957, 2390, 336}},
    {{299, 1354, 2977, 1625, 956, 2746, 336}},
    {{319, 1310, 2988, 2416, 1482, 2784, 3560}},
    {{304, 942, 2342, 3275, 2191, 1847, 3564}},
    {{300, 851, 2341, 3272, 1866, 2353, 3560}},
    {{287, 1896, 3287, 832, 1024, 1444, 983}},
    {{287, 1651, 3287, 832, 952, 1968, 983}},
    {{287, 852, 929, 3230, 1912, 2663, 1251}},
    {{2920, 1986, 3291, 1922, 952, 2522, 1353}},
    {{2919, 1983, 3289, 1921, 954, 2092, 1352}},
    {{2919, 1981, 3093, 1908, 981, 1981, 1354}},
    {{2795, 1811, 3096, 1908, 988, 1820, 1309}},
    {{2719, 1546, 3097, 1908, 965, 2122, 1305}},
    {{2604, 1415, 3097, 1912, 955, 2840, 1304}},
    {{2598, 1019, 3097, 1907, 961, 2944, 1517}},
    {{2759, 852, 3216, 2230, 3189, 2534, 3258}},
    {{2634, 2137, 922, 2315, 3188, 1949, 1649}},
    {{2774, 2040, 930, 2311, 3188, 2008, 1539}},
    {{2932, 1728, 932, 2314, 3189, 2424, 1317}},
    {{3646, 1897, 927, 2316, 3168, 2317, 537}},
    {{3755, 1797, 1305, 2337, 2825, 2265, 339}},
    {{3759, 1532, 1291, 1918, 3249, 2655, 3614}},
    {{3782, 2103, 927, 836, 3056, 1992, 2990}},
    {{3782, 2118, 927, 1253, 3013, 1944, 3380}},
    {{3746, 2114, 927, 1804, 3146, 1919, 3753}},
    {{1763, 1798, 927, 1246, 3246, 2050, 1358}},
    {{1667, 1830, 921, 2038, 3228, 2134, 2410}},
    {{1457, 1915, 924, 2730, 3169, 2246, 3262}},
    {{1085, 2034, 927, 2796, 3166, 2211, 3754}},
    {{626, 1990, 2093, 3275, 2512, 786, 3754}},
    {{577, 2051, 1997, 3097, 2344, 1035, 3550}},
    {{628, 2039, 1849, 2917, 2373, 1262, 3465}},
    {{696, 2021, 1727, 2580, 2607, 1544, 3467}},
    {{696, 1656, 1662, 2408, 2700, 2194, 3704}},
    {{813, 1448, 1798, 2408, 2525, 2369, 3443}},
    {{1047, 1251, 1797, 2410, 2493, 2600, 3354}},
    {{1261, 1202, 1797, 2905, 2247, 2033, 3157}},
    {{1682, 1075, 1797, 3111, 1976, 1933, 2802}},
    {{1458, 852, 1799, 3272, 2098, 2814, 3076}},
    {{1202, 851, 1799, 3257, 1623, 2879, 3404}},
    {{963, 851, 1799, 3269, 1363, 2653, 3725}},
    {{709, 852, 1264, 3272, 1136, 1971, 3754}},
    {{576, 852, 1028, 3272, 973, 1606, 3754}},
    {{273, 852, 931, 3147, 1729, 2938, 1336}},
    {{272, 852, 933, 2698, 1691, 2876, 1241}},
    {{273, 1381, 929, 2860, 2330, 2369, 1240}},
    {{273, 1922, 927, 2726, 2965, 2349, 337}},
    {{273, 1953, 2127, 2324, 1959, 1947, 3754}},
    {{273, 1929, 2316, 2321, 1895, 1935, 3754}},
    {{273, 1928, 2469, 2312, 1866, 1940, 3754}},
    {{273, 1931, 2659, 2315, 1870, 1867, 3754}},
    {{273, 1932, 2764, 2312, 1612, 2018, 3754}},
    {{273, 1931, 2950, 2313, 1408, 2069, 3754}},
    {{273, 1933, 3062, 2313, 1348, 2185, 3430}},
    {{322, 1952, 3159, 2384, 1267, 2212, 3655}},
    {{323, 1951, 3315, 2333, 1221, 2129, 3511}},
    {{283, 1669, 3306, 2241, 1133, 2389, 3635}},
    {{285, 1683, 3079, 2261, 1259, 2425, 3633}},
    {{285, 1729, 2836, 2265, 1518, 2339, 3632}},
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