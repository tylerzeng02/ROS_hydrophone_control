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
constexpr std::size_t POSE_COUNT = 200;

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
constexpr int NDI_SAMPLE_INTERVAL_MS = 20;
constexpr int REQUIRED_VISIBLE_MARKERS = 4;
constexpr double MAX_NDI_ERROR = 0.50;

// Hand-recorded via tests/record_hand_poses.cpp: the arm was physically
// moved by hand (torque off) into each pose and the resulting joint ticks
// captured, so every pose here is a real, human-verified configuration --
// not randomly sampled -- meaning it's known in advance to be physically
// reachable, collision-free, and (assuming it was checked live against the
// "Moving tool: DETECTED/MISSING" status while posing) marker-visible.
const std::array<std::array<uint16_t, JOINT_COUNT>, POSE_COUNT> TARGET_POSES = {{
    {{2079, 2031, 923, 2518, 3195, 2140, 2384}},
    {{2195, 1830, 922, 2489, 3200, 2262, 2248}},
    {{2424, 1900, 923, 2260, 3178, 2246, 1821}},
    {{2539, 1837, 928, 2003, 3181, 2285, 1458}},
    {{2800, 1722, 930, 1935, 3196, 2342, 1147}},
    {{3091, 1626, 928, 1882, 3156, 2483, 665}},
    {{3460, 1981, 916, 1894, 3158, 2032, 337}},
    {{3535, 944, 929, 2120, 2923, 3096, 544}},
    {{2725, 1947, 921, 2011, 3233, 3176, 999}},
    {{2720, 1731, 921, 1684, 3187, 2938, 1001}},
    {{2722, 1526, 925, 1686, 3195, 2498, 1001}},
    {{2803, 1526, 937, 1679, 3244, 2176, 999}},
    {{3030, 1337, 996, 1703, 2793, 2165, 383}},
    {{3060, 936, 999, 1707, 2938, 2195, 414}},
    {{3068, 3142, 1197, 1950, 2854, 2211, 951}},
    {{3223, 3172, 1190, 1892, 2854, 2487, 951}},
    {{3192, 3172, 1286, 1895, 2856, 2897, 952}},
    {{3133, 3228, 1523, 1913, 2771, 1191, 1037}},
    {{2895, 3225, 1527, 1910, 2770, 1179, 1038}},
    {{2667, 3111, 1529, 1528, 2126, 1632, 785}},
    {{2593, 3220, 1525, 1136, 2031, 1799, 787}},
    {{2539, 2994, 1525, 1014, 2033, 1716, 831}},
    {{2293, 3231, 1525, 1006, 2038, 1936, 1007}},
    {{2202, 3228, 1525, 830, 1596, 1330, 1006}},
    {{2200, 2792, 1233, 829, 1940, 1990, 1142}},
    {{2208, 2360, 1266, 1031, 2597, 2134, 1144}},
    {{2207, 2332, 1666, 1366, 2595, 1493, 1143}},
    {{2323, 2222, 2090, 988, 2690, 942, 1144}},
    {{2321, 2194, 2133, 829, 2381, 1366, 1146}},
    {{2351, 1945, 2249, 831, 2372, 1538, 1144}},
    {{2261, 1684, 2249, 830, 2047, 1857, 873}},
    {{2190, 1371, 2249, 833, 1746, 1864, 573}},
    {{1613, 852, 2211, 2503, 1358, 2128, 2834}},
    {{1593, 851, 2207, 1582, 1362, 2288, 3532}},
    {{1461, 911, 2105, 2144, 1039, 2826, 3527}},
    {{1176, 1077, 2081, 2345, 1487, 2930, 3525}},
    {{1158, 1356, 2080, 3245, 2021, 2569, 3303}},
    {{1160, 1801, 2080, 3273, 2022, 2759, 3306}},
    {{1135, 1508, 1853, 3267, 2129, 1877, 2979}},
    {{1107, 1679, 1851, 2898, 2034, 1795, 2976}},
    {{1107, 1560, 1850, 2794, 2032, 1589, 2976}},
    {{1058, 1106, 1712, 2989, 1959, 1988, 3238}},
    {{969, 944, 1713, 3273, 1818, 2352, 3240}},
    {{968, 852, 1713, 3272, 1442, 2557, 3563}},
    {{2233, 853, 991, 2539, 2168, 2868, 3028}},
    {{2354, 852, 991, 2911, 2169, 2917, 3028}},
    {{2383, 852, 991, 3265, 2179, 2986, 3027}},
    {{3769, 1886, 1363, 1423, 1024, 1698, 512}},
    {{3249, 1885, 1132, 1420, 1026, 2130, 337}},
    {{2487, 1809, 1197, 1062, 1301, 1998, 3383}},
    {{2236, 1809, 1197, 1062, 1303, 2114, 3378}},
    {{2151, 1809, 1197, 1061, 1277, 2270, 3377}},
    {{2056, 1810, 1198, 1061, 1211, 2366, 3376}},
    {{1981, 1811, 1198, 1061, 1186, 2466, 3376}},
    {{1946, 1809, 1198, 1061, 1171, 2757, 3376}},
    {{1881, 1813, 1199, 1061, 1137, 3122, 3382}},
    {{1612, 2218, 1140, 1215, 958, 2296, 2506}},
    {{1457, 2218, 1141, 1299, 955, 2357, 2419}},
    {{1311, 2218, 1139, 1365, 955, 2165, 2123}},
    {{620, 1961, 1155, 2602, 962, 2168, 339}},
    {{555, 1961, 1036, 2604, 1155, 2503, 340}},
    {{973, 1891, 2138, 2101, 2034, 2153, 3172}},
    {{975, 1743, 2139, 2105, 2035, 2092, 3173}},
    {{975, 1631, 2169, 2102, 2035, 1844, 3173}},
    {{973, 1516, 2188, 2109, 2034, 1738, 3149}},
    {{975, 1359, 2188, 2126, 2035, 1746, 3149}},
    {{975, 1621, 2189, 2736, 2034, 1971, 3118}},
    {{975, 1766, 2195, 2749, 2035, 2011, 3069}},
    {{971, 1935, 2217, 2767, 2037, 1988, 3069}},
    {{966, 2154, 2286, 2777, 2035, 1969, 3069}},
    {{908, 2381, 2286, 2720, 2032, 1909, 3072}},
    {{707, 2702, 2287, 2614, 2034, 1574, 3072}},
    {{687, 2952, 2270, 2391, 2006, 1262, 3072}},
    {{691, 3222, 2883, 1795, 1978, 2948, 2082}},
    {{889, 3222, 2879, 1796, 1977, 2980, 2083}},
    {{1013, 3222, 2881, 1752, 2006, 3127, 2567}},
    {{961, 852, 3307, 3187, 2128, 2025, 2241}},
    {{1175, 852, 3306, 2719, 2838, 2181, 1557}},
    {{1175, 851, 3305, 2760, 3035, 1965, 1558}},
    {{1234, 851, 2144, 3219, 2457, 2780, 2570}},
    {{1295, 1715, 2196, 3274, 2223, 2680, 2824}},
    {{1296, 2151, 1367, 3272, 2165, 2187, 3531}},
    {{984, 2274, 1322, 3273, 2244, 1968, 3664}},
    {{487, 2425, 1273, 3273, 2245, 1647, 3750}},
    {{414, 3223, 2506, 1853, 2037, 2629, 1949}},
    {{580, 3223, 2506, 1854, 2036, 2705, 1949}},
    {{758, 3223, 2505, 1851, 2036, 2787, 2268}},
    {{973, 3220, 2584, 1671, 2355, 2975, 2339}},
    {{1097, 3229, 2576, 1215, 2081, 1579, 3273}},
    {{1524, 3229, 2577, 830, 2141, 1993, 3016}},
    {{1601, 3225, 2577, 829, 2126, 1533, 3017}},
    {{1601, 3225, 2577, 829, 2116, 1493, 3017}},
    {{1723, 2512, 2578, 829, 1566, 2001, 3015}},
    {{1724, 2705, 2578, 829, 1588, 1941, 2835}},
    {{1725, 2712, 2577, 829, 1589, 1517, 2825}},
    {{1722, 2902, 2580, 829, 2240, 2653, 2679}},
    {{1431, 3224, 2580, 829, 2533, 2470, 2684}},
    {{1140, 3224, 2580, 829, 2607, 2529, 2788}},
    {{1069, 3225, 2580, 829, 2616, 1569, 3498}},
    {{758, 3226, 2579, 830, 2861, 1756, 3754}},
    {{2337, 1408, 932, 1667, 1963, 2089, 2895}},
    {{2340, 1377, 931, 1695, 2027, 1751, 2895}},
    {{2455, 1205, 933, 1686, 2029, 1621, 3027}},
    {{2693, 852, 1575, 2260, 1572, 1260, 3753}},
    {{2712, 877, 1589, 2581, 1574, 1110, 3753}},
    {{2738, 869, 1588, 2863, 1571, 1024, 3753}},
    {{2912, 851, 2010, 2997, 962, 945, 3754}},
    {{3067, 851, 2010, 3255, 2050, 2298, 796}},
    {{3077, 1229, 2009, 3163, 2049, 2002, 796}},
    {{3078, 1042, 2009, 3259, 2190, 2552, 798}},
    {{3477, 852, 2009, 3269, 2685, 2652, 351}},
    {{3657, 854, 2011, 3037, 2775, 2682, 3753}},
    {{3668, 853, 2016, 3267, 2514, 2437, 336}},
    {{3763, 852, 2017, 3274, 2615, 2276, 336}},
    {{3763, 853, 2017, 3273, 2307, 2113, 336}},
    {{3760, 853, 2015, 3272, 2042, 2024, 336}},
    {{3758, 852, 2015, 3272, 1788, 1974, 335}},
    {{3758, 852, 2015, 3272, 1567, 1904, 335}},
    {{3747, 852, 3170, 3189, 2265, 1788, 3289}},
    {{3654, 852, 3170, 3262, 2182, 1945, 3293}},
    {{3650, 852, 3170, 3274, 1986, 2657, 3294}},
    {{3646, 852, 3167, 3240, 1549, 2773, 3689}},
    {{2742, 852, 3168, 3274, 1038, 2463, 3695}},
    {{2568, 852, 3168, 3274, 1026, 2318, 3694}},
    {{2329, 852, 3169, 3274, 960, 2053, 3691}},
    {{2323, 852, 3169, 3273, 958, 1711, 3694}},
    {{2342, 852, 3170, 3260, 958, 1451, 3693}},
    {{2340, 852, 3168, 3273, 2340, 2055, 841}},
    {{2340, 853, 3168, 3269, 2678, 2419, 640}},
    {{2577, 852, 3168, 3270, 2906, 2586, 443}},
    {{2865, 852, 3168, 3260, 3176, 2186, 335}},
    {{3593, 852, 3287, 3238, 3148, 1558, 335}},
    {{3588, 852, 3287, 3261, 3196, 1629, 335}},
    {{3770, 853, 3288, 2609, 3175, 1327, 3753}},
    {{3769, 853, 3288, 2603, 3046, 1475, 3753}},
    {{2938, 852, 3287, 3021, 2229, 2294, 3752}},
    {{2937, 852, 3287, 3056, 2056, 2272, 3753}},
    {{2931, 852, 3287, 3173, 1839, 2220, 3752}},
    {{2929, 852, 3287, 3271, 1681, 2277, 3751}},
    {{2855, 852, 3287, 3272, 1439, 2330, 3752}},
    {{2166, 852, 3286, 3273, 1199, 1719, 3753}},
    {{1889, 852, 3287, 3274, 1166, 1433, 3753}},
    {{1669, 852, 3234, 2889, 1027, 756, 3752}},
    {{1535, 852, 3232, 2368, 1026, 1320, 3484}},
    {{1418, 852, 3222, 2230, 1027, 1480, 3483}},
    {{1185, 852, 2876, 2012, 1634, 1511, 3546}},
    {{1180, 852, 2883, 2003, 1639, 2470, 3073}},
    {{1206, 907, 2876, 2408, 1703, 3049, 3072}},
    {{1377, 852, 2657, 1798, 1710, 2560, 3076}},
    {{1186, 853, 2656, 1804, 1710, 3099, 3075}},
    {{864, 852, 2658, 2225, 1711, 3187, 3075}},
    {{866, 852, 2659, 2027, 1712, 3314, 3077}},
    {{701, 905, 2657, 2935, 1830, 2586, 3073}},
    {{599, 852, 2657, 3114, 2028, 2488, 3076}},
    {{598, 853, 2657, 3272, 2201, 2651, 3076}},
    {{600, 853, 2658, 3273, 2546, 2728, 2767}},
    {{1067, 853, 2657, 3259, 2834, 2340, 2376}},
    {{1424, 853, 2658, 2833, 2924, 2060, 1759}},
    {{1429, 852, 2657, 2343, 2924, 1859, 1152}},
    {{1428, 852, 2657, 2052, 2727, 1778, 762}},
    {{1571, 852, 2661, 1890, 2818, 1973, 627}},
    {{1741, 852, 2715, 1887, 2927, 2165, 626}},
    {{2119, 852, 2581, 1991, 3138, 2816, 337}},
    {{2215, 853, 2762, 2420, 3139, 2790, 336}},
    {{2214, 853, 2951, 2841, 2954, 2766, 336}},
    {{2277, 854, 3292, 3071, 2855, 2644, 337}},
    {{2334, 855, 3292, 2945, 2726, 2360, 336}},
    {{2607, 852, 3290, 2717, 2581, 2437, 3754}},
    {{2867, 856, 3291, 2842, 2477, 2211, 3751}},
    {{2868, 854, 3289, 2855, 2315, 2298, 3751}},
    {{2961, 2039, 2353, 2115, 1695, 2152, 859}},
    {{2961, 2132, 2353, 2150, 1696, 2018, 862}},
    {{2958, 2251, 2366, 2205, 1696, 1800, 935}},
    {{2980, 2364, 2368, 2208, 1696, 1762, 977}},
    {{2995, 2493, 2366, 2217, 1693, 1596, 979}},
    {{3000, 2615, 2368, 2216, 1695, 1844, 1034}},
    {{3012, 2628, 2366, 2217, 1693, 2028, 1034}},
    {{3031, 2620, 2365, 2214, 1642, 2204, 1035}},
    {{3046, 2484, 2367, 2063, 1633, 1385, 1033}},
    {{3046, 2409, 2368, 1866, 1634, 1466, 1034}},
    {{3046, 2287, 2368, 1709, 1635, 1622, 1035}},
    {{3045, 2176, 2368, 1618, 1640, 1734, 1036}},
    {{3046, 2078, 2369, 1488, 1640, 2009, 1097}},
    {{3051, 1973, 2378, 1222, 1649, 2403, 1316}},
    {{3128, 1935, 2496, 1224, 1639, 2396, 1315}},
    {{3213, 1943, 2534, 1424, 1576, 2157, 1334}},
    {{3232, 1950, 2724, 1463, 1450, 2102, 1338}},
    {{3257, 1956, 2923, 1472, 1305, 2116, 1338}},
    {{3333, 2077, 3046, 1481, 1048, 2506, 1334}},
    {{3410, 2255, 3293, 1561, 958, 2573, 1222}},
    {{3611, 2410, 3294, 1681, 960, 2593, 1226}},
    {{3735, 2632, 3293, 1694, 971, 2710, 1423}},
    {{3610, 2537, 2358, 1369, 1765, 1896, 880}},
    {{3582, 2784, 2424, 1149, 1778, 1606, 883}},
    {{3163, 1969, 2198, 1880, 1822, 2147, 881}},
    {{3121, 1904, 1857, 1729, 1694, 1976, 881}},
    {{3081, 1696, 1675, 1580, 1693, 2096, 734}},
    {{2646, 1351, 1486, 1578, 2766, 2068, 730}},
    {{2615, 1183, 1471, 1580, 2781, 1907, 543}},
    {{2500, 1110, 1461, 1651, 2789, 1632, 347}},
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
            std::cout
                << "\nWARNING: motor " << id << " ended up at position "
                << position << ", which is outside its calibrated safe "
                << "range. This pose cannot be safely captured.\n"
                << "Press 's' to skip this pose (torque stays enabled, "
                << "the arm keeps holding its current position).\n";

            while (true) {
                if (_kbhit()) {
                    const int key = _getch();
                    if (key == 's' || key == 'S') {
                        throw PoseSkippedByUser{};
                    }
                }
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(100)
                );
            }
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
        ndi.initialize(manualModeEnabled);

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