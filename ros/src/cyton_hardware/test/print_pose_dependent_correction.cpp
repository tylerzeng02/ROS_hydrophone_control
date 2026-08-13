// Manual cross-check for src/pose_dependent_correction.cpp -- NOT wired
// into this package's CMakeLists.txt/colcon build (deliberately, to avoid
// adding a permanent extra build target for a one-off verification). Prints
// the correction for the same 5 test joint configurations as
// calibration/current/verify_pose_dependent_port.py; per-joint
// correction_rad values should match that script's output to ~1e-6 or
// better (line formats differ slightly, so compare the numbers, not a raw
// line-for-line diff) -- confirmed matching to 8 decimal places on every
// value, every config, when this port was first verified (2026-08-13).
//
// To build/run standalone:
//   g++ -std=c++17 -I../../../../src print_pose_dependent_correction.cpp \
//       ../../../../src/pose_dependent_correction.cpp -o print_correction
//   ./print_correction

#include <array>
#include <cstdio>
#include "pose_dependent_correction.h"

int main() {
    const char* names[7] = {"shoulder_roll","shoulder_pitch","shoulder_yaw","elbow_pitch","elbow_yaw","wrist_pitch","wrist_roll"};
    std::array<std::array<double,7>,5> configs = {{
        {0,0,0,0,0,0,0},
        {0.3,-0.2,0.15,-0.5,0.02,0.4,-0.1},
        {-0.8,0.6,-1.0,1.2,-0.02,-0.9,0.7},
        {1.5,-1.0,1.5,-1.8,0.0,1.7,-1.5},
        {0.05,0.05,-0.05,0.1,0.01,-0.05,0.05},
    }};
    for (size_t idx = 0; idx < configs.size(); ++idx) {
        auto c = pose_dependent_correction::computeCorrection(configs[idx]);
        std::printf("config %zu:\n", idx);
        for (int i = 0; i < 7; ++i) {
            std::printf("    %-15s correction_rad=% .8f\n", names[i], c[i]);
        }
    }
    return 0;
}
