#include "robot_calibration.h"

#include <cmath>

std::vector<JointCalibration> jointCalibrations = {
    // id, zeroTick, direction, minTick, maxTick
    {0, 2048, +1, 276, 3772},
    {1, 2048, +1, 851, 3231},
    {2, 2066, +1, 912, 3327},
    {3, 2108, +1, 829, 3277},
    {4, 2078, +1, 944, 3245},
    {5, 2048, +1, 751, 3344},
    {6, 2048, +1, 335, 3761}
};

int radiansToTicks(const JointCalibration& joint, double radians)
{
    constexpr double PI = 3.14159265358979323846;
    constexpr double TICKS_PER_RADIAN = 4096.0 / (2.0 * PI);

    return static_cast<int>(
        std::round(
            joint.zeroTick
            + joint.direction * radians * TICKS_PER_RADIAN
        )
    );
}

double ticksToRadians(const JointCalibration& joint, int tick)
{
    constexpr double PI = 3.14159265358979323846;
    constexpr double TICKS_PER_RADIAN = 4096.0 / (2.0 * PI);

    return joint.direction * (tick - joint.zeroTick) / TICKS_PER_RADIAN;
}