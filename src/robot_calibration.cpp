#include "robot_calibration.h"

#include <cmath>

std::vector<JointCalibration> jointCalibrations = {
    // id, zeroTick, direction, minTick, maxTick
    {0, 2048, +1, 376, 3772},
    {1, 2048, +1, 853, 3231},
    {2, 2066, +1, 912, 3327},
    {3, 2108, +1, 829, 3274},
    {4, 2078, +1, 952, 3245},
    {5, 2048, +1, 751, 3344},
    {6, 2048, +1, 335, 3755}
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