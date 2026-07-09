#include "robot_calibration.h"

#include <cmath>
#include <algorithm>

std::vector<JointCalibration> jointCalibrations = {
    // id, zeroTick, direction, minTick, maxTick
    {0, 2015, +1,    0, 4095},
    {1,  857, +1,  855, 3245},
    {2,  935, +1,  855, 3245},
    {3, 3239, +1,  855, 3245},
    {4, 1200, +1, 1034, 3245},
    {5, 3087, +1,  855, 3245},
    {6, 1967, +1,    0, 4095},
    {7, 2350, +1, 1578, 3172}
};

int radiansToTicks(const JointCalibration& joint, double radians)
{
    constexpr double PI = 3.14159265358979323846;
    constexpr double TICKS_PER_RADIAN = 4096.0 / (2.0 * PI);

    int targetTick = static_cast<int>(
        std::round(joint.zeroTick + joint.direction * radians * TICKS_PER_RADIAN)
    );

    return std::clamp(targetTick, joint.minTick, joint.maxTick);
}

double ticksToRadians(const JointCalibration& joint, int tick)
{
    constexpr double PI = 3.14159265358979323846;
    constexpr double TICKS_PER_RADIAN = 4096.0 / (2.0 * PI);

    return joint.direction * (tick - joint.zeroTick) / TICKS_PER_RADIAN;
}