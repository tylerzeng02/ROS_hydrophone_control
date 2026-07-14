#include "robot_calibration.h"

#include <cmath>

std::vector<JointCalibration> jointCalibrations = {
    // id, zeroTick, direction, minTick, maxTick
    {0, 2048, +1,  341, 3755},
    {1, 2048, +1,  853, 3243},
    {2, 2116, +1,  912, 3320},
    {3, 2048, +1,  853, 3243},
    {4, 2048, +1,  853, 3243},
    {5, 2048, -1,  853, 3243},
    {6, 2048, +1,  341, 3755},
    {7, 2350, +1, 1578, 3172}
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