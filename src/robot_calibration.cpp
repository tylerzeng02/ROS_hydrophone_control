#include "robot_calibration.h"

#include <cmath>

// Offset/scale corrections come from the 60-param kinematic calibration fit
// (blocked-CV RMS 0.78mm; see CLAUDE.md). Offset is folded into zeroTick;
// scale is the fitted gear-ratio correction, applied in radiansToTicks()/
// ticksToRadians() below.
//
// Joints 0 (shoulder_roll) and 6 (wrist_roll) don't get their fitted offset:
// each is first/last in the chain with nothing rotational between it and
// the base/tool frame, so the offset is entangled with that (discarded)
// frame rotation -- applying it alone would introduce error, not fix it.
// Scale IS applied to both (grows with angle, not a constant bias, so it
// doesn't share the degeneracy). Joint 4 gets neither -- see its comment.
std::vector<JointCalibration> jointCalibrations = {
    // id, zeroTick, direction, minTick, maxTick, scale
    {0, 2048, +1, 276, 3772, 0.988203},                  // offset skipped (base-frame degeneracy)
    {1, 2047, +1, 851, 3231, 1.001931},                  // offset +0.0900deg
    {2, 2060, +1, 912, 3327, 0.964711},                  // offset +0.5238deg
    // minTick/maxTick have a few ticks' margin beyond the hand-verified
    // extremes: MoveIt's CheckStartStateBounds has zero tolerance for being
    // outside the declared range, and ordinary servo settling/backlash
    // noise is enough to occasionally land just past it -- see CLAUDE.md.
    {3, 2102, +1, 799, 3307, 1.014467},                  // offset +0.5297deg
    // Joint 4 (elbow_yaw) is deliberately LOCKED near its midpoint -- it's
    // the single worst-measured backlash joint (7.68mm) and a confirmed
    // coupling/gravity-deflection hotspot; excluding it from motion
    // planning measurably improved the other 6 joints' accuracy (see
    // CLAUDE.md, "reduced-DOF breakthrough"). No offset/scale applied: it
    // barely moved in the fit dataset (locked the whole time), so its own
    // correction is poorly identified and moot anyway while locked. Range
    // is a deliberately tiny ~4-degree window, with just enough slack to
    // absorb settling noise without tripping MoveIt's bounds check.
    {4, 2078, +1, 2060, 2130, 1.0},
    {5, 2042, +1, 751, 3344, 1.006796},                  // offset +0.4909deg
    {6, 2048, +1, 335, 3761, 1.002933}                   // offset skipped (tool-frame degeneracy)
};

int radiansToTicks(const JointCalibration& joint, double radians)
{
    constexpr double PI = 3.14159265358979323846;
    constexpr double TICKS_PER_RADIAN = 4096.0 / (2.0 * PI);

    return static_cast<int>(
        std::round(
            joint.zeroTick
            + joint.direction * radians * TICKS_PER_RADIAN / joint.scale
        )
    );
}

double ticksToRadians(const JointCalibration& joint, int tick)
{
    constexpr double PI = 3.14159265358979323846;
    constexpr double TICKS_PER_RADIAN = 4096.0 / (2.0 * PI);

    return joint.direction * joint.scale * (tick - joint.zeroTick) / TICKS_PER_RADIAN;
}