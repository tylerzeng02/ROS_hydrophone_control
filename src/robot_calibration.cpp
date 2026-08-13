#include "robot_calibration.h"

#include <cmath>

// Offset and scale corrections below (2026-08-06) come from the 60-param
// kinematic calibration fit on the 374-pose elbow-yaw-locked dataset (see
// CLAUDE.md's kinematic-calibration section) -- the best validated model in
// the project (blocked-CV RMS 0.78mm; physically confirmed ~2.86mm mean /
// 3.35mm RMS on genuinely new points). Offset is folded directly into
// zeroTick (zeroTick_new = zeroTick_nominal - offset_rad*TICKS_PER_RADIAN/scale);
// scale is the fitted gear-ratio correction, applied in radiansToTicks()/
// ticksToRadians() below.
//
// (2026-08-13: scale was briefly reset to 1.0 for every joint, as one step
// in an "isolate each correction's contribution" comparison series -- now
// REVERTED back to these fitted values, since this is the production/
// "optimal" configuration again, not that comparison.)
//
// Joints 0 (shoulder_roll) and 6 (wrist_roll) deliberately do NOT get their
// fitted offset applied: each is the first/last joint in the chain with
// nothing rotational between it and the base/tool frame respectively, so
// its offset is provably entangled with that frame's (discarded, not used
// at runtime) rotation -- applying it alone would silently introduce a new
// error rather than fix one. Their fitted scale IS applied (scale's effect
// grows with commanded angle rather than being a constant bias, so it does
// not share this degeneracy -- same reasoning already validated for axis
// tilt in this project, not yet independently re-verified for scale
// specifically). Joint 4 (elbow_yaw) gets neither correction -- see its own
// comment below for why.
std::vector<JointCalibration> jointCalibrations = {
    // id, zeroTick, direction, minTick, maxTick, scale
    {0, 2048, +1, 276, 3772, 0.988203},                  // offset skipped (base-frame degeneracy)
    {1, 2047, +1, 851, 3231, 1.001931},                  // offset +0.0900deg
    {2, 2060, +1, 912, 3327, 0.964711},                  // offset +0.5238deg
    // minTick/maxTick widened 30 ticks each side (2026-08-07, was
    // [829, 3277]) after repeated real-hardware MoveIt sessions showed the
    // arm's actual settled position occasionally landing a few ticks to
    // ~25 ticks past this joint's exact hand-verified extreme -- ordinary
    // servo settling/backlash noise, not a real safety concern (this
    // margin is well inside MOTOR_TOLERANCE_TICKS-scale noise, not a
    // meaningful extension into unverified territory), but MoveIt's
    // CheckStartStateBounds planning adapter has zero tolerance for it and
    // refuses to plan at all once the arm is even slightly outside the
    // declared range -- and since nothing then corrects that state, every
    // subsequent planning request fails the same way until a human (or
    // pose_commander's own recovery logic, see cyton_pose_commander)
    // manually nudges the joint back inside bounds. See CLAUDE.md's
    // kinematic-calibration section for the specific incidents.
    {3, 2102, +1, 799, 3307, 1.014467},                  // offset +0.5297deg
    // Joint 4 (elbow_yaw) RE-LOCKED (2026-08-13): was briefly widened to
    // [944, 3245] for a 7-DOF "original/raw" uncalibrated-URDF comparison
    // test, now REVERTED back to its normal production lock, since this is
    // the "optimal" configuration again, not that comparison. Locked near
    // its midpoint because it's the single worst-measured backlash joint
    // (7.68mm) and a confirmed joint-coupling/gravity-deflection hotspot --
    // locking it out of real motion planning was found to meaningfully
    // improve the other 6 joints' calibration accuracy (see CLAUDE.md's
    // kinematic-calibration section, "reduced-DOF breakthrough"). No offset/
    // scale applied here either: this joint barely moved in the dataset it
    // was fit on (locked the whole time), so its own correction is poorly
    // identified and not trustworthy -- moot anyway since it's locked and
    // never evaluated away from this narrow range. Still a deliberately
    // tiny ~70-tick (~4 degree) window, nowhere close to this joint's real,
    // wider mechanical range -- it stays "locked" in every practical sense,
    // just with enough slack (widened 15 ticks each side, 2026-08-07, from
    // [2075, 2115]) to absorb real settling noise instead of tripping
    // MoveIt's bounds check on it.
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