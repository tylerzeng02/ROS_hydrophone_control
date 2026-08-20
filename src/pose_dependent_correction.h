#pragma once

#include <array>

// Live, pose-dependent kinematic correction -- the joint-coupling,
// gravity/elastostatic-deflection, and shoulder_pitch Fourier terms from
// this project's offline 60-param calibration model (see CLAUDE.md for
// derivation/validation history). robot_calibration.cpp/the deployed URDF
// only carry the STATIC part of that model (offset/scale/tilt/origin);
// these terms are functions of the current joint configuration, so they
// can't be baked into a static geometry file and must be evaluated fresh
// each control cycle -- hence their own module, with the full 7-joint FK
// pass that requires.
//
// CONTROL DIRECTION: the offline model predicts true angle = f(commanded
// angle). For control we need the inverse -- what to command so the true
// angle lands where MoveIt wants. Approximated as one Newton step:
// evaluate the deviation AT the desired angle and subtract it. Valid since
// every correction here is a small perturbation relative to the joint
// ranges involved (see CLAUDE.md for why this and one other small
// approximation -- evaluating against the offset/scale-corrected angle
// rather than the raw encoder angle the Python model was fit against --
// are both negligible here).
//
// elbow_yaw (index 4) is permanently locked to a ~4-degree window and gets
// no correction here either, for the same reason as its offset/scale in
// robot_calibration.cpp: poorly identified (barely moved in the fit
// dataset) and moot since it's never commanded away from its lock.
//
// STATUS: unit-checked against the Python model's own math but NEVER
// validated against real hardware. Gated behind an opt-in parameter for
// exactly that reason -- do not treat as trustworthy without a real A/B
// physical test.

namespace pose_dependent_correction {

// jointAnglesRad: all 7 joints' current desired angles (radians), offset/
// scale already applied (same convention as robot_calibration.cpp),
// indexed 0=shoulder_roll..6=wrist_roll.
//
// Returns the additive correction (radians) to SUBTRACT from each joint's
// desired angle before converting to ticks. correction[4] is always 0.0.
std::array<double, 7> computeCorrection(const std::array<double, 7>& jointAnglesRad);

}  // namespace pose_dependent_correction
