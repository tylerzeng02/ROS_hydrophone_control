#pragma once

#include <array>

// Live, pose-dependent kinematic correction -- the joint-coupling,
// gravity/elastostatic-deflection, and shoulder_pitch Fourier terms from
// this project's offline 60-param calibration model
// (calibration/current/final_deployment_fit.py, fit on the elbow-yaw-locked
// 374-pose dataset -- see CLAUDE.md's kinematic-calibration section for the
// full derivation/validation history of each term).
//
// WHY THIS EXISTS: robot_calibration.cpp's jointCalibrations (zeroTick,
// direction, scale) and the deployed URDF's per-joint <axis>/<origin> only
// carry the STATIC part of that model (offset folded into zeroTick, scale,
// axis tilt, 3 origin corrections). Coupling/gravity/Fourier are functions
// of the CURRENT joint configuration, not a fixed per-joint constant, so
// they cannot be expressed as a static URDF geometry change or a
// zeroTick/scale edit -- they need to be evaluated fresh every control
// cycle against whatever pose is currently commanded. This is that
// evaluation, extracted into its own module (not folded into
// robot_calibration.cpp) specifically because it needs a small forward-
// kinematics pass across all 7 joints together, unlike every existing
// function in robot_calibration.cpp, which is single-joint/independent.
//
// CONTROL DIRECTION: the offline model is a FORWARD statement -- "given
// these (offset+scale-corrected) commanded joint angles, the TRUE angles
// (and hence true position) differ by this pose-dependent amount." For
// control we need the INVERSE: given the angles MoveIt wants to be
// TRUE (hw_commands_, already offset/scale-corrected per
// robot_calibration.cpp/the URDF's static model), what should we actually
// command so the true angles end up there once the pose-dependent effects
// apply? Implemented as a single first-order (one Newton-step) correction:
// evaluate the predicted deviation AT the desired angles and subtract it
// before converting to ticks. Since every one of these corrections is a
// small perturbation (low-single-digit degrees) relative to the ~90-180
// degree joint ranges involved, the difference between evaluating at the
// desired vs. the (unknown, to-be-achieved) true angles is second-order
// and negligible -- this is the same approximation this project already
// documented and accepted for other small corrections (see CLAUDE.md).
//
// A second approximation, also deliberate: the Python model's coupling/
// Fourier terms are functions of the RAW (pre-offset/scale) encoder angle,
// but this port evaluates them directly against the offset/scale-corrected
// angle (hw_commands_) instead of first inverting scale/offset to recover
// the raw angle. Offset (a few degrees) and scale (within ~2% of 1.0) are
// both small relative to the joints' own ranges, and the corrections being
// computed are themselves already small perturbations -- so this
// substitution is a second-order-in-a-second-order-term effect, well
// beneath anything measurable here. Documented rather than silently
// assumed.
//
// elbow_yaw (motor 4, joint index 4) is permanently locked to a ~4-degree
// window (see CLAUDE.md, "elbow_yaw permanently locked") and, per the same
// reasoning already applied to its offset/scale in robot_calibration.cpp,
// gets NEITHER a gravity correction NOR the coupling term that targets it
// (shoulder_yaw*elbow_yaw -> elbow_yaw) applied here -- that joint's own
// dynamic correction was poorly identified in the fit (it barely moved in
// the dataset) and is moot in production since it's never commanded away
// from its lock.
//
// STATUS (2026-08-13): implemented and unit-checked against the Python
// model's own math (see calibration/current/verify_pose_dependent_port.py)
// but NEVER validated against real hardware -- same standing as the
// streaming backlash compensator in cyton_hardware when it was first
// written. Gated behind an opt-in parameter for exactly that reason; do
// not treat as trustworthy until it has been through the same kind of
// real A/B physical test this project applies to everything else.

namespace pose_dependent_correction {

// jointAnglesRad: all 7 joints' CURRENT DESIRED angles (radians), same
// convention as robot_calibration.cpp's ticksToRadians()/JointCalibration
// (offset+scale already applied, i.e. what hw_commands_/hw_positions_
// already store) -- index order matches jointCalibrations (0=shoulder_roll
// .. 6=wrist_roll).
//
// Returns the ADDITIVE correction (radians) to SUBTRACT from each joint's
// desired angle before converting to ticks (see the control-direction note
// above) -- correction[4] (elbow_yaw) is always exactly 0.0.
std::array<double, 7> computeCorrection(const std::array<double, 7>& jointAnglesRad);

}  // namespace pose_dependent_correction
