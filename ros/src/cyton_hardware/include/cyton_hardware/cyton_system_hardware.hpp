#ifndef CYTON_HARDWARE__CYTON_SYSTEM_HARDWARE_HPP_
#define CYTON_HARDWARE__CYTON_SYSTEM_HARDWARE_HPP_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp_lifecycle/state.hpp"

#include "dynamixel_motor.h"
#include "robot_calibration.h"
#include "pose_dependent_correction.h"

namespace cyton_hardware
{

// ros2_control SystemInterface plugin wrapping this repo's own
// DynamixelMotor/robot_calibration (../../../src -- see this package's
// CMakeLists.txt, which compiles those files directly rather than
// duplicating them). Exposes the 7 arm joints (motor IDs 0-6; motor 7,
// the gripper, is intentionally excluded, matching the URDF's IK chain)
// as position command/state interfaces.
//
// Backlash compensation (2026-08-13, opt-in via the "compensate_backlash"
// hardware param, default "false"): the existing, hardware-validated fix
// in dynamixel_motor.cpp (moveJointSafely/moveJointsSafely's overshoot-
// then-approach dance) is designed around a single blocking move to one
// known target -- it doesn't have a well-defined meaning for a streaming
// position command that changes every control cycle while a trajectory is
// executing, where "the target" a reversal should overshoot below isn't
// known in advance. A naive per-cycle port (inject an overshoot whenever
// this cycle's target is below last cycle's) would fire on nearly every
// cycle of any decreasing trajectory segment, fighting the trajectory
// controller's own interpolation rather than correcting backlash.
//
// Implemented instead: a reversal-triggered hold-point compensator (see
// applyBacklashCompensation() in the .cpp for the exact algorithm) that
// only engages once per genuine direction reversal, not every cycle. This
// has NEVER been validated against real hardware -- it's a real, thought-
// through design (not a guess), but the old blocking-move fix went
// through extensive real A/B testing (uniform vs. per-joint overshoot
// margins, several datasets) before being trusted, and this streaming
// version hasn't had any of that yet. Enable deliberately and watch
// closely, not as a default. See CLAUDE.md's kinematic-calibration
// section for the full backlash history.
//
// Pose-dependent correction (2026-08-13, opt-in via the
// "compensate_pose_dependent" hardware param, default "false"): applies
// pose_dependent_correction::computeCorrection() (joint coupling, lumped
// gravity/elastostatic deflection, shoulder_pitch Fourier term -- see that
// module's own header comment for the exact math and the control-direction
// derivation) to every joint's commanded angle, every write() cycle,
// before the existing static tick/radian conversion and backlash
// compensation. Same standing as compensate_backlash when it was first
// written: the underlying math is checked against the Python model it
// ports, but this has never been validated as LIVE CONTROL against real
// hardware. Enable deliberately and watch closely, not as a default.
class CytonSystemHardware : public hardware_interface::SystemInterface
{
public:
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  ~CytonSystemHardware() override;

private:
  // Reversal-triggered hold-point backlash compensator, applied to the raw
  // commanded tick for one joint before it's sent to setGoalPosition().
  // See this class's own header comment above for the design rationale;
  // see the .cpp for the exact per-cycle algorithm. Only called when
  // compensate_backlash_ is true.
  int applyBacklashCompensation(int jointIndex, int rawTick);


  // Motor IDs 0-6, in the same order as jointCalibrations and the URDF's
  // kinematic chain (shoulder_roll -> ... -> wrist_roll).
  static constexpr int kNumJoints = 7;
  static constexpr const char * kJointNames[kNumJoints] = {
    "shoulder_roll_joint",
    "shoulder_pitch_joint",
    "shoulder_yaw_joint",
    "elbow_pitch_joint",
    "elbow_yaw_joint",
    "wrist_pitch_joint",
    "wrist_roll_joint",
  };

  std::unique_ptr<DynamixelMotor> motor_;

  std::string serial_port_;
  int baud_rate_ = 1000000;
  float protocol_version_ = 1.0f;
  uint16_t moving_speed_ = 40;  // matches this project's established MOVING_SPEED convention
  bool compensate_pose_dependent_ = false;

  // ros2_control state/command storage. hw_velocities_ is always reported
  // 0.0 -- DynamixelMotor::readPosition() only reads present position, not
  // present speed, on this read path.
  std::array<double, kNumJoints> hw_positions_{};
  std::array<double, kNumJoints> hw_velocities_{};
  std::array<double, kNumJoints> hw_commands_{};

  // Backlash compensation state (only meaningful when compensate_backlash_
  // is true; reset fresh on every on_activate()). Tracks, per joint: the
  // raw (uncompensated) commanded tick and travel direction across write()
  // cycles, and the currently-held reversal-recovery target if any.
  bool compensate_backlash_ = false;
  std::array<int, kNumJoints> last_raw_tick_{};
  std::array<int, kNumJoints> last_direction_{};       // -1, 0 (none yet), +1
  std::array<bool, kNumJoints> has_previous_tick_{};
  std::array<bool, kNumJoints> hold_active_{};
  std::array<int, kNumJoints> hold_point_tick_{};
  std::array<int, kNumJoints> hold_direction_{};       // -1 or +1, valid only while hold_active_
};

}  // namespace cyton_hardware

#endif  // CYTON_HARDWARE__CYTON_SYSTEM_HARDWARE_HPP_
