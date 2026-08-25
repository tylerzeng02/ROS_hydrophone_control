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

namespace cyton_hardware
{

/**
 * @brief ros2_control SystemInterface plugin wrapping this repo's own
 * DynamixelMotor/robot_calibration (../../../src, compiled directly by
 * this package's CMakeLists.txt rather than duplicated).
 *
 * Exposes the 7 arm joints (motor IDs 0-6; motor 7, the gripper, is
 * intentionally excluded, matching the URDF's IK chain) as position
 * command and state interfaces.
 *
 * @note Backlash compensation is opt-in via the "compensate_backlash"
 * hardware parameter (default "false"). The existing, hardware-validated
 * fix in dynamixel_motor.cpp is designed around a single blocking move to
 * one known target, and has no well-defined meaning for a streaming
 * position command where "the target" a reversal should overshoot below
 * is not known in advance. A naive per-cycle port would fire on nearly
 * every cycle of a decreasing segment, fighting the trajectory
 * controller's own interpolation. Implemented instead:
 * applyBacklashCompensation(), a reversal-triggered hold-point
 * compensator that only engages once per genuine reversal. Never
 * validated against real hardware; enable deliberately and watch
 * closely, not as a default.
 */
class CytonSystemHardware : public hardware_interface::SystemInterface
{
public:
  /**
   * @brief Validates the URDF's <ros2_control> joint declarations against
   * this plugin's fixed 7-joint, motor-ID-ordered expectation, and reads
   * hardware parameters (serial_port, baud_rate, protocol_version,
   * moving_speed, compensate_backlash).
   * @param info Hardware description parsed from the URDF.
   * @return SUCCESS if the joint count and names/order match
   *         kJointNames exactly; ERROR otherwise.
   */
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;

  /**
   * @brief Exports position and velocity state interfaces for all 7
   * joints. Velocity is always reported 0.0; see hw_velocities_.
   * @return The exported state interfaces.
   */
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  /**
   * @brief Exports position command interfaces for all 7 joints.
   * @return The exported command interfaces.
   */
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  /**
   * @brief Opens the serial port, enables torque and sets moving speed on
   * every joint, and seeds hw_commands_ with each joint's real current
   * position so the first write() does not command a jump from a
   * default-initialized 0.0. Also resets all backlash-compensation state.
   * @param previous_state Unused.
   * @return SUCCESS if the port opened and every joint's torque, speed,
   *         and initial position were set/read successfully; ERROR
   *         otherwise.
   */
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  /**
   * @brief Disables torque on every joint and closes the serial port.
   * @param previous_state Unused.
   * @return Always SUCCESS.
   */
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  /**
   * @brief Reads present position from every joint into hw_positions_.
   * On a per-joint read failure, logs and keeps that joint's last known
   * position rather than failing the whole cycle.
   * @param time Unused.
   * @param period Unused.
   * @return ERROR if the motor driver is not connected; otherwise OK.
   */
  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  /**
   * @brief Converts each joint's commanded radians (hw_commands_) to
   * ticks and writes it as a goal position, applying backlash
   * compensation first if enabled. A conversion that lands outside
   * 0-4095 is skipped with a logged error rather than written.
   * @param time Unused.
   * @param period Unused.
   * @return ERROR if the motor driver is not connected; otherwise OK.
   */
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  /**
   * @brief Disables torque on every joint and disconnects, if still
   * connected.
   */
  ~CytonSystemHardware() override;

private:
  /**
   * @brief Reversal-triggered hold-point backlash compensator, applied to
   * the raw commanded tick for one joint before it is sent to
   * setGoalPosition(). Only called when compensate_backlash_ is true. See
   * this class's own header comment above for the design rationale.
   *
   * Per joint, per write() cycle:
   *  1. Compares this cycle's raw tick against last cycle's to determine
   *     the raw target's direction (+1/-1), ignoring moves smaller than
   *     kDirectionDeadbandTicks (ordinary trajectory-interpolation
   *     jitter).
   *  2. On a genuine reversal (an established direction flips), pins a
   *     hold point at raw_tick +/- getBacklashOvershootTicks(jointIndex)
   *     in the new direction, clamped to this joint's safety range.
   *  3. While a hold is active, clamps the tick sent to the servo to be
   *     at least as far along as the hold point (max()/min() depending on
   *     direction) until the raw trajectory naturally reaches or passes
   *     it, at which point the hold releases and 1:1 tracking resumes.
   *
   * Engages once per genuine reversal, not every cycle of a monotonic
   * segment, unlike a naive per-cycle port of dynamixel_motor.cpp's
   * blocking-move overshoot fix.
   *
   * @param jointIndex Index into kJointNames/jointCalibrations (0-6).
   * @param rawTick This cycle's raw, uncompensated commanded tick.
   * @return The compensated tick to actually send to the servo.
   */
  int applyBacklashCompensation(int jointIndex, int rawTick);


  /**
   * @brief Motor IDs 0-6, in the same order as jointCalibrations and the
   * URDF's kinematic chain (shoulder_roll -> ... -> wrist_roll).
   */
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
  uint16_t moving_speed_ = 40;  ///< Matches this project's established MOVING_SPEED convention.

  /**
   * @brief ros2_control state/command storage. hw_velocities_ is always
   * reported 0.0, since DynamixelMotor::readPosition() only reads present
   * position, not present speed, on this read path.
   */
  std::array<double, kNumJoints> hw_positions_{};
  std::array<double, kNumJoints> hw_velocities_{};
  std::array<double, kNumJoints> hw_commands_{};

  /**
   * @brief Backlash compensation state, only meaningful when
   * compensate_backlash_ is true; reset fresh on every on_activate().
   * Tracks, per joint: the raw (uncompensated) commanded tick and travel
   * direction across write() cycles, and the currently-held
   * reversal-recovery target if any.
   */
  bool compensate_backlash_ = false;
  std::array<int, kNumJoints> last_raw_tick_{};
  std::array<int, kNumJoints> last_direction_{};       ///< -1, 0 (none yet), or +1.
  std::array<bool, kNumJoints> has_previous_tick_{};
  std::array<bool, kNumJoints> hold_active_{};
  std::array<int, kNumJoints> hold_point_tick_{};
  std::array<int, kNumJoints> hold_direction_{};       ///< -1 or +1, valid only while hold_active_.
};

}  // namespace cyton_hardware

#endif  // CYTON_HARDWARE__CYTON_SYSTEM_HARDWARE_HPP_
