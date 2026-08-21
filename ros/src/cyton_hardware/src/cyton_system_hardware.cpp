#include "cyton_hardware/cyton_system_hardware.hpp"

#include <algorithm>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/logging.hpp"

namespace cyton_hardware
{

namespace
{
rclcpp::Logger logger()
{
  return rclcpp::get_logger("cyton_hardware");
}

// xacro args always arrive as strings, and xacro's ${...} property
// substitution runs values through YAML-style type coercion before
// re-stringifying, so a lowercase "true" can arrive here as "True" or
// "TRUE" -- accept all case variants rather than trusting exact case.
bool parseBoolParam(const std::string & value)
{
  return value == "true" || value == "True" || value == "TRUE" || value == "1";
}

// Ordinary trajectory-following noise near a momentary hold shouldn't look
// like a direction reversal. Untuned guess (see this file's own header
// comment on backlash compensation) -- not yet validated against real
// hardware.
constexpr int kDirectionDeadbandTicks = 2;
}  // namespace

hardware_interface::CallbackReturn CytonSystemHardware::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (
    hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  if (info_.joints.size() != static_cast<size_t>(kNumJoints))
  {
    RCLCPP_ERROR(
      logger(), "Expected %d joints (arm motors 0-6, gripper excluded), got %zu",
      kNumJoints, info_.joints.size());
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Enforce the URDF's ros2_control block declaring joints in exactly
  // motor-ID order (0=shoulder_roll .. 6=wrist_roll) -- read()/write()
  // below index straight into jointCalibrations by this same position,
  // so a silent name/order mismatch would command the wrong motor.
  for (int i = 0; i < kNumJoints; ++i)
  {
    if (info_.joints[static_cast<size_t>(i)].name != kJointNames[i])
    {
      RCLCPP_ERROR(
        logger(),
        "Joint %d in <ros2_control> must be '%s' (got '%s') -- joints must be listed in "
        "motor-ID order, matching jointCalibrations in robot_calibration.cpp",
        i, kJointNames[i], info_.joints[static_cast<size_t>(i)].name.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  auto getParam = [this](const std::string & name, const std::string & defaultValue) -> std::string
  {
    auto it = info_.hardware_parameters.find(name);
    return it != info_.hardware_parameters.end() ? it->second : defaultValue;
  };

  serial_port_ = getParam("serial_port", "/dev/ttyUSB0");
  baud_rate_ = std::stoi(getParam("baud_rate", "1000000"));
  protocol_version_ = std::stof(getParam("protocol_version", "1.0"));
  moving_speed_ = static_cast<uint16_t>(std::stoi(getParam("moving_speed", "40")));
  compensate_backlash_ = parseBoolParam(getParam("compensate_backlash", "false"));

  hw_positions_.fill(0.0);
  hw_velocities_.fill(0.0);
  hw_commands_.fill(0.0);

  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> CytonSystemHardware::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (int i = 0; i < kNumJoints; ++i)
  {
    state_interfaces.emplace_back(
      kJointNames[i], hardware_interface::HW_IF_POSITION, &hw_positions_[static_cast<size_t>(i)]);
    state_interfaces.emplace_back(
      kJointNames[i], hardware_interface::HW_IF_VELOCITY, &hw_velocities_[static_cast<size_t>(i)]);
  }
  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> CytonSystemHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  for (int i = 0; i < kNumJoints; ++i)
  {
    command_interfaces.emplace_back(
      kJointNames[i], hardware_interface::HW_IF_POSITION, &hw_commands_[static_cast<size_t>(i)]);
  }
  return command_interfaces;
}

hardware_interface::CallbackReturn CytonSystemHardware::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  // Fresh backlash-compensation state every activation -- no stale
  // direction/hold data carried over from a previous run.
  last_raw_tick_.fill(0);
  last_direction_.fill(0);
  has_previous_tick_.fill(false);
  hold_active_.fill(false);
  hold_point_tick_.fill(0);
  hold_direction_.fill(0);

  if (compensate_backlash_)
  {
    RCLCPP_WARN(
      logger(),
      "compensate_backlash is ENABLED -- this is a NEW, real design (not a guess) but has "
      "NEVER been validated against real hardware, unlike dynamixel_motor.cpp's blocking-move "
      "fix. Watch closely.");
  }

  motor_ = std::make_unique<DynamixelMotor>(serial_port_.c_str(), baud_rate_, protocol_version_);

  if (!motor_->connect())
  {
    RCLCPP_ERROR(logger(), "Failed to open serial port '%s'", serial_port_.c_str());
    motor_.reset();
    return hardware_interface::CallbackReturn::ERROR;
  }

  for (int i = 0; i < kNumJoints; ++i)
  {
    const JointCalibration & calibration = jointCalibrations[static_cast<size_t>(i)];

    if (!motor_->enableTorque(calibration.id))
    {
      RCLCPP_ERROR(logger(), "Failed to enable torque on motor %d (%s)", calibration.id, kJointNames[i]);
      return hardware_interface::CallbackReturn::ERROR;
    }

    if (!motor_->setMovingSpeed(calibration.id, moving_speed_))
    {
      RCLCPP_WARN(
        logger(), "Failed to set moving speed on motor %d (%s), continuing with servo default",
        calibration.id, kJointNames[i]);
    }

    uint16_t tick = 0;
    if (motor_->readPosition(calibration.id, tick))
    {
      const double radians = ticksToRadians(calibration, static_cast<int>(tick));
      hw_positions_[static_cast<size_t>(i)] = radians;
      // Seed the command with the real current position so the first
      // write() cycle doesn't command a jump from wherever hw_commands_
      // happened to default-initialize to (0.0) back to this joint's
      // actual current angle.
      hw_commands_[static_cast<size_t>(i)] = radians;
    }
    else
    {
      RCLCPP_ERROR(
        logger(), "Failed to read initial position from motor %d (%s)", calibration.id,
        kJointNames[i]);
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  RCLCPP_INFO(logger(), "CytonSystemHardware activated on '%s'", serial_port_.c_str());
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn CytonSystemHardware::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  if (motor_)
  {
    for (int i = 0; i < kNumJoints; ++i)
    {
      motor_->disableTorque(jointCalibrations[static_cast<size_t>(i)].id);
    }
    motor_->disconnect();
    motor_.reset();
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type CytonSystemHardware::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  if (!motor_)
  {
    return hardware_interface::return_type::ERROR;
  }

  for (int i = 0; i < kNumJoints; ++i)
  {
    const JointCalibration & calibration = jointCalibrations[static_cast<size_t>(i)];
    uint16_t tick = 0;
    if (motor_->readPosition(calibration.id, tick))
    {
      hw_positions_[static_cast<size_t>(i)] = ticksToRadians(calibration, static_cast<int>(tick));
    }
    else
    {
      // Transient comm hiccup: log and keep the last known position rather
      // than erroring the whole hardware interface out over one dropped
      // packet. hw_positions_ simply doesn't advance for this joint this
      // cycle.
      RCLCPP_ERROR(
        logger(), "read() failed on motor %d (%s), keeping last known position", calibration.id,
        kJointNames[i]);
    }
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type CytonSystemHardware::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  if (!motor_)
  {
    return hardware_interface::return_type::ERROR;
  }

  for (int i = 0; i < kNumJoints; ++i)
  {
    const JointCalibration & calibration = jointCalibrations[static_cast<size_t>(i)];
    int tick = radiansToTicks(calibration, hw_commands_[static_cast<size_t>(i)]);

    if (tick < 0 || tick > 4095)
    {
      RCLCPP_ERROR(
        logger(), "Commanded %s = %.4f rad converts to out-of-range tick %d, skipping",
        kJointNames[i], hw_commands_[static_cast<size_t>(i)], tick);
      continue;
    }

    if (compensate_backlash_)
    {
      tick = applyBacklashCompensation(i, tick);
    }

    // setGoalPosition() re-checks jointCalibrations' min/maxTick itself
    // before writing -- same safety gate as every other motor-facing
    // program here, and a backstop against an out-of-range hold point.
    if (!motor_->setGoalPosition(calibration.id, static_cast<uint16_t>(tick)))
    {
      RCLCPP_ERROR(
        logger(), "setGoalPosition rejected/failed for motor %d (%s), target tick %d",
        calibration.id, kJointNames[i], tick);
    }
  }

  return hardware_interface::return_type::OK;
}

// Reversal-triggered hold-point backlash compensator (see this class's
// header for why not a per-cycle port of dynamixel_motor.cpp's
// blocking-move overshoot fix). Per joint, per write() cycle:
//   1. Compare this cycle's raw tick against last cycle's to determine
//      the raw target's direction (+1/-1), ignoring moves smaller than
//      kDirectionDeadbandTicks (ordinary trajectory-interpolation jitter).
//   2. On a genuine reversal (an established direction flips), pin a hold
//      point at raw_tick +/- getBacklashOvershootTicks(jointIndex) in the
//      new direction, clamped to this joint's safety range.
//   3. While a hold is active, clamp the tick sent to the servo to be at
//      least as far along as the hold point (max()/min() depending on
//      direction) until the raw trajectory naturally reaches/passes it,
//      at which point the hold releases and 1:1 tracking resumes.
//
// Engages once per genuine reversal, not every cycle of a monotonic
// segment, unlike a naive per-cycle port.
int CytonSystemHardware::applyBacklashCompensation(int jointIndex, int rawTick)
{
  const size_t idx = static_cast<size_t>(jointIndex);
  const JointCalibration & calibration = jointCalibrations[idx];

  int currentDirection = 0;
  if (has_previous_tick_[idx])
  {
    const int diff = rawTick - last_raw_tick_[idx];
    if (diff > kDirectionDeadbandTicks)
    {
      currentDirection = 1;
    }
    else if (diff < -kDirectionDeadbandTicks)
    {
      currentDirection = -1;
    }
  }

  const int previousDirection = last_direction_[idx];
  const bool isRealReversal = has_previous_tick_[idx] && previousDirection != 0 &&
    currentDirection != 0 && currentDirection != previousDirection;

  if (isRealReversal)
  {
    const int overshoot = getBacklashOvershootTicks(jointIndex);
    int holdPoint = rawTick + currentDirection * overshoot;
    holdPoint = std::clamp(holdPoint, calibration.minTick, calibration.maxTick);

    hold_active_[idx] = true;
    hold_direction_[idx] = currentDirection;
    hold_point_tick_[idx] = holdPoint;

    RCLCPP_INFO(
      logger(), "Backlash reversal on %s: raw target=%d, holding at %d (overshoot %d ticks) "
      "until the raw target catches up",
      kJointNames[jointIndex], rawTick, holdPoint, overshoot);
  }

  int effectiveTick = rawTick;
  if (hold_active_[idx])
  {
    if (hold_direction_[idx] > 0)
    {
      effectiveTick = std::max(rawTick, hold_point_tick_[idx]);
      if (rawTick >= hold_point_tick_[idx])
      {
        hold_active_[idx] = false;  // raw target caught up on its own -- release the hold
      }
    }
    else
    {
      effectiveTick = std::min(rawTick, hold_point_tick_[idx]);
      if (rawTick <= hold_point_tick_[idx])
      {
        hold_active_[idx] = false;
      }
    }
  }

  effectiveTick = std::clamp(effectiveTick, calibration.minTick, calibration.maxTick);

  if (currentDirection != 0)
  {
    last_direction_[idx] = currentDirection;
  }
  last_raw_tick_[idx] = rawTick;
  has_previous_tick_[idx] = true;

  return effectiveTick;
}

CytonSystemHardware::~CytonSystemHardware()
{
  if (motor_)
  {
    for (int i = 0; i < kNumJoints; ++i)
    {
      motor_->disableTorque(jointCalibrations[static_cast<size_t>(i)].id);
    }
    motor_->disconnect();
  }
}

}  // namespace cyton_hardware

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(cyton_hardware::CytonSystemHardware, hardware_interface::SystemInterface)
