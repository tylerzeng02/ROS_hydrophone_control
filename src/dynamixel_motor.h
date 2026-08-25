#ifndef DYNAMIXEL_MOTOR_H
#define DYNAMIXEL_MOTOR_H

#include <cstdint>
#include <vector>
#include <string>
#include "dynamixel_sdk/dynamixel_sdk.h"

/**
 * @brief Per-joint backlash overshoot margin, in ticks.
 *
 * Each value is the joint's measured backlash gap (from hand-verified
 * reversal tests) plus a modest safety margin. The margin is kept modest
 * because overshooting past the true gap only adds settling noise, not
 * correction benefit. Used by moveJointSafely()/moveJointsSafely() when
 * `compensateBacklash` is true.
 *
 * @param motorId Motor ID (0-6 for the arm's 7 joints).
 * @return Overshoot margin in ticks. Motor IDs outside 0-6 (e.g. the
 *         gripper) get a fallback value of 20.
 */
int getBacklashOvershootTicks(int motorId);

/**
 * @brief Low-level driver for the Cyton Gamma 1500's Dynamixel servos.
 *
 * Wraps the Dynamixel SDK's PortHandler/PacketHandler for connect,
 * torque, and position/speed primitives, plus safety-checked single- and
 * multi-joint moves (moveJointSafely()/moveJointsSafely()/moveTrajectory())
 * that clamp against src/robot_calibration.h's jointCalibrations table
 * before writing to the servos. Every motor-facing program in this
 * project goes through this class for safety clamping; joint limits are
 * not duplicated per caller.
 */
class DynamixelMotor
{
public:
    /**
     * @brief Constructs the driver without opening the port. Call
     * connect() to actually open it.
     *
     * @param deviceName Serial device path (e.g. "/dev/ttyUSB0" or "COM4").
     * @param baudRate Serial baud rate (e.g. 1000000).
     * @param protocolVersion Dynamixel protocol version (1.0 for this
     *        arm's MX-series servos).
     */
    DynamixelMotor(const char* deviceName, int baudRate, float protocolVersion);

    /**
     * @brief Closes the port if still connected.
     */
    ~DynamixelMotor();

    /**
     * @brief Opens the serial port and sets the configured baud rate.
     * @return True on success.
     */
    bool connect();

    /**
     * @brief Closes the serial port, if open. Safe to call when not
     * connected.
     */
    void disconnect();

    /**
     * @brief Pings a motor and prints its model number on success.
     * @param motorId Motor ID to ping.
     * @return True if the motor responded.
     */
    bool pingMotor(int motorId);

    /**
     * @brief Enables torque on a motor.
     * @param motorId Motor ID.
     * @return True on success.
     */
    bool enableTorque(int motorId);

    /**
     * @brief Disables torque on a motor.
     * @param motorId Motor ID.
     * @return True on success.
     */
    bool disableTorque(int motorId);

    /**
     * @brief Writes a goal position, after checking it against both the
     * servo's full raw range and this joint's calibrated safe range.
     *
     * @param motorId Motor ID.
     * @param position Goal position in raw ticks (0-4095).
     * @return True on success. False if the position fails either safety
     *         check, or the write itself fails.
     */
    bool setGoalPosition(int motorId, uint16_t position);

    /**
     * @brief Writes a moving speed.
     * @param motorId Motor ID.
     * @param speed Moving speed, 0-1023.
     * @return True on success.
     */
    bool setMovingSpeed(int motorId, uint16_t speed);

    /**
     * @brief Blocking, safety-checked move of a single joint, with
     * optional backlash compensation.
     *
     * Validates both the target and the current position against
     * jointCalibrations before moving, enables torque, commands the goal
     * position, then polls until the position is within `tolerance` or
     * `timeoutSeconds` elapses, disabling torque before returning either
     * way.
     *
     * If `compensateBacklash` is true and this move would arrive by
     * decreasing tick value, the joint first overshoots below the target
     * (clamped to jointCalibrations' minTick), via a recursive call with
     * `compensateBacklash=false` to prevent infinite recursion, so the
     * real approach always increases. This converts backlash from an
     * unpredictable direction-dependent error into a constant offset the
     * calibration's zeroTick/scale corrections already absorb. An
     * overshoot failure is not fatal: it warns and proceeds to the real
     * move anyway, since even an incomplete overshoot almost certainly
     * still served its purpose.
     *
     * @param motorId Motor ID.
     * @param targetPosition Target position in raw ticks.
     * @param speed Moving speed, 0-1023.
     * @param tolerance Ticks of error considered "reached".
     * @param timeoutSeconds Seconds to wait before giving up.
     * @param compensateBacklash Whether to apply the overshoot-then-
     *        approach backlash fix (see above).
     * @return True if the target was reached within tolerance and
     *         timeout. False on any safety-check failure, communication
     *         failure, or timeout.
     */
    bool moveJointSafely(
        int motorId,
        uint16_t targetPosition,
        uint16_t speed,
        int tolerance = 15,
        int timeoutSeconds = 8,
        bool compensateBacklash = true
    );

    /**
     * @brief Blocking, safety-checked synchronized move of multiple
     * joints, with optional backlash compensation, stall detection, and
     * a fine-correction pass.
     *
     * Backlash compensation (if `compensateBacklash` is true) is applied
     * per joint, in one pre-pass, before the main synchronized move: each
     * joint whose target is reached by decreasing tick value overshoots
     * below the target first (recursing with `compensateBacklash=false`
     * to prevent infinite recursion), same reasoning as
     * moveJointSafely(). `compensateOnlyJointIds`, if non-empty,
     * restricts which joint IDs get this treatment; used to isolate one
     * joint's backlash contribution for testing.
     *
     * The main move enables torque and writes goal positions for every
     * joint, then polls all of them together until every joint is within
     * its own tolerance (motor ID 7, the gripper, always uses a looser
     * 35-tick tolerance regardless of `tolerance`). If any joint's
     * current position falls outside its safe range mid-move (only
     * checked when `enforceSafetyLimits` is true), all motors are frozen
     * at their current position (torque stays enabled) and the call
     * fails immediately. If `stallRepeatsToDetect` is positive and the
     * total error across all joints stays unchanged for that many
     * consecutive polls, the move is treated as stalled and abandoned.
     * On success, a fine-correction pass re-writes any joint still more
     * than a few ticks off, since a fresh goal write can nudge a servo's
     * position-control loop closer than continuing to hold an old one.
     *
     * @param motorIds Motor IDs to move together.
     * @param targetPositions Target position per motor, same order and
     *        length as `motorIds`.
     * @param speed Moving speed applied to every motor, 0-1023.
     * @param tolerance Ticks of error considered "reached", for every
     *        joint except motor ID 7.
     * @param timeoutSeconds Seconds to wait before giving up.
     * @param holdTorque If true, leave torque enabled on all motors when
     *        this call returns (success, stall, or timeout). If false,
     *        torque is disabled before returning.
     * @param stallRepeatsToDetect If positive, abandon the move once the
     *        total error is unchanged for this many consecutive polls.
     *        0 disables stall detection.
     * @param enforceSafetyLimits If false, skips the jointCalibrations
     *        range checks and writes goal positions directly
     *        (writeGoalPositionRaw()) instead of through
     *        setGoalPosition(). Used internally by the backlash-overshoot
     *        pre-pass and for narrow, pre-verified cases elsewhere in
     *        this project.
     * @param compensateBacklash Whether to apply the per-joint backlash
     *        overshoot pre-pass described above.
     * @param compensateOnlyJointIds If non-empty, restricts backlash
     *        compensation to these motor IDs; other joints in the move
     *        are left uncompensated.
     * @return True if every joint reached its target within tolerance and
     *         timeout. False on a size mismatch, empty input, any
     *         safety-check failure, communication failure, stall
     *         detection, or timeout.
     */
    bool moveJointsSafely(
        const std::vector<int>& motorIds,
        const std::vector<uint16_t>& targetPositions,
        uint16_t speed,
        int tolerance = 15,
        int timeoutSeconds = 10,
        bool holdTorque = false,
        int stallRepeatsToDetect = 0,
        bool enforceSafetyLimits = true,
        bool compensateBacklash = true,
        const std::vector<int>& compensateOnlyJointIds = {}
    );

    /**
     * @brief Convenience wrapper: moves all 8 motors (7 arm joints plus
     * the gripper) via moveJointsSafely().
     *
     * @param targetPositions Exactly 8 target positions, for motor IDs
     *        0-7 in order.
     * @param speed Moving speed, 0-1023.
     * @param tolerance Ticks of error considered "reached".
     * @param timeoutSeconds Seconds to wait before giving up.
     * @param holdTorque If true, leave torque enabled when this call
     *        returns.
     * @return True on success. False if `targetPositions` does not
     *         contain exactly 8 entries, or moveJointsSafely() fails.
     */
    bool moveToPose(
        const std::vector<uint16_t>& targetPositions,
        uint16_t speed,
        int tolerance = 15,
        int timeoutSeconds = 10,
        bool holdTorque = false
    );

    /**
     * @brief Reads a motor's present position.
     * @param motorId Motor ID.
     * @param[out] position Present position in raw ticks, on success.
     * @return True on success.
     */
    bool readPosition(int motorId, uint16_t& position);

    /**
     * @brief Reads a motor's present speed.
     * @param motorId Motor ID.
     * @param[out] speed Present speed, on success.
     * @return True on success.
     */
    bool readSpeed(int motorId, uint16_t& speed);

    /**
     * @brief Reads a motor's present load.
     * @param motorId Motor ID.
     * @param[out] load Raw present-load register value, on success. Bits
     *        0-9 are magnitude (0-1023, scale to percent by /1023*100),
     *        bit 10 is direction.
     * @return True on success.
     */
    bool readLoad(int motorId, uint16_t& load);

    /**
     * @brief Reads a motor's present voltage.
     * @param motorId Motor ID.
     * @param[out] voltage Raw present-voltage register value, on success.
     *        Divide by 10.0 for volts.
     * @return True on success.
     */
    bool readVoltage(int motorId, uint8_t& voltage);

    /**
     * @brief Reads a motor's present temperature.
     * @param motorId Motor ID.
     * @param[out] temperature Present temperature in Celsius, on success.
     * @return True on success.
     */
    bool readTemperature(int motorId, uint8_t& temperature);

    /**
     * @brief Reads a motor's CW/CCW angle limit registers.
     * @param motorId Motor ID.
     * @param[out] cwLimit CW angle limit, on success.
     * @param[out] ccwLimit CCW angle limit, on success.
     * @return True on success.
     */
    bool readAngleLimits(int motorId, uint16_t& cwLimit, uint16_t& ccwLimit);

    /**
     * @brief Writes a motor's CW/CCW angle limit registers.
     * @param motorId Motor ID.
     * @param cwLimit CW angle limit to write.
     * @param ccwLimit CCW angle limit to write.
     * @return True on success.
     */
    bool writeAngleLimits(int motorId, uint16_t cwLimit, uint16_t ccwLimit);

    /**
     * @brief Reads and prints one motor's voltage, load, load direction,
     * and temperature.
     * @param motorId Motor ID.
     * @return True if all three reads succeeded.
     */
    bool printElectricalStatus(int motorId);

    /**
     * @brief Calls printElectricalStatus() for each motor in `motorIds`.
     * @param motorIds Motor IDs to report on.
     * @return True only if every motor's status read succeeded.
     */
    bool printElectricalStatusForMotors(const std::vector<int>& motorIds);

    /**
     * @brief Converts an angle to ticks via jointCalibrations and writes
     * it as a goal position, after a range check. Unlike
     * moveJointSafely(), this does not enable torque, poll for arrival,
     * or apply backlash compensation; it is a single non-blocking write.
     *
     * @param motorId Motor ID.
     * @param radians Target joint angle in radians.
     * @return True on success. False if `motorId` is invalid, the
     *         converted tick falls outside this joint's calibrated
     *         range, or the write fails.
     */
    bool moveJointRadians(int motorId, double radians);

    /**
     * @brief Converts a full 7-joint radians pose to ticks via
     * jointCalibrations and moves through moveToPose().
     *
     * @param jointRadians Exactly jointCalibrations.size() joint angles,
     *        in radians, one per joint in calibration-table order.
     * @param speed Moving speed, 0-1023.
     * @param tolerance Ticks of error considered "reached".
     * @param timeoutSeconds Seconds to wait before giving up.
     * @param holdTorque If true, leave torque enabled when this call
     *        returns.
     * @return True on success. False if `jointRadians` has the wrong
     *         length, any converted tick falls outside its joint's
     *         calibrated range, or the underlying move fails.
     */
    bool moveJointRadiansPose(
        const std::vector<double>& jointRadians,
        uint16_t speed,
        int tolerance = 15,
        int timeoutSeconds = 10,
        bool holdTorque = true
    );

    /**
     * @brief Public wrapper checking whether a raw position falls within
     * this joint's calibrated safe range.
     * @param motorId Motor ID.
     * @param position Position in raw ticks to check.
     * @return True if within jointCalibrations' minTick/maxTick for this
     *         motor.
     */
    bool isPositionSafe(int motorId, uint16_t position) const;

    /**
     * @brief Returns the electrical status snapshot captured during the
     * most recent moveJointsSafely() call (captured either mid-move, once
     * the total error first drops below half its starting value, or at
     * completion if that never happened).
     * @return The snapshot text, or an empty string if no
     *         moveJointsSafely() call has completed yet.
     */
    std::string getLastElectricalSnapshot() const;

    /**
     * @brief Visits a sequence of full 8-motor poses in order via
     * moveToPose().
     * @param trajectory Sequence of poses, each exactly 8 target
     *        positions (motor IDs 0-7).
     * @param speed Moving speed applied to every pose, 0-1023.
     * @param tolerance Ticks of error considered "reached", per pose.
     * @param timeoutSeconds Seconds to wait per pose before giving up.
     * @param holdTorque If true, leave torque enabled between poses and
     *        when this call returns.
     * @return True if every pose in the trajectory was reached. False if
     *         `trajectory` is empty, or any pose fails.
     */
    bool moveTrajectory(
        const std::vector<std::vector<uint16_t>>& trajectory,
        uint16_t speed,
        int tolerance = 15,
        int timeoutSeconds = 10,
        bool holdTorque = true
    );

    /**
     * @brief Disables torque on every listed motor and disconnects.
     * @param motorIds Motor IDs to disable torque on.
     */
    void emergencyShutdown(const std::vector<int>& motorIds);

    /**
     * @brief Looks up a motor ID from a joint name ("joint0".."joint7").
     * @param jointName Joint name to look up.
     * @return The motor ID, or -1 if `jointName` is not recognized.
     */
    int jointNameToMotorId(const std::string& jointName) const;

    /**
     * @brief Converts an angle to ticks via jointCalibrations and moves
     * the named joint through moveJointSafely(), after a range check.
     *
     * @param jointName Joint name ("joint0".."joint7"); see
     *        jointNameToMotorId().
     * @param radians Target joint angle in radians.
     * @param speed Moving speed, 0-1023.
     * @param tolerance Ticks of error considered "reached".
     * @param timeoutSeconds Seconds to wait before giving up.
     * @return True on success. False if `jointName` is unrecognized, the
     *         converted tick falls outside this joint's calibrated
     *         range, or the underlying move fails.
     */
    bool moveNamedJointRadians(
        const std::string& jointName,
        double radians,
        uint16_t speed,
        int tolerance = 15,
        int timeoutSeconds = 8
    );

    /**
     * @brief Writes a goal position directly, bypassing the
     * jointCalibrations range check that setGoalPosition() applies.
     *
     * Used internally where the range check has already been done (or
     * deliberately skipped via `enforceSafetyLimits=false`), and for
     * freezing a motor at its own current position when that position is,
     * by definition, out of range and would be rejected by
     * setGoalPosition().
     *
     * @param motorId Motor ID.
     * @param position Goal position in raw ticks.
     * @return True on success.
     */
    bool writeGoalPositionRaw(int motorId, uint16_t position);

    /**
     * @brief Reads a motor's Punch register (minimum current floor below
     * which the servo generates no corrective torque).
     * @param motorId Motor ID.
     * @param[out] punch Punch value, on success.
     * @return True on success.
     */
    bool readPunch(int motorId, uint16_t& punch);

    /**
     * @brief Writes a motor's Punch register.
     * @param motorId Motor ID.
     * @param punch Punch value to write.
     * @return True on success.
     */
    bool writePunch(int motorId, uint16_t punch);

    /**
     * @brief Reads a motor's model number register. Used to distinguish
     * MX-64 (model 310, shoulder joints) from MX-28 (model 29, elbow and
     * wrist joints) on this arm.
     * @param motorId Motor ID.
     * @param[out] modelNumber Model number, on success.
     * @return True on success.
     */
    bool readModelNumber(int motorId, uint16_t& modelNumber);

    /**
     * @brief Reads a motor's D/I/P PID gain registers (Protocol 1.0
     * addresses 26/27/28).
     * @param motorId Motor ID.
     * @param[out] dGain D gain, on success.
     * @param[out] iGain I gain, on success.
     * @param[out] pGain P gain, on success.
     * @return True if all three reads succeeded.
     */
    bool readGains(int motorId, uint8_t& dGain, uint8_t& iGain, uint8_t& pGain);

    /**
     * @brief Writes a motor's D/I/P PID gain registers.
     * @param motorId Motor ID.
     * @param dGain D gain to write.
     * @param iGain I gain to write.
     * @param pGain P gain to write.
     * @return True if all three writes succeeded.
     */
    bool writeGains(int motorId, uint8_t dGain, uint8_t iGain, uint8_t pGain);

private:
    const char* deviceName_;
    int baudRate_;
    float protocolVersion_;
    bool connected_;

    dynamixel::PortHandler* portHandler_;
    dynamixel::PacketHandler* packetHandler_;

    /**
     * @brief Checks a Dynamixel SDK communication result and error byte
     * together, logging a descriptive message on either failure.
     * @param commResult Return value from a PacketHandler Tx/Rx call.
     * @param dxlError Error byte populated by the same call.
     * @param motorId Motor ID, for the log message.
     * @param action Short description of the attempted action, for the
     *        log message.
     * @return True if `commResult` was COMM_SUCCESS and `dxlError` was 0.
     */
    bool checkCommResult(int commResult, uint8_t dxlError, int motorId, const char* action);

    /**
     * @brief Checks a raw position against this joint's calibrated
     * minTick/maxTick range.
     * @param motorId Motor ID.
     * @param position Position in raw ticks to check.
     * @return True if within range. False if `motorId` is not a known
     *         joint, or `position` falls outside its range.
     */
    bool isPositionWithinLimit(int motorId, uint16_t position) const;

    static const int ADDR_MODEL_NUMBER = 0;
    static const int ADDR_CW_ANGLE_LIMIT = 6;
    static const int ADDR_CCW_ANGLE_LIMIT = 8;
    static const int ADDR_TORQUE_ENABLE = 24;
    static const int ADDR_D_GAIN = 26;
    static const int ADDR_I_GAIN = 27;
    static const int ADDR_P_GAIN = 28;
    static const int ADDR_GOAL_POSITION = 30;
    static const int ADDR_MOVING_SPEED = 32;
    static const int ADDR_PUNCH = 48;
    static const int ADDR_PRESENT_POSITION = 36;
    static const int ADDR_PRESENT_SPEED = 38;
    static const int ADDR_PRESENT_LOAD = 40;
    static const int ADDR_PRESENT_VOLTAGE = 42;
    static const int ADDR_PRESENT_TEMPERATURE = 43;

    static const int TORQUE_ENABLE = 1;
    static const int TORQUE_DISABLE = 0;

    static const uint16_t MIN_RAW_POSITION = 0;
    static const uint16_t MAX_RAW_POSITION = 4095;

    std::string lastElectricalSnapshot_;

    /**
     * @brief Builds a multi-line electrical status string (voltage, load,
     * temperature per motor) for the given motors, used to populate
     * lastElectricalSnapshot_.
     * @param motorIds Motor IDs to include.
     * @return The formatted snapshot text.
     */
    std::string buildElectricalSnapshot(const std::vector<int>& motorIds);

    std::vector<std::string> jointNames_;
};

#endif
