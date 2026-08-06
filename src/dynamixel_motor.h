#ifndef DYNAMIXEL_MOTOR_H
#define DYNAMIXEL_MOTOR_H

#include <cstdint>
#include <vector>
#include <string>
#include "dynamixel_sdk/dynamixel_sdk.h"

// Per-joint backlash overshoot margin (ticks), tuned per motor from the
// hand-verified reversal tests documented in CLAUDE.md's kinematic-
// calibration section. Defined in dynamixel_motor.cpp; declared here (not
// just file-scope there) so other translation units -- notably the
// ros2_control hardware interface in ros/src/cyton_hardware -- can reuse
// the same tuned values instead of re-deriving/duplicating them.
int getBacklashOvershootTicks(int motorId);

class DynamixelMotor
{
public:
    DynamixelMotor(const char* deviceName, int baudRate, float protocolVersion);
    ~DynamixelMotor();

    bool connect();
    void disconnect();

    bool pingMotor(int motorId);

    bool enableTorque(int motorId);
    bool disableTorque(int motorId);

    bool setGoalPosition(int motorId, uint16_t position);
    bool setMovingSpeed(int motorId, uint16_t speed);

    // compensateBacklash (default true): if the joint would otherwise
    // arrive at targetPosition by DECREASING its tick value, first
    // overshoots below the target and approaches it a second time while
    // INCREASING -- so every move's final approach direction is the same,
    // regardless of which direction the joint started in. This is the fix
    // for the confirmed backlash finding in CLAUDE.md's kinematic-
    // calibration section: backlash is direction-dependent, so it's
    // invisible to any static per-joint correction (offset/scale/tilt/
    // origin) unless every real move always finishes approaching from the
    // same direction, converting it into an ordinary constant offset that
    // IS captured by those corrections.
    bool moveJointSafely(
        int motorId,
        uint16_t targetPosition,
        uint16_t speed,
        int tolerance = 15,
        int timeoutSeconds = 8,
        bool compensateBacklash = true
    );

    // compensateBacklash: see moveJointSafely() above -- applied per-joint
    // here (each joint independently overshoots only if IT would
    // otherwise arrive by decreasing).
    //
    // compensateOnlyJointIds: empty (default) means compensateBacklash
    // applies to every joint in motorIds, same as before. If non-empty,
    // ONLY joint IDs in this list are eligible for compensation --
    // everything else is left uncompensated regardless of
    // compensateBacklash, even if it would otherwise arrive by
    // decreasing. Added to isolate a single joint's backlash contribution
    // experimentally (see CLAUDE.md's kinematic-calibration section) --
    // without this, compensating every joint at once makes it impossible
    // to tell whether one specific joint's correction matters, since every
    // corrected joint adds its own extra overshoot-settling noise.
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

    bool moveToPose(
        const std::vector<uint16_t>& targetPositions,
        uint16_t speed,
        int tolerance = 15,
        int timeoutSeconds = 10,
        bool holdTorque = false
    );

    bool readPosition(int motorId, uint16_t& position);
    bool readSpeed(int motorId, uint16_t& speed);
    bool readLoad(int motorId, uint16_t& load);
    bool readVoltage(int motorId, uint8_t& voltage);
    bool readTemperature(int motorId, uint8_t& temperature);

    // Reads the servo's own EEPROM-stored CW/CCW angle limit registers --
    // independent of jointCalibrations, these are enforced by the servo's
    // firmware itself and will reject any goal position outside them
    // (RxPacketError "Angle limit error"), even if jointCalibrations
    // considers that position safe.
    bool readAngleLimits(int motorId, uint16_t& cwLimit, uint16_t& ccwLimit);

    // Writes the servo's own EEPROM-stored CW/CCW angle limit registers.
    // Callers are responsible for restoring the original values afterward
    // if they only meant to widen them temporarily -- these persist across
    // power cycles.
    bool writeAngleLimits(int motorId, uint16_t cwLimit, uint16_t ccwLimit);

    bool printElectricalStatus(int motorId);
    bool printElectricalStatusForMotors(const std::vector<int>& motorIds);

    bool moveJointRadians(int motorId, double radians);

    bool moveJointRadiansPose(
        const std::vector<double>& jointRadians,
        uint16_t speed,
        int tolerance = 15,
        int timeoutSeconds = 10,
        bool holdTorque = true
    );

    bool isPositionSafe(int motorId, uint16_t position) const;
    
    std::string getLastElectricalSnapshot() const;

    bool moveTrajectory(
        const std::vector<std::vector<uint16_t>>& trajectory,
        uint16_t speed,
        int tolerance = 15,
        int timeoutSeconds = 10,
        bool holdTorque = true
    );

    void emergencyShutdown(const std::vector<int>& motorIds);

    int jointNameToMotorId(const std::string& jointName) const;

    bool moveNamedJointRadians(
        const std::string& jointName,
        double radians,
        uint16_t speed,
        int tolerance = 15,
        int timeoutSeconds = 8
    );

    // Writes the goal position register directly, bypassing the
    // jointCalibrations safety gate that setGoalPosition() enforces.
    // ONLY for freezing a motor at a position it is already physically at
    // (e.g. moveJointsSafely()'s safety-stop handling, or
    // record_hand_poses.cpp locking a hand-verified pose that may sit
    // outside jointCalibrations' software range) -- never for commanding a
    // motor to a NEW position, since that would defeat the safety check
    // entirely.
    bool writeGoalPositionRaw(int motorId, uint16_t position);

    // Compliance margin/slope and punch (2026-07-29): AX-series Protocol
    // 1.0 registers controlling how the servo behaves near its goal
    // position -- margin is a dead-zone radius (ticks) where corrective
    // torque backs off, slope controls how steeply torque ramps down
    // approaching goal (lower = stiffer/more aggressive near target),
    // punch is a minimum-current floor for overcoming static friction.
    // Factory defaults: margin=1, slope=32, punch=32. These sit in the
    // same RAM address region as goal position/moving speed (not the
    // EEPROM region), so changes are expected to reset to firmware
    // defaults on power-cycle rather than persisting -- verify directly
    // if that matters. Investigated as a possible fix for a real,
    // load-dependent steady-state error observed after IK-commanded
    // moves (residual ticks vs. commanded target that a repeated
    // goal-position write did not resolve) -- see CLAUDE.md's kinematic-
    // calibration section.
    bool readComplianceMargins(int motorId, uint8_t& cwMargin, uint8_t& ccwMargin);
    bool writeComplianceMargins(int motorId, uint8_t cwMargin, uint8_t ccwMargin);
    bool readComplianceSlopes(int motorId, uint8_t& cwSlope, uint8_t& ccwSlope);
    bool writeComplianceSlopes(int motorId, uint8_t cwSlope, uint8_t ccwSlope);
    bool readPunch(int motorId, uint16_t& punch);
    bool writePunch(int motorId, uint16_t punch);

private:
    const char* deviceName_;
    int baudRate_;
    float protocolVersion_;
    bool connected_;

    dynamixel::PortHandler* portHandler_;
    dynamixel::PacketHandler* packetHandler_;

    bool checkCommResult(int commResult, uint8_t dxlError, int motorId, const char* action);

    bool isPositionWithinLimit(int motorId, uint16_t position) const;

    static const int ADDR_CW_ANGLE_LIMIT = 6;
    static const int ADDR_CCW_ANGLE_LIMIT = 8;
    static const int ADDR_TORQUE_ENABLE = 24;
    static const int ADDR_CW_COMPLIANCE_MARGIN = 26;
    static const int ADDR_CCW_COMPLIANCE_MARGIN = 27;
    static const int ADDR_CW_COMPLIANCE_SLOPE = 28;
    static const int ADDR_CCW_COMPLIANCE_SLOPE = 29;
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
    std::string buildElectricalSnapshot(const std::vector<int>& motorIds);
    std::vector<std::string> jointNames_;
};

#endif