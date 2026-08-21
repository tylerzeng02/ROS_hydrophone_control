#ifndef DYNAMIXEL_MOTOR_H
#define DYNAMIXEL_MOTOR_H

#include <cstdint>
#include <vector>
#include <string>
#include "dynamixel_sdk/dynamixel_sdk.h"

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
    bool moveJointSafely(
        int motorId,
        uint16_t targetPosition,
        uint16_t speed,
        int tolerance = 15,
        int timeoutSeconds = 8,
        bool compensateBacklash = true
    );

    // compensateBacklash: see moveJointSafely() -- applied per-joint here.
    // compensateOnlyJointIds: empty (default) compensates every joint in
    // motorIds; non-empty restricts compensation to just those joint IDs
    // (used to isolate one joint's backlash contribution experimentally).
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
    bool readAngleLimits(int motorId, uint16_t& cwLimit, uint16_t& ccwLimit);
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

    bool writeGoalPositionRaw(int motorId, uint16_t position);
    bool readComplianceMargins(int motorId, uint8_t& cwMargin, uint8_t& ccwMargin);
    bool writeComplianceMargins(int motorId, uint8_t cwMargin, uint8_t ccwMargin);
    bool readComplianceSlopes(int motorId, uint8_t& cwSlope, uint8_t& ccwSlope);
    bool writeComplianceSlopes(int motorId, uint8_t cwSlope, uint8_t ccwSlope);
    bool readPunch(int motorId, uint16_t& punch);
    bool writePunch(int motorId, uint16_t punch);
    bool readModelNumber(int motorId, uint16_t& modelNumber);
    bool readGains(int motorId, uint8_t& dGain, uint8_t& iGain, uint8_t& pGain);
    bool writeGains(int motorId, uint8_t dGain, uint8_t iGain, uint8_t pGain);

private:
    const char* deviceName_;
    int baudRate_;
    float protocolVersion_;
    bool connected_;

    dynamixel::PortHandler* portHandler_;
    dynamixel::PacketHandler* packetHandler_;

    bool checkCommResult(int commResult, uint8_t dxlError, int motorId, const char* action);

    bool isPositionWithinLimit(int motorId, uint16_t position) const;

    static const int ADDR_MODEL_NUMBER = 0;
    static const int ADDR_CW_ANGLE_LIMIT = 6;
    static const int ADDR_CCW_ANGLE_LIMIT = 8;
    static const int ADDR_TORQUE_ENABLE = 24;
    static const int ADDR_CW_COMPLIANCE_MARGIN = 26;
    static const int ADDR_CCW_COMPLIANCE_MARGIN = 27;
    static const int ADDR_CW_COMPLIANCE_SLOPE = 28;
    static const int ADDR_CCW_COMPLIANCE_SLOPE = 29;
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
    std::string buildElectricalSnapshot(const std::vector<int>& motorIds);
    std::vector<std::string> jointNames_;
};

#endif