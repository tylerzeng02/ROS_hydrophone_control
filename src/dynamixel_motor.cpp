#include "dynamixel_motor.h"
#include "robot_calibration.h"

#include <algorithm>
#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
#include <vector>
#include <string>

int getMotorTolerance(int motorId)
{
    if (motorId == 7)
    {
        return 35; // gripper/clamp motor can be less strict
    }

    return 10; // main arm joints stay more accurate
}

// Per-joint backlash overshoot margin, from hand-verified reversal tests
// (see CLAUDE.md). Each value is the joint's measured backlash gap plus a
// modest safety margin -- kept modest since overshooting past the true gap
// only adds settling noise, not correction benefit.
int getBacklashOvershootTicks(int motorId)
{
    static const int overshootPerJoint[7] = {
        3,   // motor 0, shoulder_roll   -- measured ~0.28mm (negligible)
        5,   // motor 1, shoulder_pitch  -- measured ~1.01mm (small)
        10,  // motor 2, shoulder_yaw    -- measured ~3.20mm
        15,  // motor 3, elbow_pitch     -- measured ~5.11mm
        30,  // motor 4, elbow_yaw       -- measured ~7.68mm (largest)
        23,  // motor 5, wrist_pitch     -- measured ~4.02mm
        20,  // motor 6, wrist_roll      -- measured ~2.09mm
    };

    if (motorId >= 0 && motorId < 7)
    {
        return overshootPerJoint[motorId];
    }

    return 20; // fallback for any motor ID outside the 7-joint arm (e.g. a gripper)
}

DynamixelMotor::DynamixelMotor(
    const char* deviceName,
    int baudRate,
    float protocolVersion
)
    : deviceName_(deviceName),
      baudRate_(baudRate),
      protocolVersion_(protocolVersion),
      connected_(false)
{
    portHandler_ = dynamixel::PortHandler::getPortHandler(deviceName_);
    packetHandler_ = dynamixel::PacketHandler::getPacketHandler(protocolVersion_);

    jointNames_ = {
        "joint0",
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7"
    };
}

DynamixelMotor::~DynamixelMotor()
{
    if (connected_)
    {
        disconnect();
    }
}

bool DynamixelMotor::connect()
{
    if (!portHandler_->openPort())
    {
        std::cerr << "Failed to open port: " << deviceName_ << std::endl;
        return false;
    }

    std::cout << "Opened port successfully." << std::endl;

    if (!portHandler_->setBaudRate(baudRate_))
    {
        std::cerr << "Failed to set baud rate: " << baudRate_ << std::endl;
        portHandler_->closePort();
        return false;
    }

    std::cout << "Baud rate set successfully." << std::endl;
    connected_ = true;
    return true;
}

void DynamixelMotor::disconnect()
{
    if (connected_)
    {
        portHandler_->closePort();
        connected_ = false;
        std::cout << "Closed port." << std::endl;
    }
}

std::string DynamixelMotor::getLastElectricalSnapshot() const
{
    return lastElectricalSnapshot_;
}

std::string DynamixelMotor::buildElectricalSnapshot(const std::vector<int>& motorIds)
{
    std::string snapshot;

    for (int id : motorIds)
    {
        uint8_t voltageRaw = 0;
        uint16_t loadRaw = 0;
        uint8_t temperature = 0;

        bool voltageOk = readVoltage(id, voltageRaw);
        bool loadOk = readLoad(id, loadRaw);
        bool tempOk = readTemperature(id, temperature);

        snapshot += "Motor " + std::to_string(id);

        if (voltageOk)
        {
            double voltage = voltageRaw / 10.0;
            snapshot += " | Voltage: " + std::to_string(voltage) + " V";
        }
        else
        {
            snapshot += " | Voltage: READ FAILED";
        }

        if (loadOk)
        {
            int loadMagnitude = loadRaw & 0x03FF;
            double loadPercent = (static_cast<double>(loadMagnitude) / 1023.0) * 100.0;
            snapshot += " | Load: " + std::to_string(loadPercent) + "%";
        }
        else
        {
            snapshot += " | Load: READ FAILED";
        }

        if (tempOk)
        {
            snapshot += " | Temp: " + std::to_string(static_cast<int>(temperature)) + " C";
        }
        else
        {
            snapshot += " | Temp: READ FAILED";
        }

        snapshot += "\n";
    }

    return snapshot;
}

bool DynamixelMotor::checkCommResult(int commResult, uint8_t dxlError, int motorId, const char* action)
{
    if (commResult != COMM_SUCCESS)
    {
        std::cerr << action << " failed for motor ID " << motorId << ": "
                  << packetHandler_->getTxRxResult(commResult)
                  << std::endl;
        return false;
    }

    if (dxlError != 0)
    {
        std::cerr << "Dynamixel error during " << action
                  << " for motor ID " << motorId << ": "
                  << packetHandler_->getRxPacketError(dxlError)
                  << std::endl;
        return false;
    }

    return true;
}

bool DynamixelMotor::isPositionWithinLimit(
    int motorId,
    uint16_t position
) const
{
    if (motorId < 0 ||
        motorId >= static_cast<int>(jointCalibrations.size()))
    {
        std::cerr << "Unknown motor ID: "
                  << motorId
                  << std::endl;

        return false;
    }

    const JointCalibration& joint =
        jointCalibrations[motorId];

    return position >= joint.minTick &&
           position <= joint.maxTick;
}

bool DynamixelMotor::isPositionSafe(int motorId, uint16_t position) const
{
    return isPositionWithinLimit(motorId, position);
}

bool DynamixelMotor::pingMotor(int motorId)
{
    uint8_t dxlError = 0;
    uint16_t modelNumber = 0;

    int commResult = packetHandler_->ping(
        portHandler_,
        motorId,
        &modelNumber,
        &dxlError
    );

    if (!checkCommResult(commResult, dxlError, motorId, "Ping"))
    {
        return false;
    }

    std::cout << "Ping successful. Motor ID " << motorId
              << " model number: " << modelNumber << std::endl;

    return true;
}

bool DynamixelMotor::enableTorque(int motorId)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->write1ByteTxRx(
        portHandler_,
        motorId,
        ADDR_TORQUE_ENABLE,
        TORQUE_ENABLE,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Enable torque");
}

bool DynamixelMotor::disableTorque(int motorId)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->write1ByteTxRx(
        portHandler_,
        motorId,
        ADDR_TORQUE_ENABLE,
        TORQUE_DISABLE,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Disable torque");
}

bool DynamixelMotor::writeGoalPositionRaw(int motorId, uint16_t position)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->write2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_GOAL_POSITION,
        position,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Set goal position");
}

bool DynamixelMotor::readComplianceMargins(int motorId, uint8_t& cwMargin, uint8_t& ccwMargin)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read1ByteTxRx(
        portHandler_, motorId, ADDR_CW_COMPLIANCE_MARGIN, &cwMargin, &dxlError
    );
    if (!checkCommResult(commResult, dxlError, motorId, "Read CW compliance margin"))
    {
        return false;
    }

    commResult = packetHandler_->read1ByteTxRx(
        portHandler_, motorId, ADDR_CCW_COMPLIANCE_MARGIN, &ccwMargin, &dxlError
    );
    return checkCommResult(commResult, dxlError, motorId, "Read CCW compliance margin");
}

bool DynamixelMotor::writeComplianceMargins(int motorId, uint8_t cwMargin, uint8_t ccwMargin)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->write1ByteTxRx(
        portHandler_, motorId, ADDR_CW_COMPLIANCE_MARGIN, cwMargin, &dxlError
    );
    if (!checkCommResult(commResult, dxlError, motorId, "Write CW compliance margin"))
    {
        return false;
    }

    commResult = packetHandler_->write1ByteTxRx(
        portHandler_, motorId, ADDR_CCW_COMPLIANCE_MARGIN, ccwMargin, &dxlError
    );
    return checkCommResult(commResult, dxlError, motorId, "Write CCW compliance margin");
}

bool DynamixelMotor::readComplianceSlopes(int motorId, uint8_t& cwSlope, uint8_t& ccwSlope)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read1ByteTxRx(
        portHandler_, motorId, ADDR_CW_COMPLIANCE_SLOPE, &cwSlope, &dxlError
    );
    if (!checkCommResult(commResult, dxlError, motorId, "Read CW compliance slope"))
    {
        return false;
    }

    commResult = packetHandler_->read1ByteTxRx(
        portHandler_, motorId, ADDR_CCW_COMPLIANCE_SLOPE, &ccwSlope, &dxlError
    );
    return checkCommResult(commResult, dxlError, motorId, "Read CCW compliance slope");
}

bool DynamixelMotor::writeComplianceSlopes(int motorId, uint8_t cwSlope, uint8_t ccwSlope)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->write1ByteTxRx(
        portHandler_, motorId, ADDR_CW_COMPLIANCE_SLOPE, cwSlope, &dxlError
    );
    if (!checkCommResult(commResult, dxlError, motorId, "Write CW compliance slope"))
    {
        return false;
    }

    commResult = packetHandler_->write1ByteTxRx(
        portHandler_, motorId, ADDR_CCW_COMPLIANCE_SLOPE, ccwSlope, &dxlError
    );
    return checkCommResult(commResult, dxlError, motorId, "Write CCW compliance slope");
}

bool DynamixelMotor::readPunch(int motorId, uint16_t& punch)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read2ByteTxRx(
        portHandler_, motorId, ADDR_PUNCH, &punch, &dxlError
    );
    return checkCommResult(commResult, dxlError, motorId, "Read punch");
}

bool DynamixelMotor::writePunch(int motorId, uint16_t punch)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->write2ByteTxRx(
        portHandler_, motorId, ADDR_PUNCH, punch, &dxlError
    );
    return checkCommResult(commResult, dxlError, motorId, "Write punch");
}

bool DynamixelMotor::readModelNumber(int motorId, uint16_t& modelNumber)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read2ByteTxRx(
        portHandler_, motorId, ADDR_MODEL_NUMBER, &modelNumber, &dxlError
    );
    return checkCommResult(commResult, dxlError, motorId, "Read model number");
}

bool DynamixelMotor::readGains(int motorId, uint8_t& dGain, uint8_t& iGain, uint8_t& pGain)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read1ByteTxRx(
        portHandler_, motorId, ADDR_D_GAIN, &dGain, &dxlError
    );
    if (!checkCommResult(commResult, dxlError, motorId, "Read D gain"))
    {
        return false;
    }

    commResult = packetHandler_->read1ByteTxRx(
        portHandler_, motorId, ADDR_I_GAIN, &iGain, &dxlError
    );
    if (!checkCommResult(commResult, dxlError, motorId, "Read I gain"))
    {
        return false;
    }

    commResult = packetHandler_->read1ByteTxRx(
        portHandler_, motorId, ADDR_P_GAIN, &pGain, &dxlError
    );
    return checkCommResult(commResult, dxlError, motorId, "Read P gain");
}

bool DynamixelMotor::writeGains(int motorId, uint8_t dGain, uint8_t iGain, uint8_t pGain)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->write1ByteTxRx(
        portHandler_, motorId, ADDR_D_GAIN, dGain, &dxlError
    );
    if (!checkCommResult(commResult, dxlError, motorId, "Write D gain"))
    {
        return false;
    }

    commResult = packetHandler_->write1ByteTxRx(
        portHandler_, motorId, ADDR_I_GAIN, iGain, &dxlError
    );
    if (!checkCommResult(commResult, dxlError, motorId, "Write I gain"))
    {
        return false;
    }

    commResult = packetHandler_->write1ByteTxRx(
        portHandler_, motorId, ADDR_P_GAIN, pGain, &dxlError
    );
    return checkCommResult(commResult, dxlError, motorId, "Write P gain");
}

bool DynamixelMotor::setGoalPosition(int motorId, uint16_t position)
{
    if (position > MAX_RAW_POSITION)
    {
        std::cerr << "Goal position out of full MX range: " << position
                  << ". Valid full range is 0 to " << MAX_RAW_POSITION << std::endl;
        return false;
    }

    if (!isPositionWithinLimit(motorId, position))
    {
        std::cerr << "Command cancelled. Goal position " << position
                  << " is outside the safe joint range for motor ID "
                  << motorId << std::endl;
        return false;
    }

    return writeGoalPositionRaw(motorId, position);
}

bool DynamixelMotor::setMovingSpeed(int motorId, uint16_t speed)
{
    if (speed > 1023)
    {
        std::cerr << "Moving speed out of range: " << speed
                  << ". Valid range is 0 to 1023." << std::endl;
        return false;
    }

    uint8_t dxlError = 0;

    int commResult = packetHandler_->write2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_MOVING_SPEED,
        speed,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Set moving speed");
}

    bool DynamixelMotor::moveJointSafely(
        int motorId,
        uint16_t targetPosition,
        uint16_t speed,
        int tolerance,
        int timeoutSeconds,
        bool compensateBacklash
    )
    {
    if (!isPositionWithinLimit(motorId, targetPosition))
    {
        std::cerr << "Target position " << targetPosition
                  << " is outside the safe range for motor ID "
                  << motorId << std::endl;
        return false;
    }

    uint16_t startPosition = 0;
    if (!readPosition(motorId, startPosition))
    {
        std::cerr << "Could not read starting position for motor ID "
                  << motorId << std::endl;
        return false;
    }

    if (!isPositionWithinLimit(motorId, startPosition))
    {
        std::cerr << "Starting position " << startPosition
                  << " is outside the safe range for motor ID "
                  << motorId << std::endl;
        return false;
    }

    if (compensateBacklash && targetPosition < startPosition)
    {
        // Would arrive by decreasing -- overshoot below the target first so
        // the real approach always increases instead (see header comment).
        int overshoot = static_cast<int>(targetPosition) -
            getBacklashOvershootTicks(motorId);

        const JointCalibration& joint = jointCalibrations[motorId];
        if (overshoot < joint.minTick)
        {
            overshoot = joint.minTick;
        }

        std::cout << "Backlash compensation: overshooting motor " << motorId
                  << " to " << overshoot << " before approaching "
                  << targetPosition << " from below." << std::endl;

        if (!moveJointSafely(
                motorId,
                static_cast<uint16_t>(overshoot),
                speed,
                tolerance,
                timeoutSeconds,
                /*compensateBacklash=*/false
            ))
        {
            // Not fatal: even a partial overshoot almost certainly still
            // got past the target in the right direction, so proceed to
            // the real move rather than abort.
            std::cerr << "Warning: backlash-compensation overshoot for "
                       << "motor ID " << motorId
                       << " did not fully converge; proceeding to the "
                       << "real move anyway with whatever position was "
                       << "reached." << std::endl;
        }

        // Re-read: the overshoot move above changed the real position,
        // so the earlier startPosition read is now stale.
        if (!readPosition(motorId, startPosition))
        {
            std::cerr << "Could not re-read position for motor ID "
                      << motorId << " after backlash overshoot." << std::endl;
            return false;
        }
    }

    std::cout << "Motor " << motorId
              << " starting position: " << startPosition << std::endl;

    if (!enableTorque(motorId))
    {
        std::cerr << "Failed to enable torque on motor ID "
                  << motorId << std::endl;
        return false;
    }

    if (!setMovingSpeed(motorId, speed))
    {
        std::cerr << "Failed to set moving speed on motor ID "
                  << motorId << std::endl;
        disableTorque(motorId);
        return false;
    }

    if (!setGoalPosition(motorId, targetPosition))
    {
        std::cerr << "Failed to set goal position on motor ID "
                  << motorId << std::endl;
        disableTorque(motorId);
        return false;
    }

    std::cout << "Move command sent. Monitoring motor "
              << motorId << "..." << std::endl;

    auto startTime = std::chrono::steady_clock::now();

    while (true)
    {
        uint16_t currentPosition = 0;

        if (!readPosition(motorId, currentPosition))
        {
            std::cerr << "Failed to read position during movement. Stopping motor."
                      << std::endl;
            disableTorque(motorId);
            return false;
        }

        if (!isPositionWithinLimit(motorId, currentPosition))
        {
            std::cerr << "SAFETY STOP: motor " << motorId
                      << " moved outside safe range at position "
                      << currentPosition << std::endl;

            disableTorque(motorId);
            return false;
        }

        int error = static_cast<int>(targetPosition) - static_cast<int>(currentPosition);

        if (error < 0)
        {
            error = -error;
        }

        std::cout << "Current position: " << currentPosition
                  << " | Error: " << error << std::endl;

        if (error <= tolerance)
        {
            std::cout << "Target reached." << std::endl;
            disableTorque(motorId);
            return true;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsedSeconds =
            std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();

        if (elapsedSeconds >= timeoutSeconds)
        {
            std::cerr << "Timeout: motor did not reach target in time." << std::endl;
            disableTorque(motorId);
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

bool DynamixelMotor::moveJointsSafely(
    const std::vector<int>& motorIds,
    const std::vector<uint16_t>& targetPositions,
    uint16_t speed,
    int tolerance,
    int timeoutSeconds,
    bool holdTorque,
    int stallRepeatsToDetect,
    bool enforceSafetyLimits,
    bool compensateBacklash,
    const std::vector<int>& compensateOnlyJointIds
)
{
    if (motorIds.size() != targetPositions.size())
    {
        std::cerr << "Motor ID count does not match target position count." << std::endl;
        return false;
    }

    if (motorIds.empty())
    {
        std::cerr << "No motors provided." << std::endl;
        return false;
    }

    if (compensateBacklash)
    {
        // Per joint: if this move would arrive by decreasing, overshoot
        // below the target first so the real approach always increases
        // instead (see moveJointSafely()'s header comment). Joints already
        // approaching by increasing (or already at target) are unmodified.

        std::vector<uint16_t> overshootTargets;
        overshootTargets.reserve(motorIds.size());
        bool anyNeedsOvershoot = false;

        for (size_t i = 0; i < motorIds.size(); ++i)
        {
            int motorId = motorIds[i];
            uint16_t targetPosition = targetPositions[i];

            uint16_t currentPosition = 0;
            if (!readPosition(motorId, currentPosition))
            {
                std::cerr << "Could not read starting position for motor ID "
                          << motorId << " during backlash pre-check."
                          << std::endl;
                return false;
            }

            const bool eligibleForCompensation =
                compensateOnlyJointIds.empty() ||
                std::find(
                    compensateOnlyJointIds.begin(),
                    compensateOnlyJointIds.end(),
                    motorId
                ) != compensateOnlyJointIds.end();

            if (eligibleForCompensation && targetPosition < currentPosition)
            {
                anyNeedsOvershoot = true;

                int overshoot = static_cast<int>(targetPosition) -
                    getBacklashOvershootTicks(motorId);

                if (enforceSafetyLimits)
                {
                    const JointCalibration& joint = jointCalibrations[motorId];
                    if (overshoot < joint.minTick)
                    {
                        overshoot = joint.minTick;
                    }
                }
                else if (overshoot < 0)
                {
                    overshoot = 0;
                }

                overshootTargets.push_back(static_cast<uint16_t>(overshoot));
            }
            else
            {
                overshootTargets.push_back(targetPosition);
            }
        }

        if (anyNeedsOvershoot)
        {
            std::cout
                << "Backlash compensation: overshooting below-target "
                << "joints before the real move so every joint's final "
                << "approach is in the same (increasing) direction."
                << std::endl;

            if (!moveJointsSafely(
                    motorIds,
                    overshootTargets,
                    speed,
                    tolerance,
                    timeoutSeconds,
                    /*holdTorque=*/true,
                    stallRepeatsToDetect,
                    enforceSafetyLimits,
                    /*compensateBacklash=*/false
                ))
            {
                // Not fatal -- same reasoning as moveJointSafely() above.
                std::cerr
                    << "Warning: backlash-compensation overshoot did not "
                    << "fully converge for all joints; proceeding to the "
                    << "real move anyway with whatever positions were "
                    << "reached." << std::endl;
            }
        }
    }

    int initialTotalError = 0;

    for (size_t i = 0; i < motorIds.size(); ++i)
    {
        int motorId = motorIds[i];
        uint16_t targetPosition = targetPositions[i];

        if (enforceSafetyLimits && !isPositionWithinLimit(motorId, targetPosition))
        {
            std::cerr << "Target position " << targetPosition
                      << " is outside the safe range for motor ID "
                      << motorId << std::endl;
            return false;
        }

        uint16_t currentPosition = 0;

        if (!readPosition(motorId, currentPosition))
        {
            std::cerr << "Could not read starting position for motor ID "
                      << motorId << std::endl;
            return false;
        }

        if (enforceSafetyLimits && !isPositionWithinLimit(motorId, currentPosition))
        {
            std::cerr << "Starting position " << currentPosition
                      << " is outside the safe range for motor ID "
                      << motorId << std::endl;
            return false;
        }

        int startError = static_cast<int>(targetPosition) - static_cast<int>(currentPosition);

        if (startError < 0)
        {
            startError = -startError;
        }

        initialTotalError += startError;

        std::cout << "Motor " << motorId
                  << " starting position: " << currentPosition << std::endl;
    }

    for (int motorId : motorIds)
    {
        if (!enableTorque(motorId))
        {
            std::cerr << "Failed to enable torque on motor ID "
                      << motorId << std::endl;

            for (int id : motorIds)
            {
                disableTorque(id);
            }

            return false;
        }

        if (!setMovingSpeed(motorId, speed))
        {
            std::cerr << "Failed to set moving speed on motor ID "
                      << motorId << std::endl;

            for (int id : motorIds)
            {
                disableTorque(id);
            }

            return false;
        }
    }

    for (size_t i = 0; i < motorIds.size(); ++i)
    {
        const bool goalAccepted = enforceSafetyLimits
            ? setGoalPosition(motorIds[i], targetPositions[i])
            : writeGoalPositionRaw(motorIds[i], targetPositions[i]);

        if (!goalAccepted)
        {
            std::cerr << "Failed to set goal position for motor ID "
                      << motorIds[i] << std::endl;

            for (int id : motorIds)
            {
                disableTorque(id);
            }

            return false;
        }
    }

    std::cout << "Multi-motor move command sent. Monitoring..." << std::endl;

    auto startTime = std::chrono::steady_clock::now();

    bool midSnapshotCaptured = false;
    lastElectricalSnapshot_.clear();

    constexpr int PRINT_EVERY_N_ITERATIONS = 10;
    int iterationCount = 0;

    int previousTotalError = -1;
    int stallRepeatCount = 0;

    while (true)
    {
        const bool shouldPrint =
            (iterationCount % PRINT_EVERY_N_ITERATIONS) == 0;
        ++iterationCount;

        bool allReached = true;
        int currentTotalError = 0;

        for (size_t i = 0; i < motorIds.size(); ++i)
        {
            int motorId = motorIds[i];
            uint16_t targetPosition = targetPositions[i];

            uint16_t currentPosition = 0;

            if (!readPosition(motorId, currentPosition))
            {
                std::cerr << "Failed to read position for motor ID "
                          << motorId << ". Stopping all motors." << std::endl;

                for (int id : motorIds)
                {
                    disableTorque(id);
                }

                return false;
            }

            if (enforceSafetyLimits && !isPositionWithinLimit(motorId, currentPosition))
            {
                std::cerr << "SAFETY STOP: motor " << motorId
                          << " moved outside safe range at position "
                          << currentPosition << ". Freezing all motors "
                          << "at their current position -- torque stays "
                          << "enabled rather than being released."
                          << std::endl;

                // Hold each motor at its current position (raw writer, not
                // setGoalPosition(), since the offending motor's current
                // position is by definition out of range and would be
                // rejected).
                for (int id : motorIds)
                {
                    uint16_t freezePosition = 0;
                    if (readPosition(id, freezePosition))
                    {
                        writeGoalPositionRaw(id, freezePosition);
                    }
                }

                return false;
            }

            int error = static_cast<int>(targetPosition) - static_cast<int>(currentPosition);

            if (error < 0)
            {
                error = -error;
            }

            currentTotalError += error;

            int motorTolerance = tolerance;

            if (motorId == 7)
            {
                motorTolerance = 35;
            }

            if (shouldPrint)
            {
                std::cout << "Motor " << motorId
                          << " | Current: " << currentPosition
                          << " | Target: " << targetPosition
                          << " | Error: " << error
                          << " | Tolerance: " << motorTolerance
                          << std::endl;
            }

            if (error > motorTolerance)
            {
                allReached = false;
            }
        }

        if (!midSnapshotCaptured && initialTotalError > 0 &&
            currentTotalError <= initialTotalError / 2)
        {
            lastElectricalSnapshot_ = buildElectricalSnapshot(motorIds);
            midSnapshotCaptured = true;
        }

        if (stallRepeatsToDetect > 0 && !allReached)
        {
            if (currentTotalError == previousTotalError)
            {
                ++stallRepeatCount;
            }
            else
            {
                stallRepeatCount = 1;
                previousTotalError = currentTotalError;
            }

            if (stallRepeatCount >= stallRepeatsToDetect)
            {
                std::cerr
                    << "Error values unchanged for " << stallRepeatCount
                    << " consecutive checks -- motors have stopped "
                    << "moving; accepting current position." << std::endl;

                if (!holdTorque)
                {
                    for (int id : motorIds)
                    {
                        disableTorque(id);
                    }
                }
                else
                {
                    std::cout << "Torque remains enabled to hold current position." << std::endl;
                }

                return false;
            }
        }

        if (allReached)
        {
            std::cout << "All target positions reached." << std::endl;

            // Fine-correction pass: the +-tolerance check above accepts
            // "close enough", but re-writing the SAME goal position again
            // often lets a servo's position-control loop nudge closer
            // (a fresh goal write behaves differently than continuing to
            // hold an old one). No-op for joints already within
            // FINE_CORRECTION_TOLERANCE_TICKS.
            constexpr int FINE_CORRECTION_TOLERANCE_TICKS = 3;
            constexpr int FINE_CORRECTION_ATTEMPTS = 5;
            constexpr int FINE_CORRECTION_WAIT_MS = 200;

            for (int attempt = 0; attempt < FINE_CORRECTION_ATTEMPTS; ++attempt)
            {
                bool anyNeedsCorrection = false;

                for (size_t i = 0; i < motorIds.size(); ++i)
                {
                    int motorId = motorIds[i];
                    uint16_t targetPosition = targetPositions[i];

                    uint16_t currentPosition = 0;
                    if (!readPosition(motorId, currentPosition))
                    {
                        continue;
                    }

                    int fineError = static_cast<int>(targetPosition) -
                                    static_cast<int>(currentPosition);
                    if (fineError < 0)
                    {
                        fineError = -fineError;
                    }

                    if (fineError > FINE_CORRECTION_TOLERANCE_TICKS)
                    {
                        anyNeedsCorrection = true;
                        const bool goalAccepted = enforceSafetyLimits
                            ? setGoalPosition(motorId, targetPosition)
                            : writeGoalPositionRaw(motorId, targetPosition);
                        if (!goalAccepted)
                        {
                            std::cerr
                                << "Fine-correction: failed to re-write "
                                << "goal for motor " << motorId << std::endl;
                        }
                    }
                }

                if (!anyNeedsCorrection)
                {
                    break;
                }

                std::this_thread::sleep_for(
                    std::chrono::milliseconds(FINE_CORRECTION_WAIT_MS)
                );
            }

            if (!midSnapshotCaptured)
            {
                lastElectricalSnapshot_ = buildElectricalSnapshot(motorIds);
                midSnapshotCaptured = true;
            }

            if (!holdTorque)
            {
                for (int id : motorIds)
                {
                    disableTorque(id);
                }
            }
            else
            {
                std::cout << "Torque remains enabled to hold position." << std::endl;
            }

            return true;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsedSeconds =
            std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();

        if (elapsedSeconds >= timeoutSeconds)
        {
            std::cerr << "Timeout: motors did not reach target in time." << std::endl;

            if (!holdTorque)
            {
                for (int id : motorIds)
                {
                    disableTorque(id);
                }
            }
            else
            {
                std::cout << "Torque remains enabled to hold current position." << std::endl;
            }

            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

bool DynamixelMotor::moveToPose(
    const std::vector<uint16_t>& targetPositions,
    uint16_t speed,
    int tolerance,
    int timeoutSeconds,
    bool holdTorque
)

{
    std::vector<int> allMotorIds = {0, 1, 2, 3, 4, 5, 6, 7};

    if (targetPositions.size() != allMotorIds.size())
    {
        std::cerr << "Pose must contain exactly 8 target positions." << std::endl;
        std::cerr << "Received " << targetPositions.size()
                  << " positions." << std::endl;
        return false;
    }

    return moveJointsSafely(
        allMotorIds,
        targetPositions,
        speed,
        tolerance,
        timeoutSeconds,
        holdTorque
    );
}

int DynamixelMotor::jointNameToMotorId(const std::string& jointName) const
{
    for (size_t i = 0; i < jointNames_.size(); ++i)
    {
        if (jointNames_[i] == jointName)
        {
            return static_cast<int>(i);
        }
    }

    std::cerr << "Unknown joint name: " << jointName << std::endl;
    return -1;
}

bool DynamixelMotor::moveNamedJointRadians(
    const std::string& jointName,
    double radians,
    uint16_t speed,
    int tolerance,
    int timeoutSeconds
)
{
    int motorId =
        jointNameToMotorId(jointName);

    if (motorId < 0)
    {
        return false;
    }

    if (motorId >=
        static_cast<int>(jointCalibrations.size()))
    {
        std::cerr
            << "Invalid motor ID for named radians move: "
            << motorId
            << std::endl;

        return false;
    }

    const JointCalibration& joint =
        jointCalibrations[motorId];

    int targetTick =
        radiansToTicks(joint, radians);

    if (targetTick < joint.minTick ||
        targetTick > joint.maxTick)
    {
        std::cerr
            << "Named radians command rejected."
            << " Motor ID: " << motorId
            << " | Calculated tick: " << targetTick
            << " | Safe range: "
            << joint.minTick
            << " to "
            << joint.maxTick
            << std::endl;

        return false;
    }

    return moveJointSafely(
        motorId,
        static_cast<uint16_t>(targetTick),
        speed,
        tolerance,
        timeoutSeconds
    );
}

bool DynamixelMotor::moveTrajectory(
    const std::vector<std::vector<uint16_t>>& trajectory,
    uint16_t speed,
    int tolerance,
    int timeoutSeconds,
    bool holdTorque
)
{
    if (trajectory.empty())
    {
        std::cerr << "Trajectory is empty." << std::endl;
        return false;
    }

    for (size_t i = 0; i < trajectory.size(); ++i)
    {
        std::cout << "\nMoving to trajectory point " << i << "..." << std::endl;

        if (!moveToPose(trajectory[i], speed, tolerance, timeoutSeconds, holdTorque))
        {
            std::cerr << "Failed at trajectory point " << i << std::endl;
            return false;
        }
    }

    std::cout << "\nTrajectory complete." << std::endl;
    return true;
}

void DynamixelMotor::emergencyShutdown(const std::vector<int>& motorIds)
{
    std::cerr << "\nEMERGENCY SHUTDOWN: disabling torque on all motors." << std::endl;

    for (int id : motorIds)
    {
        disableTorque(id);
    }

    disconnect();
}

bool DynamixelMotor::readAngleLimits(
    int motorId,
    uint16_t& cwLimit,
    uint16_t& ccwLimit
)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_CW_ANGLE_LIMIT,
        &cwLimit,
        &dxlError
    );

    if (!checkCommResult(commResult, dxlError, motorId, "Read CW angle limit"))
    {
        return false;
    }

    commResult = packetHandler_->read2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_CCW_ANGLE_LIMIT,
        &ccwLimit,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Read CCW angle limit");
}

bool DynamixelMotor::writeAngleLimits(
    int motorId,
    uint16_t cwLimit,
    uint16_t ccwLimit
)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->write2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_CW_ANGLE_LIMIT,
        cwLimit,
        &dxlError
    );

    if (!checkCommResult(commResult, dxlError, motorId, "Write CW angle limit"))
    {
        return false;
    }

    commResult = packetHandler_->write2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_CCW_ANGLE_LIMIT,
        ccwLimit,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Write CCW angle limit");
}

bool DynamixelMotor::readPosition(int motorId, uint16_t& position)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_PRESENT_POSITION,
        &position,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Read position");
}

bool DynamixelMotor::readSpeed(int motorId, uint16_t& speed)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_PRESENT_SPEED,
        &speed,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Read speed");
}

bool DynamixelMotor::readLoad(int motorId, uint16_t& load)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_PRESENT_LOAD,
        &load,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Read load");
}

bool DynamixelMotor::readVoltage(int motorId, uint8_t& voltage)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read1ByteTxRx(
        portHandler_,
        motorId,
        ADDR_PRESENT_VOLTAGE,
        &voltage,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Read voltage");
}

bool DynamixelMotor::readTemperature(int motorId, uint8_t& temperature)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read1ByteTxRx(
        portHandler_,
        motorId,
        ADDR_PRESENT_TEMPERATURE,
        &temperature,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Read temperature");
}

bool DynamixelMotor::printElectricalStatus(int motorId)
{
    uint8_t voltageRaw = 0;
    uint16_t loadRaw = 0;
    uint8_t temperature = 0;

    if (!readVoltage(motorId, voltageRaw))
    {
        std::cerr << "Failed to read voltage for motor " << motorId << std::endl;
        return false;
    }

    if (!readLoad(motorId, loadRaw))
    {
        std::cerr << "Failed to read load for motor " << motorId << std::endl;
        return false;
    }

    if (!readTemperature(motorId, temperature))
    {
        std::cerr << "Failed to read temperature for motor " << motorId << std::endl;
        return false;
    }

    double voltage = voltageRaw / 10.0;

    int loadMagnitude = loadRaw & 0x03FF;
    bool loadDirection = loadRaw & 0x0400;
    double loadPercent = (static_cast<double>(loadMagnitude) / 1023.0) * 100.0;

    std::cout << "Motor " << motorId
              << " | Voltage: " << voltage << " V"
              << " | Load: " << loadPercent << "%"
              << " | Load direction: " << (loadDirection ? "CW" : "CCW")
              << " | Temp: " << static_cast<int>(temperature) << " C"
              << std::endl;

    return true;
}

bool DynamixelMotor::printElectricalStatusForMotors(const std::vector<int>& motorIds)
{
    bool allOk = true;

    for (int id : motorIds)
    {
        if (!printElectricalStatus(id))
        {
            allOk = false;
        }
    }

    return allOk;
}

bool DynamixelMotor::moveJointRadians(
    int motorId,
    double radians
)
{
    if (motorId < 0 ||
        motorId >= static_cast<int>(jointCalibrations.size()))
    {
        std::cerr << "Invalid motor ID for radians move: "
                  << motorId
                  << std::endl;

        return false;
    }

    const JointCalibration& joint =
        jointCalibrations[motorId];

    int targetTick =
        radiansToTicks(joint, radians);

    if (targetTick < joint.minTick ||
        targetTick > joint.maxTick)
    {
        std::cerr
            << "Radians command rejected for motor "
            << motorId
            << ". Calculated tick: "
            << targetTick
            << ". Safe range: "
            << joint.minTick
            << " to "
            << joint.maxTick
            << std::endl;

        return false;
    }

    return setGoalPosition(
        motorId,
        static_cast<uint16_t>(targetTick)
    );
}

bool DynamixelMotor::moveJointRadiansPose(
    const std::vector<double>& jointRadians,
    uint16_t speed,
    int tolerance,
    int timeoutSeconds,
    bool holdTorque
)
{
    if (jointRadians.size() != jointCalibrations.size())
    {
        std::cerr << "Radians pose must contain exactly "
                  << jointCalibrations.size()
                  << " joint angles." << std::endl;

        std::cerr << "Received " << jointRadians.size()
                  << " angles." << std::endl;

        return false;
    }

    std::vector<uint16_t> rawPositions;
    rawPositions.reserve(jointRadians.size());

    for (size_t i = 0; i < jointRadians.size(); ++i)
    {
        const JointCalibration& joint =
            jointCalibrations[i];

        int targetTick =
            radiansToTicks(
                joint,
                jointRadians[i]
            );

        std::cout
            << "Motor " << joint.id
            << " | Angle: " << jointRadians[i]
            << " rad"
            << " | Target tick: " << targetTick
            << " | Safe range: "
            << joint.minTick
            << " to "
            << joint.maxTick
            << std::endl;

        if (targetTick < joint.minTick ||
            targetTick > joint.maxTick)
        {
            std::cerr
                << "Radians pose rejected."
                << " Motor ID: " << joint.id
                << " | Calculated tick: " << targetTick
                << " | Safe range: "
                << joint.minTick
                << " to "
                << joint.maxTick
                << std::endl;

            return false;
        }

        rawPositions.push_back(
            static_cast<uint16_t>(targetTick)
        );
    }

    return moveToPose(
        rawPositions,
        speed,
        tolerance,
        timeoutSeconds,
        holdTorque
    );
}
