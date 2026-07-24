#include <iostream>
#include <thread>
#include <chrono>
#include <cstdint>
#include <vector>
#include <cmath>

#include "dynamixel_motor.h"
#include "robot_calibration.h"

int main()
{
    const char* DEVICENAME = "/dev/ttyUSB1";
    const int BAUDRATE = 1000000;
    const float PROTOCOL_VERSION = 1.0;

    DynamixelMotor motor(DEVICENAME, BAUDRATE, PROTOCOL_VERSION);

    if (!motor.connect())
    {
        std::cerr << "Failed to connect to Dynamixel motors." << std::endl;
        return 1;
    }

    // Joint calibration index:
    // 0 = shoulder roll
    // 1 = shoulder pitch
    // 2 = shoulder yaw
    // 3 = elbow pitch
    // 4 = elbow yaw
    // 5 = wrist pitch
    // 6 = wrist roll
    const int jointIndex = 5;

    if (jointIndex < 0 ||
        jointIndex >= static_cast<int>(jointCalibrations.size()))
    {
        std::cerr << "Invalid joint index: " << jointIndex << std::endl;
        motor.disconnect();
        return 1;
    }

    const JointCalibration& testJoint = jointCalibrations[jointIndex];
    const int testMotorId = testJoint.id;

    const double relativeMoveRadians = 0.05;
    const uint16_t speed = 40;
    const int tolerance = 7;
    const int returnTolerance = 15;
    const int timeoutSeconds = 8;

    std::vector<int> allMotorIds;

    for (const JointCalibration& joint : jointCalibrations)
    {
        allMotorIds.push_back(joint.id);
    }

    std::cout << "\n===== RELATIVE RADIANS DIRECTION TEST ====="
              << std::endl;
    std::cout << "Joint index: " << jointIndex << std::endl;
    std::cout << "Motor ID: " << testMotorId << std::endl;
    std::cout << "Relative movement: +"
              << relativeMoveRadians << " rad" << std::endl;

    std::cout << "\nMake sure the arm and surrounding area are clear."
              << std::endl;

    // ------------------------------------------------------------
    // Phase 1: Enable torque at the arm's current pose
    // ------------------------------------------------------------

    std::cout << "\nPress ENTER to enable torque for all motors "
              << "at their current positions...";
    std::cin.get();

    /*
     * Read and hold every motor's current position before enabling
     * torque. This prevents a stale goal-position register from
     * causing a motor to jump when torque is enabled.
     */
    std::vector<uint16_t> startingTicks;
    startingTicks.reserve(allMotorIds.size());

    for (int motorId : allMotorIds)
    {
        uint16_t currentTick = 0;

        if (!motor.readPosition(motorId, currentTick))
        {
            std::cerr << "Failed to read motor "
                      << motorId << " before enabling torque."
                      << std::endl;

            motor.disconnect();
            return 1;
        }

        if (!motor.isPositionSafe(motorId, currentTick))
        {
            std::cerr << "Current position " << currentTick
                      << " is outside the configured safe range for motor "
                      << motorId << "." << std::endl;

            motor.disconnect();
            return 1;
        }

        startingTicks.push_back(currentTick);
    }

    /*
     * Set each goal position to its present position before torque is
     * enabled. This is important because enabling torque alone may make
     * a Dynamixel move toward an old goal-position value.
     */
    for (std::size_t i = 0; i < allMotorIds.size(); ++i)
    {
        if (!motor.setGoalPosition(allMotorIds[i], startingTicks[i]))
        {
            std::cerr << "Failed to set the holding position for motor "
                      << allMotorIds[i] << "." << std::endl;

            motor.disconnect();
            return 1;
        }
    }

    std::vector<int> enabledMotorIds;

    for (int motorId : allMotorIds)
    {
        if (!motor.enableTorque(motorId))
        {
            std::cerr << "Failed to enable torque for motor "
                      << motorId << "." << std::endl;

            for (int enabledId : enabledMotorIds)
            {
                motor.disableTorque(enabledId);
            }

            motor.disconnect();
            return 1;
        }

        enabledMotorIds.push_back(motorId);

        std::cout << "Torque enabled for motor "
                  << motorId << "." << std::endl;
    }

    std::cout << "All motors are holding their current positions."
              << std::endl;

    // Find the selected motor's captured starting tick.
    uint16_t testStartTick = 0;
    bool foundTestMotor = false;

    for (std::size_t i = 0; i < allMotorIds.size(); ++i)
    {
        if (allMotorIds[i] == testMotorId)
        {
            testStartTick = startingTicks[i];
            foundTestMotor = true;
            break;
        }
    }

    if (!foundTestMotor)
    {
        std::cerr << "Selected motor is missing from the motor list."
                  << std::endl;

        for (int motorId : enabledMotorIds)
        {
            motor.disableTorque(motorId);
        }

        motor.disconnect();
        return 1;
    }

    const double startRadians =
        ticksToRadians(testJoint, testStartTick);

    const double targetRadians =
        startRadians + relativeMoveRadians;

    const int calculatedTargetTick =
        radiansToTicks(testJoint, targetRadians);

    if (calculatedTargetTick < 0 || calculatedTargetTick > 4095)
    {
        std::cerr << "Calculated target tick is invalid: "
                  << calculatedTargetTick << std::endl;

        for (int motorId : enabledMotorIds)
        {
            motor.disableTorque(motorId);
        }

        motor.disconnect();
        return 1;
    }

    const uint16_t targetTick =
        static_cast<uint16_t>(calculatedTargetTick);

    if (!motor.isPositionSafe(testMotorId, targetTick))
    {
        std::cerr << "Relative target tick " << targetTick
                  << " is outside the safe range for motor "
                  << testMotorId << "." << std::endl;

        for (int motorId : enabledMotorIds)
        {
            motor.disableTorque(motorId);
        }

        motor.disconnect();
        return 1;
    }

    std::cout << "\nCaptured starting position:" << std::endl;
    std::cout << "Starting tick: " << testStartTick << std::endl;
    std::cout << "Starting calibrated radians: "
              << startRadians << std::endl;
    std::cout << "Target calibrated radians: "
              << targetRadians << std::endl;
    std::cout << "Target tick: " << targetTick << std::endl;

    // ------------------------------------------------------------
    // Phase 2: Move +0.05 rad from the captured starting position
    // ------------------------------------------------------------

    std::cout << "\nPress ENTER to move motor "
              << testMotorId << " by +"
              << relativeMoveRadians
              << " rad from its current position...";
    std::cin.get();

    if (!motor.moveJointsSafely(
            std::vector<int>{testMotorId},
            std::vector<uint16_t>{targetTick},
            speed,
            tolerance,
            timeoutSeconds,
            true))
    {
        std::cerr << "Failed to move the selected joint."
                  << std::endl;

        for (int motorId : enabledMotorIds)
        {
            motor.disableTorque(motorId);
        }

        motor.disconnect();
        return 1;
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));

    uint16_t testReadback = 0;

    if (motor.readPosition(testMotorId, testReadback))
    {
        const double readbackRadians =
            ticksToRadians(testJoint, testReadback);

        const double actualRelativeMove =
            readbackRadians - startRadians;

        std::cout << "\n===== TEST RESULT =====" << std::endl;
        std::cout << "Commanded relative movement: +"
                  << relativeMoveRadians << " rad" << std::endl;
        std::cout << "Starting tick: "
                  << testStartTick << std::endl;
        std::cout << "Target tick: "
                  << targetTick << std::endl;
        std::cout << "Actual tick: "
                  << testReadback << std::endl;
        std::cout << "Actual relative movement: "
                  << actualRelativeMove << " rad" << std::endl;
        std::cout << "Tick error: "
                  << static_cast<int>(targetTick) -
                         static_cast<int>(testReadback)
                  << std::endl;
        std::cout << "Relative radian error: "
                  << relativeMoveRadians - actualRelativeMove
                  << std::endl;
    }
    else
    {
        std::cerr << "Warning: failed to read the selected joint "
                  << "after the test movement." << std::endl;
    }

    // ------------------------------------------------------------
    // Phase 3: Return only the tested joint to its captured position
    // ------------------------------------------------------------

    std::cout << "\nPress ENTER to return motor "
              << testMotorId
              << " to the position captured when torque was enabled...";
    std::cin.get();

    if (!motor.moveJointsSafely(
            std::vector<int>{testMotorId},
            std::vector<uint16_t>{testStartTick},
            speed,
            returnTolerance,
            timeoutSeconds,
            true))
    {
        std::cerr << "Failed to return the selected joint."
                  << std::endl;

        for (int motorId : enabledMotorIds)
        {
            motor.disableTorque(motorId);
        }

        motor.disconnect();
        return 1;
    }

    std::cout << "The selected joint returned to its captured "
              << "starting position." << std::endl;

    // ------------------------------------------------------------
    // Phase 4: Disable torque for all motors
    // ------------------------------------------------------------

    std::cout << "\nWarning: the arm may fall or move under gravity "
              << "when torque is disabled." << std::endl;
    std::cout << "Support the arm if necessary, then press ENTER "
              << "to disable torque for all motors...";
    std::cin.get();

    bool allTorqueDisabled = true;

    for (int motorId : allMotorIds)
    {
        if (!motor.disableTorque(motorId))
        {
            std::cerr << "Failed to disable torque for motor "
                      << motorId << "." << std::endl;

            allTorqueDisabled = false;
        }
        else
        {
            std::cout << "Torque disabled for motor "
                      << motorId << "." << std::endl;
        }
    }

    motor.disconnect();

    if (!allTorqueDisabled)
    {
        std::cerr << "Test finished, but torque was not disabled "
                  << "for every motor." << std::endl;
        return 1;
    }

    std::cout << "\nRelative radians test complete." << std::endl;
    return 0;
}