#include <iostream>
#include <array>
#include "dynamixel_motor.h"

struct Pose {
    std::array<double, 7> jointAngles;
};

class CytonArm {
private:
    std::array<DynamixelMotor, 7> motors;

public:
    CytonArm()
        : motors{
            {
                DynamixelMotor(1),
                DynamixelMotor(2),
                DynamixelMotor(3),
                DynamixelMotor(4),
                DynamixelMotor(5),
                DynamixelMotor(6),
                DynamixelMotor(7)
            }
        }
    {
    }

    void moveToPose(const Pose& pose) {
        std::cout << "Moving Cyton arm to pose..." << std::endl;

        for (int i = 0; i < 7; i++) {
            motors[i].moveToAngle(pose.jointAngles[i]);
        }

        std::cout << "Pose complete." << std::endl;
    }

    void printCurrentPose() const {
        std::cout << "\nCurrent joint angles:" << std::endl;

        for (int i = 0; i < 7; i++) {
            std::cout << "Joint " << i + 1
                      << ": " << motors[i].getAngle()
                      << " radians" << std::endl;
        }
    }
};

int main() {
    CytonArm arm;

    Pose home = {{
        0.0,
        -0.5,
        1.0,
        0.0,
        0.5,
        0.0,
        0.0
    }};

    Pose ready = {{
        0.2,
        -0.4,
        0.8,
        0.1,
        0.3,
        -0.2,
        0.0
    }};

    arm.moveToPose(home);
    arm.printCurrentPose();

    std::cout << "\n--- Moving to ready pose ---\n" << std::endl;

    arm.moveToPose(ready);
    arm.printCurrentPose();

    return 0;
}