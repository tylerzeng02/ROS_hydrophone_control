#include "dynamixel_motor.h"
#include <iostream>

DynamixelMotor::DynamixelMotor(int motorId) {
    id = motorId;
    currentAngle = 0.0;
}

void DynamixelMotor::moveToAngle(double radians) {
    currentAngle = radians;

    std::cout << "Motor " << id
              << " moved to " << currentAngle
              << " radians" << std::endl;
}

double DynamixelMotor::getAngle() const {
    return currentAngle;
}

int DynamixelMotor::getId() const {
    return id;
}