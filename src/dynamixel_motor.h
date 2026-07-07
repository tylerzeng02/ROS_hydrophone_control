#ifndef DYNAMIXEL_MOTOR_H
#define DYNAMIXEL_MOTOR_H

class DynamixelMotor {
private:
    int id;
    double currentAngle;

public:
    DynamixelMotor(int motorId);

    void moveToAngle(double radians);
    double getAngle() const;
    int getId() const;
};

#endif