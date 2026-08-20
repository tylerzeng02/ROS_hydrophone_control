#pragma once

#include <vector>
#include <cstdint>

struct JointCalibration
{
    int id;
    int zeroTick;
    int direction;
    int minTick;
    int maxTick;
    // Gear-ratio scale correction from the kinematic calibration fit.
    // 1.0 = uncorrected. See radiansToTicks()/ticksToRadians().
    double scale = 1.0;
};

int radiansToTicks(const JointCalibration& joint, double radians);
double ticksToRadians(const JointCalibration& joint, int tick);

extern std::vector<JointCalibration> jointCalibrations;