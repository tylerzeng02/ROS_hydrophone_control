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
};

int radiansToTicks(const JointCalibration& joint, double radians);
double ticksToRadians(const JointCalibration& joint, int tick);

extern std::vector<JointCalibration> jointCalibrations;