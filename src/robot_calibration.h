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
    // Gear-ratio scale correction (2026-08-06, from the kinematic
    // calibration fit -- see CLAUDE.md's kinematic-calibration section).
    // 1.0 = uncorrected (4096 ticks/revolution assumed exact). Multiplies
    // the raw tick-derived angle before the zero-offset is applied; see
    // radiansToTicks()/ticksToRadians() below for the exact formula.
    double scale = 1.0;
};

int radiansToTicks(const JointCalibration& joint, double radians);
double ticksToRadians(const JointCalibration& joint, int tick);

extern std::vector<JointCalibration> jointCalibrations;