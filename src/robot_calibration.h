#pragma once

#include <vector>
#include <cstdint>

/**
 * @brief Physical calibration for one Dynamixel joint: its zero point,
 * direction, safe tick range, and fitted gear-ratio scale correction.
 *
 * This struct, together with the jointCalibrations table below, is the
 * single source of truth for per-joint physical limits and zero points.
 * Every motor-facing program in this project goes through
 * radiansToTicks()/ticksToRadians() for safety clamping and unit
 * conversion; joint limits are not duplicated per caller.
 */
struct JointCalibration
{
    int id;
    int zeroTick;
    int direction;
    int minTick;
    int maxTick;
    /**
     * @brief Gear-ratio scale correction from the kinematic calibration
     * fit. 1.0 means uncorrected. See radiansToTicks()/ticksToRadians().
     */
    double scale = 1.0;
};

/**
 * @brief Converts a joint angle to a raw Dynamixel tick, applying this
 * joint's zero point, direction, and gear-ratio scale correction.
 * @param joint Calibration for the joint being converted.
 * @param radians Joint angle in radians.
 * @return The corresponding raw tick value, rounded to the nearest
 *         integer. Not clamped to `joint.minTick`/`maxTick`; callers are
 *         responsible for range-checking the result before writing it to
 *         a servo (see src/dynamixel_motor.h).
 */
int radiansToTicks(const JointCalibration& joint, double radians);

/**
 * @brief Converts a raw Dynamixel tick to a joint angle, applying this
 * joint's zero point, direction, and gear-ratio scale correction. Inverse
 * of radiansToTicks().
 * @param joint Calibration for the joint being converted.
 * @param tick Raw tick value.
 * @return The corresponding joint angle in radians.
 */
double ticksToRadians(const JointCalibration& joint, int tick);

/**
 * @brief Per-joint calibration table, one entry per motor ID 0-6, in
 * motor ID order. See robot_calibration.cpp for the fitted values and the
 * reasoning behind which joints do and don't get the fitted offset
 * applied.
 */
extern std::vector<JointCalibration> jointCalibrations;
