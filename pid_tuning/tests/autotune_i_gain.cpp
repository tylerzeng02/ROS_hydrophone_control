// autotune_i_gain: automatic I-Gain sweep for one joint, evaluated across
// a small set of REAL, previously-measured target points instead of one
// hand-picked spot -- picks the N rows where the tuned joint's achieved
// tick was already furthest from commanded (the documented worst cases),
// not arbitrary ones.
//
// Default joint is motor 1 (shoulder_pitch): an MX-64 at the base carrying
// the entire rest of the arm downstream, so it's under real gravity load
// almost everywhere in its range, unlike e.g. elbow_pitch whose load
// varies a lot with position.
//
// SAFETY DESIGN (runs an unattended multi-point, multi-candidate loop):
//   - Fixed candidate list (0, 4, 8, 12, 16, 20, 24, 30), I=0 evaluated
//     first as the reference.
//   - A point that fails to settle is treated as an instability signal --
//     the sweep stops immediately, that candidate is not counted usable.
//   - Early stopping: 2 candidates in a row worse than the best-so-far
//     stops the sweep rather than escalating past the optimum.
//   - Original D/I/P gains are restored at the end regardless of outcome
//     -- "tunes and reports" means exactly that, nothing is left applied.
//
// Still watch and listen while this runs -- settle-timeout is a backup,
// not a substitute for noticing something wrong. Ctrl+C is always safe.
//
// Small-N diagnostic, not a full validation -- confirm any winning value
// against the full validation point set and a real NDI before/after.
//
// Usage: autotune_i_gain [motorId] [pointsCsv] [numPoints]
//   motorId:   default 1 (shoulder_pitch).
//   pointsCsv: a CSV with tick_0..tick_6/achieved_tick_0..tick_6 columns,
//              used only to select which points to sweep against (the
//              per-candidate errors are always freshly measured live).
//   numPoints: how many worst-for-this-joint points to use (default 5).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "dynamixel_motor.h"
#include "robot_calibration.h"

namespace {

constexpr const char* DEVICE = "/dev/ttyUSB0";
constexpr int BAUD_RATE = 1000000;
constexpr float PROTOCOL_VERSION = 1.0F;
constexpr uint16_t MOVING_SPEED = 40;

constexpr int POLL_INTERVAL_MS = 100;
constexpr int SETTLE_TIMEOUT_MS = 5000;
constexpr int STABLE_TICKS = 1;
constexpr int STABLE_POLLS_REQUIRED = 5;

// Capped, conservative sweep -- see file header comment.
const std::vector<int> CANDIDATE_I_GAINS = {0, 4, 8, 12, 16, 20, 24, 30};

struct TargetPoint {
    uint16_t ticks[7];
};

const JointCalibration* findCalibration(int motorId) {
    for (const auto& cal : jointCalibrations) {
        if (cal.id == motorId) {
            return &cal;
        }
    }
    return nullptr;
}

// Loads tick_0..tick_6 / achieved_tick_0..tick_6 from a CSV (the format
// build/validation_results.csv already has), computes |achieved-commanded|
// for `motorId` per row, and returns the `numPoints` rows with the
// largest such error, largest first.
std::vector<TargetPoint> loadWorstPoints(const std::string& path, int motorId, int numPoints) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open points CSV: " + path);
    }

    std::string headerLine;
    if (!std::getline(file, headerLine)) {
        throw std::runtime_error("Points CSV is empty: " + path);
    }
    std::vector<std::string> columns;
    {
        std::stringstream ss(headerLine);
        std::string field;
        while (std::getline(ss, field, ',')) {
            columns.push_back(field);
        }
    }

    std::vector<int> tickCol(7, -1), achievedCol(7, -1);
    for (std::size_t i = 0; i < columns.size(); ++i) {
        for (int j = 0; j < 7; ++j) {
            if (columns[i] == "tick_" + std::to_string(j)) tickCol[j] = static_cast<int>(i);
            if (columns[i] == "achieved_tick_" + std::to_string(j)) achievedCol[j] = static_cast<int>(i);
        }
    }
    for (int j = 0; j < 7; ++j) {
        if (tickCol[j] < 0 || achievedCol[j] < 0) {
            throw std::runtime_error(
                "Points CSV " + path + " is missing tick_" + std::to_string(j) + "/achieved_tick_" +
                std::to_string(j) + " columns."
            );
        }
    }

    struct ScoredPoint {
        TargetPoint point;
        int errorForTunedJoint;
    };
    std::vector<ScoredPoint> scored;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) {
            fields.push_back(field);
        }
        if (fields.size() < columns.size()) continue;

        TargetPoint point{};
        for (int j = 0; j < 7; ++j) {
            point.ticks[j] = static_cast<uint16_t>(std::stoi(fields[static_cast<std::size_t>(tickCol[j])]));
        }
        const int achieved = std::stoi(fields[static_cast<std::size_t>(achievedCol[motorId])]);
        const int error = std::abs(achieved - static_cast<int>(point.ticks[motorId]));
        scored.push_back({point, error});
    }

    std::sort(scored.begin(), scored.end(), [](const ScoredPoint& a, const ScoredPoint& b) {
        return a.errorForTunedJoint > b.errorForTunedJoint;
    });

    std::vector<TargetPoint> result;
    for (int i = 0; i < numPoints && i < static_cast<int>(scored.size()); ++i) {
        result.push_back(scored[static_cast<std::size_t>(i)].point);
    }
    return result;
}

// Same stability-based settle detection as test_i_gain.cpp, but silent
// unless something is wrong -- this tool runs many more polls in total
// (up to 8 candidates x 5 points), so a full per-sample trace for every
// one would be too much output to usefully watch. Prints only entry,
// timeout, and final results.
struct SettleResult {
    uint16_t finalTick;
    bool settled;
};

SettleResult pollUntilSettled(DynamixelMotor& motor, int motorId, uint16_t target) {
    int elapsedMs = 0;
    int stableCount = 0;
    uint16_t lastPosition = 0;
    bool havePrevious = false;

    while (elapsedMs < SETTLE_TIMEOUT_MS) {
        uint16_t position = 0;
        if (!motor.readPosition(motorId, position)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
            elapsedMs += POLL_INTERVAL_MS;
            continue;
        }

        if (havePrevious &&
            std::abs(static_cast<int>(position) - static_cast<int>(lastPosition)) <= STABLE_TICKS) {
            ++stableCount;
            if (stableCount >= STABLE_POLLS_REQUIRED) {
                return {position, true};
            }
        } else {
            stableCount = 0;
        }

        lastPosition = position;
        havePrevious = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
        elapsedMs += POLL_INTERVAL_MS;
    }

    return {lastPosition, false};
}

}  // namespace

int main(int argc, char** argv) {
    const int motorId = argc > 1 ? std::stoi(argv[1]) : 1;
    const std::string pointsCsv = argc > 2 ? argv[2] : "validation_results.csv";
    const int numPoints = argc > 3 ? std::stoi(argv[3]) : 5;

    const JointCalibration* calibration = findCalibration(motorId);
    if (calibration == nullptr) {
        std::cerr << "No jointCalibrations entry for motor " << motorId << ".\n";
        return 1;
    }

    std::vector<TargetPoint> points;
    try {
        points = loadWorstPoints(pointsCsv, motorId, numPoints);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load points: " << e.what() << '\n';
        return 1;
    }
    if (points.empty()) {
        std::cerr << "No usable points loaded from " << pointsCsv << ".\n";
        return 1;
    }

    std::cout << "=== I-Gain autotune: motor " << motorId << " ===\n"
              << "Using " << points.size() << " real point(s) from " << pointsCsv
              << " (selected as the worst-observed for this joint).\n"
              << "Candidate I values to sweep: ";
    for (int c : CANDIDATE_I_GAINS) std::cout << c << ' ';
    std::cout << "\n\n"
              << "WATCH AND LISTEN for the whole run. This sweep can move through several "
                 "candidates unattended between prints -- a settle-timeout is treated as an "
                 "automatic abort signal, but that is a backup, not a substitute for you noticing "
                 "something wrong. Ctrl+C is always safe.\n\n"
              << "Press Enter to begin: ";
    std::cin.get();

    DynamixelMotor motor(DEVICE, BAUD_RATE, PROTOCOL_VERSION);
    if (!motor.connect()) {
        std::cerr << "Could not connect to " << DEVICE << ".\n";
        return 1;
    }

    std::cout << "Enabling torque on all 7 joints...\n";
    for (int id = 0; id < 7; ++id) {
        if (!motor.pingMotor(id) || !motor.enableTorque(id)) {
            std::cerr << "Could not ping/enable torque on motor " << id << ".\n";
            motor.disconnect();
            return 1;
        }
    }

    uint8_t originalD = 0, originalI = 0, originalP = 0;
    if (!motor.readGains(motorId, originalD, originalI, originalP)) {
        std::cerr << "Could not read baseline gains -- aborting before touching anything.\n";
        motor.disconnect();
        return 1;
    }
    std::cout << "Baseline gains on motor " << motorId << ": D=" << static_cast<int>(originalD)
              << " I=" << static_cast<int>(originalI) << " P=" << static_cast<int>(originalP) << "\n\n";

    struct CandidateResult {
        int iGain;
        double meanAbsError;
        int maxAbsError;
        bool safe;
    };
    std::vector<CandidateResult> results;

    int bestIGain = static_cast<int>(originalI);
    double bestScore = 1e18;
    int worseStreak = 0;
    bool abortedForSafety = false;

    for (int candidate : CANDIDATE_I_GAINS) {
        std::cout << "--- Candidate I=" << candidate << " ---\n";
        if (!motor.writeGains(motorId, originalD, static_cast<uint8_t>(candidate), originalP)) {
            std::cerr << "FAILED to write I=" << candidate << " -- stopping sweep.\n";
            break;
        }

        std::vector<int> pointErrors;
        bool candidateSafe = true;

        for (std::size_t p = 0; p < points.size(); ++p) {
            std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};
            std::vector<uint16_t> targets(points[p].ticks, points[p].ticks + 7);

            // holdTorque=true is REQUIRED -- the default (false) disables
            // torque on all 7 joints after every move, which would drop
            // the whole arm between every point in this loop.
            // tolerance/timeoutSeconds are passed explicitly only to reach
            // holdTorque positionally (no named parameters in C++).
            motor.moveJointsSafely(motorIds, targets, MOVING_SPEED, 15, 10, /*holdTorque=*/true);
            std::this_thread::sleep_for(std::chrono::milliseconds(300));

            const SettleResult settle = pollUntilSettled(motor, motorId, points[p].ticks[motorId]);
            const int error = static_cast<int>(settle.finalTick) - static_cast<int>(points[p].ticks[motorId]);

            std::cout << "  point " << p << ": target=" << points[p].ticks[motorId]
                      << " final=" << settle.finalTick << " error=" << error
                      << (settle.settled ? "" : "  *** DID NOT SETTLE ***") << '\n';

            if (!settle.settled) {
                candidateSafe = false;
                break;
            }
            pointErrors.push_back(std::abs(error));
        }

        if (!candidateSafe) {
            std::cerr << "Candidate I=" << candidate
                      << " failed to settle on at least one point -- treating as UNSAFE, stopping "
                         "the sweep here (not trying any larger candidate).\n";
            results.push_back({candidate, -1.0, -1, false});
            abortedForSafety = true;
            break;
        }

        const double meanAbsError =
            static_cast<double>(std::accumulate(pointErrors.begin(), pointErrors.end(), 0)) /
            static_cast<double>(pointErrors.size());
        const int maxAbsError = *std::max_element(pointErrors.begin(), pointErrors.end());
        std::cout << "  -> mean|error|=" << meanAbsError << "  max|error|=" << maxAbsError << "\n\n";
        results.push_back({candidate, meanAbsError, maxAbsError, true});

        if (meanAbsError < bestScore) {
            bestScore = meanAbsError;
            bestIGain = candidate;
            worseStreak = 0;
        } else {
            ++worseStreak;
            if (worseStreak >= 2) {
                std::cout << "Two candidates in a row did not improve on the best result -- "
                             "stopping sweep early (assuming the optimum has been passed).\n";
                break;
            }
        }
    }

    std::cout << "\n--- Restoring original gains (D=" << static_cast<int>(originalD)
              << " I=" << static_cast<int>(originalI) << " P=" << static_cast<int>(originalP)
              << ") ---\n";
    if (!motor.writeGains(motorId, originalD, originalI, originalP)) {
        std::cerr << "WARNING: FAILED to restore original gains on motor " << motorId
                  << " -- verify manually before trusting this motor's behavior in anything else.\n";
    } else {
        uint8_t rD = 0, rI = 0, rP = 0;
        motor.readGains(motorId, rD, rI, rP);
        std::cout << "Restored: D=" << static_cast<int>(rD) << " I=" << static_cast<int>(rI)
                  << " P=" << static_cast<int>(rP)
                  << ((rD == originalD && rI == originalI && rP == originalP)
                          ? " (confirmed matches original)\n"
                          : " (WARNING: does NOT match original)\n");
    }

    std::cout << "\n=== Summary (motor " << motorId << ", " << points.size() << " point(s) from "
              << pointsCsv << ") ===\n"
              << "I_gain  mean|error|  max|error|  safe\n";
    for (const auto& r : results) {
        std::cout << r.iGain << "       ";
        if (r.safe) {
            std::cout << r.meanAbsError << "         " << r.maxAbsError << "          yes\n";
        } else {
            std::cout << "--            --           NO (did not settle)\n";
        }
    }

    std::cout << "\nBest I Gain found: " << bestIGain << " (mean|error|=" << bestScore << " ticks)\n";
    if (abortedForSafety) {
        std::cout << "Sweep stopped early due to a settling failure -- the best value above is "
                     "only confirmed safe up to that point, nothing larger was tried.\n";
    }
    std::cout << "Original gains have been restored -- this value is NOT currently applied "
                 "anywhere. Validate against the full point set (and ideally a real NDI "
                 "before/after) before adopting it.\n";

    motor.disconnect();
    return 0;
}
