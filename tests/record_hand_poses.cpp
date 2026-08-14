#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "dynamixel_motor.h"
#include "robot_calibration.h"

namespace {

constexpr const char* CYTON_DEVICE = "COM4";
constexpr int CYTON_BAUD_RATE = 1000000;
constexpr float CYTON_PROTOCOL_VERSION = 1.0F;

constexpr std::size_t JOINT_COUNT = 7;

// Multi-joint backlash test (2026-07-30): generalizes the original single-
// joint version (which confirmed real backlash at wrist_pitch and
// elbow_pitch individually) to walk through ALL joints in TEST_JOINT_IDS,
// one after another, in a single sitting -- so a per-joint backlash gap
// can be measured for every joint, not just the two already tested.
//
// For each joint in turn, records exactly POSES_PER_JOINT (4) poses while
// every OTHER joint stays locked (torqued) at a fixed position:
//   1. test joint well BELOW a chosen target
//   2. test joint AT the target, arriving from below (continuing #1's direction)
//   3. test joint well ABOVE the same target
//   4. test joint AT the exact same target again, but auto-driven back to
//      pose 2's exact recorded tick, arriving from above
// Comparing the NDI-measured position of pose 2 vs pose 4 (via
// test_five_pose_ndi_capture.cpp, same as the original single-joint
// workflow) isolates that joint's own backlash: a gap larger than the
// tracker's own repeatability floor means real backlash for that joint.
// See CLAUDE.md's kinematic-calibration section for the wrist_pitch/
// elbow_pitch results this generalizes.
//
// Only the joint currently under test is ever free (torque off); every
// other joint -- including ones already tested earlier in this same
// session -- is locked (torqued, held wherever it happens to be) so it
// can't drift while a different joint is being hand-moved. Order matters
// only in that TEST_JOINT_IDS is processed front-to-back; feel free to
// reorder or trim this list to re-test a subset instead of all 7.
constexpr std::array<int, 7> TEST_JOINT_IDS = {0, 1, 2, 3, 4, 5, 6};
constexpr int POSES_PER_JOINT = 4;

// Full raw tick range for these servos (4096 ticks/revolution). Widening
// each motor's hardware CW/CCW angle-limit registers to this for the
// duration of this program only lets the servo ACCEPT holding wherever a
// human already physically placed it by hand -- it never causes the arm to
// move anywhere it hasn't already been placed, so it introduces no new
// mechanical risk specific to this tool.
constexpr uint16_t FULL_RANGE_CW_LIMIT = 0;
constexpr uint16_t FULL_RANGE_CCW_LIMIT = 4095;

constexpr const char* OUTPUT_CSV = "recorded_hand_poses_PID.csv",
constexpr const char* OUTPUT_CPP_SNIPPET = "recorded_hand_poses_target_poses.txt";

struct RecordedPose {
    int testJointId;
    int poseIndexInTest;  // 1-based, 1..POSES_PER_JOINT
    std::array<uint16_t, JOINT_COUNT> ticks;
};

void waitForEnter(const std::string& message) {
    std::cout << message << std::flush;
    std::string line;
    std::getline(std::cin, line);
}

// Loads any poses already recorded by a previous run of this program, so a
// new session continues the same batch instead of starting over. Silently
// does nothing if the file doesn't exist yet (first run).
void loadExistingPoses(
    const std::string& path,
    std::vector<RecordedPose>& poses
) {
    std::ifstream in(path);
    if (!in) {
        return;
    }

    std::string line;
    std::getline(in, line);  // header row, discarded

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream fields(line);
        std::string field;
        std::getline(fields, field, ',');  // pose_id, discarded (renumbered on write)

        RecordedPose pose{};
        bool valid = true;

        if (!std::getline(fields, field, ',')) { valid = false; }
        else { pose.testJointId = std::stoi(field); }

        if (valid) {
            if (!std::getline(fields, field, ',')) { valid = false; }
            else { pose.poseIndexInTest = std::stoi(field); }
        }

        if (valid) {
            for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
                if (!std::getline(fields, field, ',')) {
                    valid = false;
                    break;
                }
                pose.ticks[i] = static_cast<uint16_t>(std::stoi(field));
            }
        }

        if (valid) {
            poses.push_back(pose);
        }
    }
}

}  // namespace

int main() {
    const std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};

    DynamixelMotor motor(
        CYTON_DEVICE,
        CYTON_BAUD_RATE,
        CYTON_PROTOCOL_VERSION
    );

    std::vector<RecordedPose> recordedPoses;
    loadExistingPoses(OUTPUT_CSV, recordedPoses);

    // Group loaded poses by test joint, discarding any INCOMPLETE group
    // (fewer than POSES_PER_JOINT poses) -- safer and simpler than trying
    // to resume mid-sequence, since pose 4 depends on pose 2's tick from
    // the SAME session. A joint with a partial group just gets re-tested
    // from scratch.
    std::map<int, std::vector<RecordedPose>> byJoint;
    for (const auto& pose : recordedPoses) {
        byJoint[pose.testJointId].push_back(pose);
    }

    std::vector<RecordedPose> keptPoses;
    for (const auto& entry : byJoint) {
        if (static_cast<int>(entry.second.size()) == POSES_PER_JOINT) {
            for (const auto& pose : entry.second) {
                keptPoses.push_back(pose);
            }
        }
    }
    recordedPoses = keptPoses;

    auto jointAlreadyComplete = [&](int jointId) {
        int count = 0;
        for (const auto& pose : recordedPoses) {
            if (pose.testJointId == jointId) {
                ++count;
            }
        }
        return count == POSES_PER_JOINT;
    };

    std::vector<int> jointsRemaining;
    for (int jointId : TEST_JOINT_IDS) {
        if (!jointAlreadyComplete(jointId)) {
            jointsRemaining.push_back(jointId);
        }
    }

    if (recordedPoses.size() > 0) {
        std::cout
            << "Loaded " << recordedPoses.size() << " pose(s) already "
            << "recorded in " << OUTPUT_CSV << " across "
            << (TEST_JOINT_IDS.size() - jointsRemaining.size())
            << " fully-completed joint(s) -- continuing that batch.\n";
    }

    if (jointsRemaining.empty()) {
        std::cout
            << "All " << TEST_JOINT_IDS.size() << " requested joint(s) "
            << "already have " << POSES_PER_JOINT << " poses recorded. "
            << "Nothing to do -- edit TEST_JOINT_IDS if you want to "
            << "re-test something.\n";
        return 0;
    }

    std::cout << "Joint(s) still needing testing this session: ";
    for (std::size_t i = 0; i < jointsRemaining.size(); ++i) {
        std::cout << jointsRemaining[i];
        if (i + 1 < jointsRemaining.size()) {
            std::cout << ", ";
        }
    }
    std::cout << '\n';

    std::vector<std::pair<uint16_t, uint16_t>> originalAngleLimits(
        motorIds.size()
    );
    bool angleLimitsWidened = false;

    auto writeCsv = [&]() {
        std::ofstream csvOut(OUTPUT_CSV, std::ios::out | std::ios::trunc);
        if (!csvOut) {
            return;
        }
        csvOut << "pose_id,test_joint_id,pose_index_in_test";
        for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
            csvOut << ",tick_" << i;
        }
        csvOut << '\n';
        for (std::size_t p = 0; p < recordedPoses.size(); ++p) {
            const auto& pose = recordedPoses[p];
            csvOut << p + 1 << ',' << pose.testJointId << ','
                   << pose.poseIndexInTest;
            for (uint16_t tick : pose.ticks) {
                csvOut << ',' << tick;
            }
            csvOut << '\n';
        }
    };

    auto writeSnippet = [&]() {
        std::ofstream snippet(
            OUTPUT_CPP_SNIPPET,
            std::ios::out | std::ios::trunc
        );
        if (!snippet) {
            return;
        }
        snippet
            << "// Paste into TARGET_POSES in tests/test_five_pose_ndi_"
            << "capture.cpp. Each joint's 4 poses are contiguous and in "
            << "order (below, target-from-below, above, target-from-"
            << "above) -- comment marks where each joint's group starts.\n"
            << "constexpr std::size_t POSE_COUNT = "
            << recordedPoses.size() << ";\n"
            << "const std::array<std::array<uint16_t, JOINT_COUNT>, "
            << "POSE_COUNT> TARGET_POSES = {{\n";
        int lastJointId = -1;
        for (std::size_t idx = 0; idx < recordedPoses.size(); ++idx) {
            const auto& pose = recordedPoses[idx];
            if (pose.testJointId != lastJointId) {
                snippet
                    << "    // motor " << pose.testJointId
                    << " backlash test (index " << idx << "-"
                    << idx + POSES_PER_JOINT - 1 << ")\n";
                lastJointId = pose.testJointId;
            }
            snippet << "    {{";
            for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
                snippet << pose.ticks[i];
                if (i + 1 < JOINT_COUNT) {
                    snippet << ", ";
                }
            }
            snippet << "}},\n";
        }
        snippet << "}};\n";
    };

    auto restoreAngleLimits = [&]() {
        if (!angleLimitsWidened) {
            return;
        }
        std::cout << "\nRestoring original hardware angle limits...\n";
        for (std::size_t idx = 0; idx < motorIds.size(); ++idx) {
            const auto& limits = originalAngleLimits[idx];
            if (!motor.writeAngleLimits(
                    motorIds[idx], limits.first, limits.second
                )) {
                std::cerr
                    << "WARNING: failed to restore original angle limits "
                    << "for motor " << motorIds[idx] << ". They were ["
                    << limits.first << ", " << limits.second << "].\n";
            }
        }
    };

    // Locks a motor at whatever position it is CURRENTLY physically at --
    // never commands it anywhere new, so this introduces no new mechanical
    // risk regardless of which joint it's called on or when.
    auto lockAtCurrentPosition = [&](int jointId) {
        uint16_t position = 0;
        if (!motor.readPosition(jointId, position)) {
            throw std::runtime_error(
                "Failed to read position for motor " +
                std::to_string(jointId) + " while locking it."
            );
        }
        if (!motor.writeGoalPositionRaw(jointId, position)) {
            throw std::runtime_error(
                "Failed to freeze motor " + std::to_string(jointId) +
                " at its current position."
            );
        }
        if (!motor.enableTorque(jointId)) {
            throw std::runtime_error(
                "Failed to enable torque on motor " + std::to_string(jointId) +
                " to lock it."
            );
        }
    };

    try {
        if (!motor.connect()) {
            throw std::runtime_error(
                "Could not connect to the Cyton motors."
            );
        }

        for (int id : motorIds) {
            if (!motor.pingMotor(id)) {
                throw std::runtime_error(
                    "Could not ping motor " + std::to_string(id)
                );
            }
        }

        std::cout << "\nServo hardware angle limits (independent of "
                   << "jointCalibrations -- the servo itself rejects any "
                   << "goal position outside these):\n";
        for (std::size_t idx = 0; idx < motorIds.size(); ++idx) {
            const int id = motorIds[idx];
            uint16_t cwLimit = 0;
            uint16_t ccwLimit = 0;
            if (!motor.readAngleLimits(id, cwLimit, ccwLimit)) {
                throw std::runtime_error(
                    "Could not read angle limits for motor " +
                    std::to_string(id)
                );
            }
            originalAngleLimits[idx] = {cwLimit, ccwLimit};

            const JointCalibration& joint = jointCalibrations[id];
            std::cout
                << "  Motor " << id << " hardware limits: ["
                << cwLimit << ", " << ccwLimit << "]"
                << "  jointCalibrations range: ["
                << joint.minTick << ", " << joint.maxTick << "]";
            if (cwLimit > joint.minTick || ccwLimit < joint.maxTick) {
                std::cout
                    << "  <- hardware is TIGHTER than jointCalibrations "
                    << "here; hand-posing near that edge may be rejected";
            }
            std::cout << '\n';
        }

        std::cout
            << "\nWidening every motor's hardware angle limits to the "
            << "full [" << FULL_RANGE_CW_LIMIT << ", " << FULL_RANGE_CCW_LIMIT
            << "] range for this session, so any hand-posed position gets "
            << "accepted. Original limits are printed above and will be "
            << "restored automatically when this program exits normally "
            << "or hits an error -- but NOT if it's killed via Ctrl+C, so "
            << "avoid that this run, or restore them manually afterward if "
            << "you do.\n";
        for (int id : motorIds) {
            if (!motor.writeAngleLimits(
                    id, FULL_RANGE_CW_LIMIT, FULL_RANGE_CCW_LIMIT
                )) {
                throw std::runtime_error(
                    "Could not widen angle limits for motor " +
                    std::to_string(id)
                );
            }
        }
        angleLimitsWidened = true;

        // Start fully free-moving so the WHOLE arm can be posed by hand into
        // a good reference configuration before the first joint's test.
        for (int id : motorIds) {
            motor.disableTorque(id);
        }

        const int firstJoint = jointsRemaining.front();
        waitForEnter(
            "\nPosition the WHOLE arm (all 7 joints free) into a reference "
            "configuration you're comfortable holding for this session, "
            "then press Enter. Every joint except motor " +
            std::to_string(firstJoint) + " (the first one to test) will "
            "then be LOCKED (torqued, held exactly where it is)...\n"
        );

        for (int id : motorIds) {
            if (id == firstJoint) {
                continue;
            }
            lockAtCurrentPosition(id);
        }
        // firstJoint stays free (already disabled above).

        std::cout
            << "\nHand-pose recorder -- multi-joint backlash test\n"
            << "Will walk through " << jointsRemaining.size()
            << " joint(s) this session, " << POSES_PER_JOINT
            << " poses each: ";
        for (std::size_t i = 0; i < jointsRemaining.size(); ++i) {
            std::cout << jointsRemaining[i];
            if (i + 1 < jointsRemaining.size()) {
                std::cout << ", ";
            }
        }
        std::cout
            << "\nAt any point only the joint currently under test is "
            << "free -- every other joint (including ones already "
            << "finished this session) stays locked.\n"
            << "Press Ctrl+C to stop early -- poses recorded so far are "
            << "already saved to both " << OUTPUT_CSV << " and "
            << OUTPUT_CPP_SNIPPET << " (rewritten after every pose, not "
            << "batched at the end). Re-running this program later "
            << "resumes from the next incomplete joint.\n";

        for (std::size_t jointPos = 0; jointPos < jointsRemaining.size();
             ++jointPos) {
            const int testJointId = jointsRemaining[jointPos];

            if (jointPos > 0) {
                // Lock the PREVIOUS joint (just finished) before freeing
                // this one -- reads its current position (wherever pose 4
                // of ITS test left it) and holds it there.
                const int previousJoint = jointsRemaining[jointPos - 1];
                lockAtCurrentPosition(previousJoint);
                motor.disableTorque(testJointId);
            }

            std::cout
                << "\n=== Testing motor " << testJointId << " ("
                << jointPos + 1 << " of " << jointsRemaining.size()
                << " this session) ===\n"
                << "Motor " << testJointId << " is free -- move ONLY that "
                << "joint by hand into a pose. Every other motor is "
                << "locked.\n"
                << "Press Enter to lock and record each pose ("
                << POSES_PER_JOINT << " poses for this joint).\n"
                << "Torque briefly enables on motor " << testJointId
                << " to hold exactly where it is, the position is "
                << "recorded, then it releases again so you can move it "
                << "to the next pose.\n";

            uint16_t poseTwoTestJointTick = 0;
            bool havePoseTwoTick = false;

            for (int poseIndex = 0; poseIndex < POSES_PER_JOINT;
                 ++poseIndex) {
                std::array<uint16_t, JOINT_COUNT> ticks{};

                // Pose 4 (poseIndex == 3): rather than asking you to
                // hand-pose back to "the same spot" -- impossible to do
                // precisely by hand -- the servo drives itself to the
                // EXACT tick recorded for pose 2, arriving from wherever
                // pose 3 ("above") left it. This is more precise than
                // hand-positioning could ever be, and it's actually the
                // more correct test: backlash only cares about final-
                // tick-vs-approach-direction, not how carefully a human
                // reproduced a number.
                if (poseIndex == 3) {
                    if (!havePoseTwoTick) {
                        throw std::runtime_error(
                            "Pose 2's tick value was never captured -- "
                            "cannot auto-drive pose 4."
                        );
                    }
                    std::cout
                        << "\nAuto-driving motor " << testJointId
                        << " back to pose 2's exact tick ("
                        << poseTwoTestJointTick << "), arriving from "
                        << "above. Using a raw goal-position write here "
                        << "instead of moveJointSafely() -- its "
                        << "jointCalibrations range check aborts based on "
                        << "the CURRENT position, which doesn't work when "
                        << "pose 3 itself sits right at/past that "
                        << "boundary (as hand-recorded poses are allowed "
                        << "to). Safe here specifically because: only "
                        << "this one joint moves (every other joint stays "
                        << "locked), and both the start (pose 3, just "
                        << "hand-verified) and the target (pose 2's tick, "
                        << "already recorded) were already physically "
                        << "verified safe moments ago.\n";

                    if (!motor.setMovingSpeed(testJointId, 30)) {
                        throw std::runtime_error(
                            "Failed to set moving speed on motor " +
                            std::to_string(testJointId) +
                            " for the auto-drive move."
                        );
                    }
                    if (!motor.writeGoalPositionRaw(
                            testJointId, poseTwoTestJointTick
                        )) {
                        throw std::runtime_error(
                            "Failed to write goal position for motor " +
                            std::to_string(testJointId) +
                            " (auto-drive to pose 2's tick)."
                        );
                    }
                    if (!motor.enableTorque(testJointId)) {
                        throw std::runtime_error(
                            "Failed to enable torque on motor " +
                            std::to_string(testJointId) +
                            " for the auto-drive move."
                        );
                    }

                    // Tightened from 15 (2026-07-29): the elbow_pitch
                    // backlash test landed 11 ticks off pose 2's target
                    // under the old 15-tick tolerance -- some joints'
                    // longer lever arms to the marker make a loose tick-
                    // match a much bigger confound than for a joint close
                    // to the end effector, so keeping this tight matters
                    // across the board now that every joint is tested.
                    constexpr int AUTO_DRIVE_TOLERANCE_TICKS = 4;
                    constexpr int AUTO_DRIVE_TIMEOUT_MS = 8000;
                    constexpr int AUTO_DRIVE_POLL_MS = 100;
                    bool settled = false;
                    for (int elapsed = 0; elapsed <= AUTO_DRIVE_TIMEOUT_MS;
                         elapsed += AUTO_DRIVE_POLL_MS) {
                        uint16_t currentPos = 0;
                        if (!motor.readPosition(testJointId, currentPos)) {
                            throw std::runtime_error(
                                "Failed to read position for motor " +
                                std::to_string(testJointId) +
                                " while monitoring the auto-drive move."
                            );
                        }
                        if (std::abs(
                                static_cast<int>(currentPos) -
                                static_cast<int>(poseTwoTestJointTick)
                            ) <= AUTO_DRIVE_TOLERANCE_TICKS) {
                            settled = true;
                            break;
                        }
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(AUTO_DRIVE_POLL_MS)
                        );
                    }

                    if (!settled) {
                        throw std::runtime_error(
                            "Motor " + std::to_string(testJointId) +
                            " did not reach pose 2's tick (" +
                            std::to_string(poseTwoTestJointTick) +
                            ") within the timeout."
                        );
                    }

                    for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
                        uint16_t position = 0;
                        if (!motor.readPosition(motorIds[i], position)) {
                            throw std::runtime_error(
                                "Failed to read position for motor " +
                                std::to_string(motorIds[i])
                            );
                        }
                        ticks[i] = position;
                    }
                } else {
                    const char* stepNames[3] =
                        {"BELOW a chosen target", "AT that target (arriving from below -- continue the same direction as the last move)", "ABOVE the same target"};
                    waitForEnter(
                        "\nMove motor " + std::to_string(testJointId) +
                        " " + stepNames[poseIndex] + " (pose " +
                        std::to_string(poseIndex + 1) + " of " +
                        std::to_string(POSES_PER_JOINT) + " for this "
                        "joint), then press Enter..."
                    );

                    bool locked = false;

                    while (!locked) {
                        for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
                            uint16_t position = 0;
                            if (!motor.readPosition(motorIds[i], position)) {
                                throw std::runtime_error(
                                    "Failed to read position for motor " +
                                    std::to_string(motorIds[i])
                                );
                            }
                            ticks[i] = position;
                        }

                        // Only the TEST joint needs locking here -- every
                        // other joint is already torqued/held from
                        // before this joint's test started.
                        const uint16_t testJointPos =
                            ticks[static_cast<std::size_t>(testJointId)];
                        if (!motor.writeGoalPositionRaw(
                                testJointId, testJointPos
                            )) {
                            std::cout
                                << "Motor " << testJointId << "'s current "
                                << "position (" << testJointPos << ") "
                                << "failed to write (see the error above "
                                << "-- likely a communication issue, not "
                                << "a range check).\n";
                            waitForEnter(
                                "Press Enter once repositioned, to try "
                                "locking pose " +
                                std::to_string(poseIndex + 1) + " again..."
                            );
                            continue;
                        }

                        if (!motor.enableTorque(testJointId)) {
                            std::cout
                                << "Failed to enable torque on motor "
                                << testJointId << " (see the error "
                                << "above).\n";
                            waitForEnter(
                                "Press Enter to retry pose " +
                                std::to_string(poseIndex + 1) + "..."
                            );
                            continue;
                        }

                        locked = true;
                    }
                }

                if (poseIndex == 1) {
                    poseTwoTestJointTick =
                        ticks[static_cast<std::size_t>(testJointId)];
                    havePoseTwoTick = true;
                }

                std::cout
                    << "Recorded motor " << testJointId << " pose "
                    << poseIndex + 1 << " of " << POSES_PER_JOINT << ": ";
                for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
                    const JointCalibration& joint = jointCalibrations[i];
                    if (ticks[i] < joint.minTick || ticks[i] > joint.maxTick) {
                        std::cout
                            << "\n  WARNING: motor " << i << " reading "
                            << ticks[i] << " is outside its calibrated "
                            << "range [" << joint.minTick << ", "
                            << joint.maxTick << "]";
                    }
                    std::cout << ticks[i] << ' ';
                }
                std::cout << '\n';

                recordedPoses.push_back(
                    RecordedPose{testJointId, poseIndex + 1, ticks}
                );
                writeCsv();
                writeSnippet();

                // Only release the TEST joint back to free-moving for the
                // next pose -- everything else stays locked.
                motor.disableTorque(testJointId);
            }

            std::cout
                << "Motor " << testJointId << " done ("
                << jointPos + 1 << " of " << jointsRemaining.size()
                << " joints this session).\n";
        }

        // Lock the final joint tested too, for a consistent end state,
        // before releasing everything.
        lockAtCurrentPosition(jointsRemaining.back());

        std::cout
            << "\nAll requested joints done -- " << recordedPoses.size()
            << " total poses recorded.\n"
            << "Saved to " << OUTPUT_CSV << " and " << OUTPUT_CPP_SNIPPET
            << '\n'
            << "Releasing all 7 motors now.\n";

        for (int id : motorIds) {
            motor.disableTorque(id);
        }
        restoreAngleLimits();
        motor.disconnect();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\nERROR: " << error.what() << '\n';
        std::cerr
            << recordedPoses.size() << " pose(s) were already saved to "
            << OUTPUT_CSV << " before this error. Re-run this program to "
            << "resume from the next incomplete joint.\n";

        for (int id : motorIds) {
            motor.disableTorque(id);
        }
        restoreAngleLimits();
        motor.disconnect();
        return 1;
    }
}
