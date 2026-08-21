// record_waypoints: jog/plan the arm via RViz's MotionPlanning panel
// (drag the marker, Plan, Execute, however you like), and press Enter in
// this terminal at each pose you want to keep. Each press reads the arm's
// live current joint state and appends one row to a CSV. No NDI tracker
// is needed at all; this is pure MoveIt/joint-state recording.
//
// Output uses the exact same column schema (capture_id + the 7
// *_joint_rad columns, header-resolved) that replay_ndi_capture.cpp /
// replay_ndi_capture_sim.cpp already read, so a file recorded here can
// be fed straight back into either of those existing tools to replay the
// sequence automatically, with no changes needed on that side. Extra
// columns those tools' full ndi_measure-oriented schema also recognizes
// (timestamp_ms, moveit_pose_*, moving_camera_*, etc.) are simply omitted
// here. replay_ndi_capture only requires capture_id and the 7 joint
// columns, nothing else.
//
// Deliberately does not append to an existing file by default. This
// project has hit the same "hardcoded output path + silent resume"
// cross-session-mixing bug more than once (record_hand_poses_fixed_
// elbow_yaw.cpp, ndi_measure.cpp). If the target file already exists,
// this tool refuses to start rather than silently gluing a new session
// onto old data; pass a different filename instead.

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "moveit/move_group_interface/move_group_interface.hpp"

namespace {

constexpr const char* DEFAULT_OUTPUT_CSV = "recorded_waypoints.csv";
constexpr const char* PLANNING_GROUP = "arm";

constexpr std::array<const char*, 7> JOINT_NAMES = {
    "shoulder_roll_joint", "shoulder_pitch_joint", "shoulder_yaw_joint",
    "elbow_pitch_joint",   "elbow_yaw_joint",       "wrist_pitch_joint",
    "wrist_roll_joint",
};

bool fileExists(const std::string& path) {
    struct stat buffer;
    return stat(path.c_str(), &buffer) == 0;
}

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [output_csv]\n"
              << "  output_csv: where to write recorded waypoints. Refuses to overwrite an "
                 "existing file -- pass a fresh filename each session.\n"
              << "  Default: " << DEFAULT_OUTPUT_CSV << '\n'
              << "  Workflow: jog/plan the arm via RViz (drag marker, Plan, Execute), then press "
                 "Enter in this terminal to record that pose. Repeat. Ctrl+C (or Ctrl+\\ if that "
                 "doesn't respond) to stop.\n"
              << "  The output is directly compatible with replay_ndi_capture / "
                 "replay_ndi_capture_sim -- no NDI tracker involved here at all.\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);

    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        printUsage(argv[0]);
        return 0;
    }
    const std::string outputCsv = argc > 1 ? argv[1] : DEFAULT_OUTPUT_CSV;

    if (fileExists(outputCsv)) {
        std::cerr << "Refusing to start: '" << outputCsv
                  << "' already exists. Pass a different filename, or move/rename the existing "
                     "file first -- this avoids silently mixing two recording sessions together "
                     "(a real bug this project has hit more than once).\n";
        return 1;
    }

    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("record_waypoints");

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinThread([&executor]() { executor.spin(); });

    int exitCode = 0;
    try {
        moveit::planning_interface::MoveGroupInterface moveGroup(node, PLANNING_GROUP);

        std::ofstream csv(outputCsv, std::ios::out | std::ios::trunc);
        if (!csv) {
            throw std::runtime_error("Could not open output CSV: " + outputCsv);
        }
        csv << "capture_id,shoulder_roll_joint_rad,shoulder_pitch_joint_rad,shoulder_yaw_joint_rad,"
               "elbow_pitch_joint_rad,elbow_yaw_joint_rad,wrist_pitch_joint_rad,"
               "wrist_roll_joint_rad\n";
        csv.flush();

        std::cout << "\nReady. Jog/plan the arm via RViz, then press Enter here to record the "
                     "current pose. Ctrl+C to stop.\n"
                     "Writing to: "
                  << outputCsv << "\n\n";

        int captureId = 0;
        std::string line;
        while (true) {
            std::cout << "[waypoint " << (captureId + 1) << "] Press Enter to record (Ctrl+C to stop): "
                      << std::flush;
            if (!std::getline(std::cin, line)) {
                break;  // EOF (e.g. piped input ran out); stop cleanly
            }

            auto currentState = moveGroup.getCurrentState(2.0);
            if (!currentState) {
                std::cout << "  WARNING: could not read current robot state -- try again.\n";
                continue;
            }

            ++captureId;
            csv << captureId;
            for (const char* jointName : JOINT_NAMES) {
                csv << ',' << currentState->getVariablePosition(jointName);
            }
            csv << '\n';
            csv.flush();

            std::cout << "  Recorded waypoint " << captureId << ".\n";
        }

        std::cout << "\nDone. " << captureId << " waypoint(s) written to " << outputCsv << ".\n";
    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << '\n';
        exitCode = 1;
    }

    executor.cancel();
    spinThread.join();
    rclcpp::shutdown();
    return exitCode;
}
