// waypoint_sequence_demo: reads a CSV of Cartesian waypoints (x,y,z,roll,
// pitch,yaw per row, meters/radians), and visits them ONE AT A TIME --
// each point gets its own independent setPoseTarget()/IK solve/plan/
// execute, then the arm pauses briefly before moving to the next point.
// This is deliberately different from cartesian_path_demo.cpp's single
// continuous interpolated path -- that one produces one smooth motion
// through all waypoints; this one produces N separate point-to-point
// moves with a visible stop at each, which is what was actually wanted
// for this demo (both tools are kept, since they answer different
// questions -- smooth sweep vs. discrete visit-and-pause).
//
// Runs fully automatically (no per-point Enter confirmation) -- advances
// on its own after each pause, since the point is a hands-off animation/
// demo, not an interactive review workflow like pose_commander's.
//
// Same elbow_yaw-locked-window IK considerations as move_x_test.cpp/
// move_between_points.cpp apply here (each point does an independent
// Cartesian IK solve) -- setPlanningTime/setNumPlanningAttempts are set
// the same way, see those files' own comments for why.

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "moveit/move_group_interface/move_group_interface.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace {

constexpr const char* DEFAULT_INPUT_CSV = "demo_waypoints_20.csv";
constexpr const char* PLANNING_GROUP = "arm";
constexpr double PLANNING_TIME_SECONDS = 30.0;
constexpr unsigned int NUM_PLANNING_ATTEMPTS = 15;
constexpr int PAUSE_AT_WAYPOINT_MS = 500;

struct Waypoint {
    double x, y, z, roll, pitch, yaw;
};

std::vector<Waypoint> loadWaypoints(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open input CSV: " + path);
    }

    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("Input CSV is empty: " + path);
    }
    // First line assumed to be a header, discarded unconditionally -- see
    // the identical note in cartesian_path_demo.cpp for why.

    std::vector<Waypoint> waypoints;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<double> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) {
            fields.push_back(std::stod(field));
        }
        if (fields.size() != 6) {
            std::cout << "WARNING: skipping malformed row (expected 6 fields, "
                         "got " << fields.size() << "): " << line << '\n';
            continue;
        }
        waypoints.push_back({fields[0], fields[1], fields[2], fields[3], fields[4], fields[5]});
    }
    return waypoints;
}

geometry_msgs::msg::Pose toPose(const Waypoint& w) {
    geometry_msgs::msg::Pose pose;
    pose.position.x = w.x;
    pose.position.y = w.y;
    pose.position.z = w.z;
    tf2::Quaternion q;
    q.setRPY(w.roll, w.pitch, w.yaw);
    pose.orientation = tf2::toMsg(q);
    return pose;
}

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [input_csv]\n"
              << "  input_csv: x,y,z,roll,pitch,yaw per row (meters/radians), "
                 "first line is a discarded header.\n"
              << "  Default: " << DEFAULT_INPUT_CSV << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);

    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        printUsage(argv[0]);
        return 0;
    }
    const std::string inputCsv = argc > 1 ? argv[1] : DEFAULT_INPUT_CSV;

    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("waypoint_sequence_demo");

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinThread([&executor]() { executor.spin(); });

    int exitCode = 0;
    try {
        std::cout << "Loading waypoints from " << inputCsv << "...\n";
        std::vector<Waypoint> waypoints = loadWaypoints(inputCsv);
        std::cout << "Loaded " << waypoints.size() << " waypoints.\n";
        if (waypoints.empty()) {
            throw std::runtime_error("No waypoints loaded -- nothing to do.");
        }

        moveit::planning_interface::MoveGroupInterface moveGroup(node, PLANNING_GROUP);
        moveGroup.setPlanningTime(PLANNING_TIME_SECONDS);
        moveGroup.setNumPlanningAttempts(NUM_PLANNING_ATTEMPTS);

        int succeeded = 0;
        for (std::size_t i = 0; i < waypoints.size(); ++i) {
            const Waypoint& w = waypoints[i];
            std::cout << "\n=== Waypoint " << (i + 1) << "/" << waypoints.size()
                      << " === (x=" << w.x << ", y=" << w.y << ", z=" << w.z
                      << ", rpy=" << w.roll << "," << w.pitch << "," << w.yaw << ")\n";

            moveGroup.setPoseTarget(toPose(w));
            std::cout << "  Planning and executing...\n";
            auto result = moveGroup.move();

            if (result == moveit::core::MoveItErrorCode::SUCCESS) {
                std::cout << "  Reached waypoint " << (i + 1) << ".\n";
                ++succeeded;
            } else {
                std::cout << "  FAILED, error code " << result.val
                          << " -- skipping to next waypoint.\n";
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(PAUSE_AT_WAYPOINT_MS));
        }

        std::cout << "\nDone. " << succeeded << "/" << waypoints.size()
                  << " waypoints reached.\n";
        if (succeeded < static_cast<int>(waypoints.size())) {
            exitCode = 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << '\n';
        exitCode = 1;
    }

    executor.cancel();
    spinThread.join();
    rclcpp::shutdown();
    return exitCode;
}
