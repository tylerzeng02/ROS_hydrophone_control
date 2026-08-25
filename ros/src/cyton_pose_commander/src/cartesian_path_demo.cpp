/**
 * @file cartesian_path_demo.cpp
 * @brief Reads a CSV of Cartesian waypoints (x,y,z,roll,pitch,yaw per
 * row, meters/radians), computes one continuous Cartesian path through
 * all of them via MoveGroupInterface::computeCartesianPath(), previews
 * it in RViz, waits for Enter, then executes it, driving the arm
 * smoothly through every waypoint in one motion rather than
 * planning/executing each point separately.
 *
 * Built for demo purposes: default usage is against mock_components
 * (hardware_type:=mock_components), a safe, repeatable, simulated
 * animation of the arm sweeping through a sequence of points. No real
 * hardware is required, though it works identically against real
 * hardware too (same command, real launch instead of mock).
 *
 * eef_step, the Cartesian interpolation resolution, is deliberately
 * modest (1cm): fine enough for a smooth-looking path, coarse enough that
 * computing the path over 20 widely-spaced points does not take long.
 */

#include <array>
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
constexpr double PLANNING_TIME_SECONDS = 30.0;  // see move_x_test.cpp's own
                                                  // comment for why this
                                                  // is not the 5s default:
                                                  // same elbow_yaw-locked-
                                                  // window IK search issue.
constexpr double EEF_STEP_M = 0.01;
constexpr double MIN_ACCEPTABLE_FRACTION = 0.9;  // warn (not abort) below this

void waitForEnter(const std::string& message) {
    std::cout << message << std::flush;
    std::string line;
    std::getline(std::cin, line);
}

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
    // First line is assumed to be a header ("x,y,z,roll,pitch,yaw") and is
    // discarded unconditionally. Unlike some other tools in this project,
    // this one does not try to detect whether it is numeric, since a plain
    // 6-column waypoint file has no unambiguous way to tell a header from
    // a legitimate first data row.

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
    auto node = std::make_shared<rclcpp::Node>("cartesian_path_demo");

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

        std::vector<geometry_msgs::msg::Pose> poseWaypoints;
        poseWaypoints.reserve(waypoints.size());
        for (const auto& w : waypoints) {
            poseWaypoints.push_back(toPose(w));
        }

        std::cout << "Computing Cartesian path through all " << waypoints.size()
                  << " waypoints (eef_step=" << EEF_STEP_M << "m)...\n";
        moveit_msgs::msg::RobotTrajectory trajectory;
        const double fraction = moveGroup.computeCartesianPath(
            poseWaypoints, EEF_STEP_M, trajectory
        );
        std::cout << "Achieved " << (fraction * 100.0) << "% of the requested path.\n";
        if (fraction < MIN_ACCEPTABLE_FRACTION) {
            std::cout << "WARNING: less than " << (MIN_ACCEPTABLE_FRACTION * 100.0)
                      << "% of the path was achievable -- some waypoints may be "
                         "unreachable or blocked by a joint limit/collision. "
                         "Check RViz's preview before executing.\n";
        }

        std::cout << "\nPath computed -- check RViz's MotionPlanning display for a preview "
                     "(published automatically, no action needed here).\n";
        waitForEnter(
            "Press Enter to execute the full path, or Ctrl+C to stop without moving... "
        );

        std::cout << "Executing...\n";
        moveit::planning_interface::MoveGroupInterface::Plan planMsg;
        planMsg.trajectory = trajectory;
        auto result = moveGroup.execute(planMsg);
        if (result == moveit::core::MoveItErrorCode::SUCCESS) {
            std::cout << "Execution succeeded -- moved through all " << waypoints.size()
                      << " waypoints.\n";
        } else {
            std::cout << "Execution FAILED, error code " << result.val << '\n';
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
