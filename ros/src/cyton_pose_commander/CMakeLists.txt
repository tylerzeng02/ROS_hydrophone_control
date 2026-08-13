cmake_minimum_required(VERSION 3.22)
project(cyton_pose_commander)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Compiles the top-level repo's own src/robot_calibration.cpp directly
# (same pattern as cyton_hardware building dynamixel_motor.cpp/
# robot_calibration.cpp) so tick<->radian conversion always matches the
# exact calibrated model, not a reimplementation.
get_filename_component(CYTON_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

add_executable(pose_commander
  src/pose_commander.cpp
  "${CYTON_REPO_ROOT}/src/robot_calibration.cpp"
)

target_include_directories(pose_commander PRIVATE
  "${CYTON_REPO_ROOT}/src"
)

ament_target_dependencies(pose_commander
  rclcpp
  moveit_ros_planning_interface
)

# cartesian_path_demo (2026-08-11): reads a CSV of Cartesian waypoints,
# computes one continuous path through all of them, previews it, and
# executes on Enter -- see the .cpp's own header comment.
add_executable(cartesian_path_demo
  src/cartesian_path_demo.cpp
)

ament_target_dependencies(cartesian_path_demo
  rclcpp
  moveit_ros_planning_interface
  tf2
  tf2_geometry_msgs
)

# waypoint_sequence_demo (2026-08-11): visits each waypoint one at a time
# (independent IK solve + plan + execute + pause) instead of one
# continuous interpolated path -- see the .cpp's own header comment for
# why this is kept as a separate tool from cartesian_path_demo.
add_executable(waypoint_sequence_demo
  src/waypoint_sequence_demo.cpp
)

ament_target_dependencies(waypoint_sequence_demo
  rclcpp
  moveit_ros_planning_interface
  tf2
  tf2_geometry_msgs
)

# replay_ndi_capture (2026-08-11): reads a cyton_ndi_capture/ndi_measure
# output CSV (joint angles already in radians as named columns) and drives
# the arm back through each recorded pose via MoveIt -- see the .cpp's own
# header comment. Reuses pose_commander's plan-preview-confirm workflow and
# out-of-bounds recovery logic, so it needs the same robot_calibration.cpp
# dependency for jointCalibrations/ticksToRadians.
add_executable(replay_ndi_capture
  src/replay_ndi_capture.cpp
  "${CYTON_REPO_ROOT}/src/robot_calibration.cpp"
)

target_include_directories(replay_ndi_capture PRIVATE
  "${CYTON_REPO_ROOT}/src"
)

ament_target_dependencies(replay_ndi_capture
  rclcpp
  moveit_ros_planning_interface
)

# replay_ndi_capture_sim (2026-08-11): automatic-animation version of
# replay_ndi_capture -- visits every pose in order, pausing 1s between
# each, no per-pose Enter confirmation. Built for demo/simulation purposes
# (mock hardware by default) -- see the .cpp's own header comment.
add_executable(replay_ndi_capture_sim
  src/replay_ndi_capture_sim.cpp
  "${CYTON_REPO_ROOT}/src/robot_calibration.cpp"
)

target_include_directories(replay_ndi_capture_sim PRIVATE
  "${CYTON_REPO_ROOT}/src"
)

ament_target_dependencies(replay_ndi_capture_sim
  rclcpp
  moveit_ros_planning_interface
)

# constrained_pose_ik_test (2026-08-13): replicates the ORIGINAL, harder
# setPoseTarget() IK scenario (position + orientation both constrained,
# OMPL random-seeded goal sampling) that move_between_points.cpp used
# before its seeded-IK fix -- built specifically to A/B compare ik_solver:=
# kdl vs. ik_solver:=trac_ik on that exact scenario. No NDI dependency.
add_executable(constrained_pose_ik_test
  src/constrained_pose_ik_test.cpp
)

ament_target_dependencies(constrained_pose_ik_test
  rclcpp
  moveit_ros_planning_interface
)

# record_waypoints (2026-08-13): jog/plan the arm via RViz, press Enter to
# record the current joint state to a CSV -- no NDI tracker involved.
# Output uses the same column schema replay_ndi_capture/
# replay_ndi_capture_sim already read, so recorded waypoints can be
# replayed automatically with either of those existing tools unchanged.
add_executable(record_waypoints
  src/record_waypoints.cpp
)

ament_target_dependencies(record_waypoints
  rclcpp
  moveit_ros_planning_interface
)

install(
  TARGETS pose_commander cartesian_path_demo waypoint_sequence_demo replay_ndi_capture
          replay_ndi_capture_sim constrained_pose_ik_test record_waypoints
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
