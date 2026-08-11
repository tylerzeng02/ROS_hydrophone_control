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

install(
  TARGETS pose_commander cartesian_path_demo waypoint_sequence_demo
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
