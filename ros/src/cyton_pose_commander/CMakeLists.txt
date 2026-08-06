cmake_minimum_required(VERSION 3.22)
project(cyton_pose_commander)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

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

install(
  TARGETS pose_commander
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
