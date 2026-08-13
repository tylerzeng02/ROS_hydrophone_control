cmake_minimum_required(VERSION 3.22)
project(cyton_hardware)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# This package compiles the top-level repo's own src/dynamixel_motor.cpp
# and src/robot_calibration.cpp directly (not a copy/reimplementation) so
# the hardware interface always uses the exact same safety clamps, tick/
# radian calibration, and backlash-overshoot table as every other
# motor-facing program in this repo. See CLAUDE.md's kinematic-calibration
# section, "the one non-obvious, easy-to-get-wrong integration requirement".
get_filename_component(CYTON_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

if(NOT EXISTS "${CYTON_REPO_ROOT}/external/DynamixelSDK/c++/CMakeLists.txt")
  message(FATAL_ERROR
    "DynamixelSDK not found at ${CYTON_REPO_ROOT}/external/DynamixelSDK -- "
    "see the top-level repo's CLAUDE.md ('Dependencies' section)."
  )
endif()

add_subdirectory(
  "${CYTON_REPO_ROOT}/external/DynamixelSDK/c++"
  "${CMAKE_CURRENT_BINARY_DIR}/DynamixelSDK_build"
)

add_library(cyton_hardware SHARED
  src/cyton_system_hardware.cpp
  "${CYTON_REPO_ROOT}/src/dynamixel_motor.cpp"
  "${CYTON_REPO_ROOT}/src/robot_calibration.cpp"
  "${CYTON_REPO_ROOT}/src/pose_dependent_correction.cpp"
)

target_include_directories(cyton_hardware PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  "${CYTON_REPO_ROOT}/src"
  "${CYTON_REPO_ROOT}/external/DynamixelSDK/c++/include"
)

ament_target_dependencies(cyton_hardware
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

target_link_libraries(cyton_hardware dynamixel_sdk)

pluginlib_export_plugin_description_file(hardware_interface cyton_hardware.xml)

install(
  TARGETS cyton_hardware
  DESTINATION lib
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  FILES cyton_hardware.xml
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(cyton_hardware)
ament_export_dependencies(hardware_interface pluginlib rclcpp rclcpp_lifecycle)

ament_package()
