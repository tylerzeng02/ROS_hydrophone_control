cmake_minimum_required(VERSION 3.22)
project(cyton_ndi_capture)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

# Builds the top-level repo's own external/ndicapi directly (same pattern
# as cyton_hardware's CMakeLists.txt building external/DynamixelSDK) so
# this stays a single source of truth rather than a copy.
get_filename_component(CYTON_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

if(NOT EXISTS "${CYTON_REPO_ROOT}/external/ndicapi/CMakeLists.txt")
  message(FATAL_ERROR
    "ndicapi not found at ${CYTON_REPO_ROOT}/external/ndicapi -- "
    "run 'git submodule update --init external/ndicapi' from the repo root."
  )
endif()

set(ndicapi_BUILD_APPLICATIONS OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON OFF CACHE BOOL "" FORCE)

add_subdirectory(
  "${CYTON_REPO_ROOT}/external/ndicapi"
  "${CMAKE_CURRENT_BINARY_DIR}/ndicapi_build"
)

add_executable(ndi_measure
  src/ndi_measure.cpp
)

target_include_directories(ndi_measure PRIVATE
  "${CYTON_REPO_ROOT}/external/ndicapi"
)

ament_target_dependencies(ndi_measure
  rclcpp
  sensor_msgs
  moveit_ros_planning_interface
)

target_link_libraries(ndi_measure ndicapi)

# Plain marker-visibility diagnostic -- no ROS dependency at all (doesn't
# need joint states), just ndicapi.
add_executable(ndi_status_monitor
  src/ndi_status_monitor.cpp
)

target_include_directories(ndi_status_monitor PRIVATE
  "${CYTON_REPO_ROOT}/external/ndicapi"
)

target_link_libraries(ndi_status_monitor ndicapi)

install(
  TARGETS ndi_measure ndi_status_monitor
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
