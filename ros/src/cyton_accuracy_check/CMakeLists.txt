cmake_minimum_required(VERSION 3.22)
project(cyton_accuracy_check)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

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

add_executable(run_accuracy_check
  src/run_accuracy_check.cpp
  "${CYTON_REPO_ROOT}/src/robot_calibration.cpp"
)

target_include_directories(run_accuracy_check PRIVATE
  "${CYTON_REPO_ROOT}/src"
  "${CYTON_REPO_ROOT}/external/ndicapi"
)

ament_target_dependencies(run_accuracy_check
  rclcpp
  moveit_ros_planning_interface
)

target_link_libraries(run_accuracy_check ndicapi)

# move_x_test: commands a fixed Cartesian delta (default 5cm
# +X) via MoveIt's setPoseTarget()/IK, checks the result with NDI -- see
# the .cpp's own header comment for why this is a genuinely different test
# from run_accuracy_check (relative move, real IK, no base/tool-frame fit
# needed for the comparison).
add_executable(move_x_test
  src/move_x_test.cpp
)

target_include_directories(move_x_test PRIVATE
  "${CYTON_REPO_ROOT}/external/ndicapi"
)

ament_target_dependencies(move_x_test
  rclcpp
  moveit_ros_planning_interface
)

target_link_libraries(move_x_test ndicapi)

# move_xyz_test: like move_x_test, but commands a delta on all
# three axes at once (dx, dy, dz all plugged in on the command line)
# instead of one axis at a time -- see the .cpp's own header comment.
add_executable(move_xyz_test
  src/move_xyz_test.cpp
)

target_include_directories(move_xyz_test PRIVATE
  "${CYTON_REPO_ROOT}/external/ndicapi"
)

ament_target_dependencies(move_xyz_test
  rclcpp
  moveit_ros_planning_interface
)

target_link_libraries(move_xyz_test ndicapi)

# move_between_points: visits a sequence of real, NDI-measured
# target points (collected separately via cyton_ndi_capture's ndi_measure)
# by commanding a relative Cartesian move from the arm's live current
# position to each one in turn -- see the .cpp's own header comment for the
# NDI-frame -> MoveIt-frame rotation this depends on.
add_executable(move_between_points
  src/move_between_points.cpp
)

target_include_directories(move_between_points PRIVATE
  "${CYTON_REPO_ROOT}/external/ndicapi"
)

ament_target_dependencies(move_between_points
  rclcpp
  moveit_ros_planning_interface
)

target_link_libraries(move_between_points ndicapi)

install(
  TARGETS run_accuracy_check move_x_test move_xyz_test move_between_points
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
