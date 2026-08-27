cmake_minimum_required(VERSION 3.22)
project(cyton_trac_ik_kinematics_plugin)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(pluginlib REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(urdf REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS date_time)

find_package(PkgConfig REQUIRED)
pkg_check_modules(NLOPT REQUIRED nlopt)
pkg_check_modules(OROCOS_KDL REQUIRED orocos-kdl)
pkg_check_modules(EIGEN3 REQUIRED eigen3)

# Compiles third_party/trac_ik_lib's core solver source, a vendored and
# modified copy of trac_ik_lib (see third_party/trac_ik_lib/README.md for
# what was changed and why this is vendored rather than a submodule).
# Deliberately excludes trac_ik_ros.cpp (ROS1/catkin-only, not needed --
# see this package's plugin header comment) -- only kdl_tl.cpp/
# nlopt_ik.cpp/trac_ik.cpp, which have zero ROS dependency of any kind.
get_filename_component(TRAC_IK_LIB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/trac_ik_lib" ABSOLUTE)

add_library(cyton_trac_ik_kinematics_plugin SHARED
  src/trac_ik_kinematics_plugin.cpp
  "${TRAC_IK_LIB_ROOT}/src/kdl_tl.cpp"
  "${TRAC_IK_LIB_ROOT}/src/nlopt_ik.cpp"
  "${TRAC_IK_LIB_ROOT}/src/trac_ik.cpp"
)

target_include_directories(cyton_trac_ik_kinematics_plugin PRIVATE
  include
  "${TRAC_IK_LIB_ROOT}/include"
  ${EIGEN3_INCLUDE_DIRS}
  ${OROCOS_KDL_INCLUDE_DIRS}
  ${NLOPT_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

target_compile_options(cyton_trac_ik_kinematics_plugin PRIVATE
  ${EIGEN3_CFLAGS_OTHER}
  ${OROCOS_KDL_CFLAGS_OTHER}
  ${NLOPT_CFLAGS_OTHER}
)

ament_target_dependencies(cyton_trac_ik_kinematics_plugin
  rclcpp
  moveit_core
  pluginlib
  kdl_parser
  urdf
  geometry_msgs
  moveit_msgs
)

target_link_libraries(cyton_trac_ik_kinematics_plugin
  ${OROCOS_KDL_LIBRARIES}
  ${NLOPT_LIBRARIES}
  ${Boost_LIBRARIES}
  Threads::Threads
)

pluginlib_export_plugin_description_file(moveit_core trac_ik_kinematics_description.xml)

install(
  TARGETS cyton_trac_ik_kinematics_plugin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  FILES trac_ik_kinematics_description.xml
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
