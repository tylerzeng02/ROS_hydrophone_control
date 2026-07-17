cmake_minimum_required(VERSION 3.10)
project(cyton_setup)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)

include_directories(src)
include_directories(
    external/DynamixelSDK/c++/include/dynamixel_sdk
)

add_executable(cyton_setup
    src/main.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_link_libraries(cyton_setup
    dxl_x64_cpp
)

add_executable(test_radians
    tests/test_radians.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(test_radians PRIVATE
    src
)

target_link_libraries(test_radians
    dxl_x64_cpp
)

add_executable(test_radians_pose
    tests/test_radians_pose.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(test_radians_pose PRIVATE
    src
)

target_link_libraries(test_radians_pose
    dxl_x64_cpp
)

add_executable(read_motor_positions
    tests/read_motor_positions.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(read_motor_positions PRIVATE
    src
    external/DynamixelSDK/c++/include/dynamixel_sdk
)

target_link_libraries(read_motor_positions
    dxl_x64_cpp
)

add_executable(test_ticks
    tests/test_ticks.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(test_ticks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
)

target_link_libraries(test_ticks PRIVATE
    dxl_x64_cpp
)

add_subdirectory(
    external/trac_ik/trac_ik_lib
    external/trac_ik_build
)

add_executable(test_trac_ik
    tests/test_trac_ik.cpp
)

target_link_libraries(test_trac_ik PRIVATE
    trac_ik
)

add_executable(test_cyton_ik
    tests/test_cyton_ik.cpp
    src/robot_calibration.cpp
)

target_include_directories(test_cyton_ik PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/trac_ik/trac_ik_lib/include
    ${orocos_kdl_INCLUDE_DIRS}
)

target_link_libraries(test_cyton_ik PRIVATE
    trac_ik
    ${orocos_kdl_LIBRARIES}
)

add_executable(test_cyton_ik_move
    tests/test_cyton_ik_move.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(test_cyton_ik_move PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/trac_ik/trac_ik_lib/include
    ${orocos_kdl_INCLUDE_DIRS}
)

target_link_libraries(test_cyton_ik_move PRIVATE
    trac_ik
    ${orocos_kdl_LIBRARIES}
    dxl_x64_cpp
)

add_executable(test_cyton_ik_target
    tests/test_cyton_ik_target.cpp
    src/robot_calibration.cpp
)

target_include_directories(test_cyton_ik_target PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/trac_ik/trac_ik_lib/include
    ${orocos_kdl_INCLUDE_DIRS}
)

target_link_libraries(test_cyton_ik_target PRIVATE
    trac_ik
    ${orocos_kdl_LIBRARIES}
)

add_executable(test_cyton_ik_target_move
    tests/test_cyton_ik_target_move.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(test_cyton_ik_target_move PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/trac_ik/trac_ik_lib/include
    ${orocos_kdl_INCLUDE_DIRS}
    ${kdl_parser_INCLUDE_DIRS}
)

target_link_libraries(test_cyton_ik_target_move PRIVATE
    trac_ik
    ${orocos_kdl_LIBRARIES}
    kdl_parser
    dxl_x64_cpp
)

add_executable(test_home_pose
    tests/test_home_pose.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(test_home_pose PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include/dynamixel_sdk
)

target_link_libraries(test_home_pose PRIVATE
    dxl_x64_cpp
)