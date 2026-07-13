cmake_minimum_required(VERSION 3.10)
project(cyton_setup)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src)
include_directories(external/DynamixelSDK/c++/include/dynamixel_sdk)

add_executable(cyton_setup
    src/main.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_link_libraries(cyton_setup dxl_x64_cpp)

add_executable(test_radians
    tests/test_radians.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(test_radians PRIVATE src)

target_link_libraries(test_radians dxl_x64_cpp)

add_executable(test_radians_pose
    tests/test_radians_pose.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(test_radians_pose PRIVATE src)

target_link_libraries(test_radians_pose dxl_x64_cpp)