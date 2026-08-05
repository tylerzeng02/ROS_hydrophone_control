cmake_minimum_required(VERSION 3.22.6)
project(cyton_setup)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# Dynamixel SDK
# ------------------------------------------------------------

if(EXISTS
    "${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/CMakeLists.txt"
)
    add_subdirectory(
        external/DynamixelSDK/c++
        external/DynamixelSDK_build
    )
else()
    message(FATAL_ERROR
        "DynamixelSDK not found at external/DynamixelSDK"
    )
endif()

# ------------------------------------------------------------
# Common include directories
# ------------------------------------------------------------

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
)

# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

add_executable(cyton_setup
    src/main.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_link_libraries(cyton_setup
    PRIVATE
        dynamixel_sdk
)

# ------------------------------------------------------------
# Basic Dynamixel tests
# ------------------------------------------------------------

add_executable(test_radians
    tests/test_radians.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(test_radians
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
)

target_link_libraries(test_radians
    PRIVATE
        dynamixel_sdk
)

add_executable(test_radians_pose
    tests/test_radians_pose.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(test_radians_pose
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
)

target_link_libraries(test_radians_pose
    PRIVATE
        dynamixel_sdk
)

add_executable(read_motor_positions
    tests/read_motor_positions.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(read_motor_positions
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
)

target_link_libraries(read_motor_positions
    PRIVATE
        dynamixel_sdk
)

add_executable(test_ticks
    tests/test_ticks.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(test_ticks
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
)

target_link_libraries(test_ticks
    PRIVATE
        dynamixel_sdk
)

add_executable(test_home_pose
    tests/test_home_pose.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(test_home_pose
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
)

target_link_libraries(test_home_pose
    PRIVATE
        dynamixel_sdk
)

add_executable(record_hand_poses
    tests/record_hand_poses.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(record_hand_poses
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
)

target_link_libraries(record_hand_poses
    PRIVATE
        dynamixel_sdk
)

add_executable(record_hand_poses_fixed_elbow_yaw
    tests/record_hand_poses_fixed_elbow_yaw.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(record_hand_poses_fixed_elbow_yaw
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
)

target_link_libraries(record_hand_poses_fixed_elbow_yaw
    PRIVATE
        dynamixel_sdk
)

add_executable(reset_compliance
    tests/reset_compliance.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(reset_compliance
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
)

target_link_libraries(reset_compliance
    PRIVATE
        dynamixel_sdk
)

# ------------------------------------------------------------
# TRAC-IK targets
# Only enabled when TRAC-IK exists
# ------------------------------------------------------------

if(EXISTS
    "${CMAKE_CURRENT_SOURCE_DIR}/external/trac_ik/trac_ik_lib/CMakeLists.txt"
)

    add_subdirectory(
        external/trac_ik/trac_ik_lib
        external/trac_ik_build
    )

    add_executable(test_trac_ik
        tests/test_trac_ik.cpp
    )

    target_link_libraries(test_trac_ik
        PRIVATE
            trac_ik
    )

    add_executable(test_cyton_ik
        tests/test_cyton_ik.cpp
        src/robot_calibration.cpp
    )

    target_include_directories(test_cyton_ik
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/external/trac_ik/trac_ik_lib/include
    )

    target_link_libraries(test_cyton_ik
        PRIVATE
            trac_ik
            ${orocos_kdl_LIBRARIES}
    )

    add_executable(test_cyton_ik_move
        tests/test_cyton_ik_move.cpp
        src/dynamixel_motor.cpp
        src/robot_calibration.cpp
    )

    target_include_directories(test_cyton_ik_move
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
            ${CMAKE_CURRENT_SOURCE_DIR}/external/trac_ik/trac_ik_lib/include
    )

    target_link_libraries(test_cyton_ik_move
        PRIVATE
            trac_ik
            ${orocos_kdl_LIBRARIES}
            dynamixel_sdk
    )

    add_executable(test_cyton_ik_target
        tests/test_cyton_ik_target.cpp
        src/robot_calibration.cpp
    )

    target_include_directories(test_cyton_ik_target
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/external/trac_ik/trac_ik_lib/include
    )

    target_link_libraries(test_cyton_ik_target
        PRIVATE
            trac_ik
            ${orocos_kdl_LIBRARIES}
    )

    add_executable(test_cyton_ik_target_move
        tests/test_cyton_ik_target_move.cpp
        src/dynamixel_motor.cpp
        src/robot_calibration.cpp
    )

    target_include_directories(test_cyton_ik_target_move
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
            ${CMAKE_CURRENT_SOURCE_DIR}/external/trac_ik/trac_ik_lib/include
    )

    target_link_libraries(test_cyton_ik_target_move
        PRIVATE
            trac_ik
            ${orocos_kdl_LIBRARIES}
            dynamixel_sdk
    )

else()

    message(STATUS
        "TRAC-IK not found; skipping IK targets"
    )

endif()

# ------------------------------------------------------------
# NDI API
# ------------------------------------------------------------

set(ndicapi_BUILD_APPLICATIONS OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON OFF CACHE BOOL "" FORCE)

if(EXISTS
    "${CMAKE_CURRENT_SOURCE_DIR}/external/ndicapi/CMakeLists.txt"
)

    add_subdirectory(
        external/ndicapi
        external/ndicapi_build
    )

    add_executable(test_five_pose_ndi_capture
        tests/test_five_pose_ndi_capture.cpp
        src/dynamixel_motor.cpp
        src/robot_calibration.cpp
    )

    add_executable(test_ndi_single_tool
        tests/test_ndi_single_tool.cpp
    )

    target_include_directories(test_ndi_single_tool
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/external/ndicapi
    )

    target_link_libraries(test_ndi_single_tool
        PRIVATE
            ndicapi
    )

    target_include_directories(test_five_pose_ndi_capture
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
            ${CMAKE_CURRENT_SOURCE_DIR}/external/ndicapi
    )

    target_link_libraries(test_five_pose_ndi_capture
        PRIVATE
            dynamixel_sdk
            ndicapi
    )

else()

    message(STATUS
        "ndicapi not found; skipping NDI capture target"
    )

endif()