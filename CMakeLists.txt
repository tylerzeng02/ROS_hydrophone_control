cmake_minimum_required(VERSION 3.22.6)
project(cyton_setup)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dynamixel SDK

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

# Common include directories

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
)

# Main program

add_executable(cyton_setup
    src/main.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_link_libraries(cyton_setup
    PRIVATE
        dynamixel_sdk
)

# Basic Dynamixel tests

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

# One-off recovery utility (2026-08-07, see the .cpp's own comment) --
# not meant to be permanent, safe to remove once no longer needed.
add_executable(recover_motor4
    tests/recover_motor4.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(recover_motor4
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
)

target_link_libraries(recover_motor4
    PRIVATE
        dynamixel_sdk
)

add_executable(disable_torque_prompt
    tests/disable_torque_prompt.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(disable_torque_prompt
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
)

target_link_libraries(disable_torque_prompt
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

add_executable(record_hand_poses_multijoint_backlash_test
    tests/record_hand_poses_multijoint_backlash_test.cpp
    src/dynamixel_motor.cpp
    src/robot_calibration.cpp
)

target_include_directories(record_hand_poses_multijoint_backlash_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
)

target_link_libraries(record_hand_poses_multijoint_backlash_test
    PRIVATE
        dynamixel_sdk
)

# Renamed from record_hand_poses_fixed_elbow_yaw 2026-08-20, moved out of
# tests/ into calibration/collection/ -- this is the general-purpose hand-
# pose recorder (locks elbow_yaw, records the other 6 joints on Enter) that
# produced the deployed model's training data; not to be confused with
# tests/record_hand_poses_multijoint_backlash_test.cpp, a different,
# special-purpose tool despite the similar name.
add_executable(record_hand_poses
    calibration/collection/record_hand_poses.cpp
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

# TRAC-IK targets
#
# Archived 2026-08-20 (tests/archive/test_trac_ik.cpp, test_cyton_ik.cpp,
# test_cyton_ik_move.cpp, test_cyton_ik_target.cpp,
# test_cyton_ik_target_move.cpp) -- external/trac_ik was empty on this
# checkout at the time and these 5 targets were silently skipped by the
# existence-guard that used to live here. If TRAC-IK is populated again and
# these are wanted back, restore this block from git history (see the
# commit that removed it) alongside the archived .cpp files.

# NDI API

set(ndicapi_BUILD_APPLICATIONS OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON OFF CACHE BOOL "" FORCE)

if(EXISTS
    "${CMAKE_CURRENT_SOURCE_DIR}/external/ndicapi/CMakeLists.txt"
)

    add_subdirectory(
        external/ndicapi
        external/ndicapi_build
    )

    # Renamed from test_five_pose_ndi_capture 2026-08-20, moved out of
    # tests/ into calibration/collection/ -- the "five_pose" name was a
    # historical artifact (it started as a 5-pose demo and grew into the
    # project's main pose-list-driven NDI capture/validation tool, with
    # --quick-test, --validate, and --settling-diagnostic modes). This is
    # the tool behind every calibration dataset in this project.
    add_executable(ndi_capture_and_validate
        calibration/collection/ndi_capture_and_validate.cpp
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

    target_include_directories(ndi_capture_and_validate
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/external/DynamixelSDK/c++/include
            ${CMAKE_CURRENT_SOURCE_DIR}/external/ndicapi
    )

    target_link_libraries(ndi_capture_and_validate
        PRIVATE
            dynamixel_sdk
            ndicapi
    )

else()

    message(STATUS
        "ndicapi not found; skipping NDI capture target"
    )

endif()