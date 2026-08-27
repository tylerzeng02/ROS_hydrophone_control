# Cyton Gamma 1500 ROS and tFUS Acoustic Characterization Applications

Control and integration codebase for a **Cyton Gamma 1500** 7-DOF
Dynamixel-servo robot arm, kinematically calibrated against an **NDI
Polaris Spectra** optical tracker, and driven through **ROS 2 + MoveIt 2**
for real motion planning and execution.

`ros/src/fus_targeting_gui/` is the operator-facing application for tFUS acoustic characterization with this arm.

The native C++ hardware tools (`src/`, `tests/`, `calibration/collection/`)
build with plain CMake and can run on the machine connected to the arm and
tracker. The `ros/` workspace (ROS 2 + MoveIt 2) needs a Linux ROS 2
install with `colcon`.

## Quick setup

With ROS 2 Jazzy already installed ([official install docs](https://docs.ros.org/en/jazzy/Installation.html)):

```bash
./setup_system.sh          # one-time, needs sudo; add --hardware if connecting the real arm
./setup.sh                 # everything else: submodules, pymoveit2, deps, both builds
```

`setup.sh` needs no sudo and is safe to rerun (it skips work that's
already done and leaves incremental rebuilds to `colcon`/`cmake`). See the
sections below for what each step does and how to run them individually.

## Table of contents

- [Quick setup](#quick-setup)
- [Hardware](#hardware)
- [Repository layout](#repository-layout)
- [Software prerequisites](#software-prerequisites)
- [Getting started: native C++ build](#getting-started-native-c-build)
- [Getting started: ROS 2 / MoveIt workspace](#getting-started-ros-2--moveit-workspace)
- [Getting started: Python calibration tooling](#getting-started-python-calibration-tooling)
- [Getting started: fus_targeting_gui](#getting-started-fus_targeting_gui)
- [Typical workflow, in order](#typical-workflow-in-order)

## Hardware

- **Cyton Gamma 1500** arm, connected over USB-to-serial (Dynamixel bus,
  1,000,000 baud, Protocol 1.0). Motor IDs 0-6 are the arm joints
  (`shoulder_roll`, `shoulder_pitch`, `shoulder_yaw`, `elbow_pitch`,
  `elbow_yaw`, `wrist_pitch`, `wrist_roll`). Motor 7 is the gripper and is
  excluded from the calibrated IK chain.
- **NDI Polaris Spectra** optical tracker, connected over its own
  USB-to-serial adapter, with two passive marker rigid bodies (`.rom`
  geometry files). One is mounted on the arm's end effector (the "moving"
  tool) and one is fixed relative to the work area (the "fixed" tool). See
  `references/marker_mount.stl` for the moving-marker mounting bracket.
- A skull or phantom target, with a segmented surface mesh (STL) for
  `fus_targeting_gui` to load.

## Repository layout

- **`src/`**: core motor control (`dynamixel_motor.{h,cpp}`) and the
  per-joint calibration table (`robot_calibration.{h,cpp}`) that every
  motor-facing program in this repo goes through for safety clamping and
  tick/radian conversion.
- **`tests/`**: one standalone hardware-in-the-loop program per file
  (tick/radian checks, home-pose move, NDI single-tool diagnostic,
  multi-joint backlash test). Not a unit test suite; each is built and run
  individually against the physical robot and/or tracker.
- **`calibration/`**: the kinematic-calibration pipeline.
  - `collection/`: the C++ tools that talk to the hardware and collect
    data. `ndi_capture_and_validate.cpp` is the main NDI capture and
    `--validate` tool, used for every calibration dataset in this project.
    `record_hand_poses.cpp` records hand-posed joint configurations for
    later NDI capture.
  - `current/`: the active Python fitting/validation scripts for the
    deployed 48-parameter model (`calibrate_kinematics.py`,
    `final_deployment_fit.py`, `deployed_model_predictions.py`, and a few
    others). 
  - `data/`: the dataset the deployed model was fit on
    (`deployed_model_training_dataset_374pose.csv`). See
    `calibration/data/README.md`.
- **`ros/`**: the ROS 2 + MoveIt 2 workspace. `src/` holds the packages.
  - `cyton_description`: URDF/xacro and meshes for the arm.
  - `cyton_hardware`: the `ros2_control` hardware interface plugin
    bridging the Dynamixel servos to ROS 2/MoveIt.
  - `cyton_moveit_config`: MoveIt 2 configuration (SRDF, kinematics,
    planning, controllers) and the main `demo.launch.py`.
  - `cyton_bringup`: the single top-level launch entry point
    (`bringup.launch.py`), a thin wrapper around `cyton_moveit_config`'s
    launch file.
  - `cyton_trac_ik_kinematics_plugin`: an alternative TRAC-IK-based IK
    solver to the default KDL.
  - `cyton_ndi_capture`: `ndi_measure`, a standalone NDI measurement tool
    for checking MoveIt-commanded poses against independent tracker
    ground truth.
  - `cyton_pose_commander`: drives the arm through a CSV of tick-domain
    joint targets via MoveIt, with a plan-preview-then-confirm workflow.
  - `cyton_accuracy_check`: `run_accuracy_check`, a combined
    move-then-measure program that commands the arm via MoveIt and
    captures each pose's NDI measurement in one synchronized loop.
  - `fus_targeting_gui`: the skull-mesh point-picking targeting
    application. See its own `README.md` for setup and usage.

  Loose files at the top level of `ros/` are collected accuracy data.
- **`external/`**: vendored dependencies as git submodules
  (`DynamixelSDK`, `ndicapi`, `trac_ik`). `trac_ik` is only needed if you
  plan to build `cyton_trac_ik_kinematics_plugin`; the other two are
  needed for everything else (see
  [Software prerequisites](#software-prerequisites)).
- **`references/`**: the robot's URDF (`cyton_gamma_1500_trac_ik.urdf`)
  and the moving-marker mounting bracket CAD (`marker_mount.stl`).

## Software prerequisites

- CMake and a C++17 compiler toolchain, for the native hardware tools.
- ROS 2 (Jazzy or later) with MoveIt 2 and `colcon`, for the `ros/`
  workspace.
- `ros-<distro>-joint-state-broadcaster` and
  `ros-<distro>-joint-trajectory-controller`. These are not always
  installed by default; without them, `ros2_control`'s controllers fail
  to load.
- Git, with submodule support.
- [`uv`](https://docs.astral.sh/uv/) for the Python calibration scripts,
  or plain `pip` using `pyproject.toml`'s dependency list (`numpy`,
  `scipy`).

**Submodules:** clone with `--recurse-submodules`, or run
`git submodule update --init --recursive` after a plain clone, to
populate all three of `external/DynamixelSDK`, `external/ndicapi`, and
`external/trac_ik`. The root `CMakeLists.txt` and
`ros/src/cyton_hardware`'s `CMakeLists.txt` both stop with a
`FATAL_ERROR` if `external/DynamixelSDK` is missing. NDI and TRAC-IK
targets are optional and are skipped if their `external/` directory is
not populated, so `trac_ik` can be left uninitialized if you don't need
`cyton_trac_ik_kinematics_plugin`.

## Getting started: native C++ build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Every hardware-facing tool has its own device path and baud rate as
constants near the top of its `.cpp` file. Update these to match your
machine before running anything against real hardware.

There is no unit test framework. `tests/*.cpp` and
`calibration/collection/*.cpp` are standalone, hardware-in-the-loop
programs, each built into its own executable and run individually against
the physical robot and/or tracker, typically with interactive
Enter-to-proceed prompts between motion steps.

## Getting started: ROS 2 / MoveIt workspace

`fus_targeting_gui` depends on `pymoveit2`, which is not vendored in this
repo and is not pip-installable. Clone it into `ros/src/` before building:

```bash
git clone https://github.com/AndrejOrsula/pymoveit2.git ros/src/pymoveit2
```

```bash
cd ros
colcon build
source install/setup.bash
```

`fus_targeting_gui` is the one pure-Python (`ament_python`) package in
this workspace and needs its own PyPI dependencies installed first, since
`colcon`/`rosdep` do not manage third-party PyPI packages:

```bash
pip install -r ros/src/fus_targeting_gui/requirements.txt
```

See `ros/src/fus_targeting_gui/README.md`'s own "Setup" section for the
full sequence (including `rosdep install` and the venv workaround needed
on systems that block `pip install` outside a virtual environment).

Bring the whole stack up with a safe, hardware-free default (nothing
moves):

```bash
ros2 launch cyton_bringup bringup.launch.py
```

Against real hardware, with the arm connected, powered, and clear to
move:

```bash
ros2 launch cyton_bringup bringup.launch.py hardware_type:=real serial_port:=/dev/ttyUSB0
```

See `ros/src/cyton_bringup/launch/bringup.launch.py`'s own docstring for
every other launch argument (backlash compensation, uncalibrated A/B
comparison, TRAC-IK vs. KDL).

## Getting started: Python calibration tooling

```bash
uv run python calibration/current/calibrate_kinematics.py --selftest
```

`uv run` reads dependencies (`numpy`, `scipy`) straight from the root
`pyproject.toml`/`uv.lock`, no manual `--with` flags or virtual
environment setup needed. Without `uv`, `pip install numpy scipy` into
any Python 3.12+ environment works the same way.

`--selftest` runs against synthetic ground-truth data and needs no
hardware. Run this before trusting the script on real data. To refit
against the real deployed dataset:

```bash
uv run python calibration/current/final_deployment_fit.py
```

## Getting started: fus_targeting_gui

With `move_group` already running (see above), in a second terminal:

```bash
ros2 launch fus_targeting_gui targeting_gui.launch.py
```

Or run the node directly with `ros2 run fus_targeting_gui targeting_gui`.
See `ros/src/fus_targeting_gui/README.md` for the app's architecture and
`ros/src/fus_targeting_gui/config/default_config.yaml` for the one file
that would need editing to point this at a different, already-calibrated
robot arm.
