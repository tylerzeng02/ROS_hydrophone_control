# cyton_setup

Control and integration codebase for a **Cyton Gamma 1500** 7-DOF
Dynamixel-servo robot arm, kinematically calibrated against an **NDI
Polaris Spectra** optical tracker, and driven through **ROS 2 + MoveIt 2**
for real motion planning and execution.

**The application:** positioning a hydrophone probe against a skull to
validate that a transcranial focused ultrasound (tFUS) helmet is actually
directing energy where intended. This is an alternative to gantry-style
positioning systems, trading some raw precision for reach into positions a
gantry cannot access. `ros/src/fus_targeting_gui/` is the operator-facing
application for this: load a skull mesh, click a target point on its
surface, and send the probe there via MoveIt.

This repo spans two machines in practice. The native C++ tools
(`src/`, `tests/`, `calibration/collection/`) are built and run against
real hardware from Windows, using MSVC/NMake. The `ros/` workspace (ROS 2
+ MoveIt 2) needs a Linux ROS 2 install and has only ever been built and
run there. Both halves are described below.

## Table of contents

- [Hardware](#hardware)
- [Repository layout](#repository-layout)
- [Software prerequisites](#software-prerequisites)
- [Getting started: native C++ build (Windows)](#getting-started-native-c-build-windows)
- [Getting started: ROS 2 / MoveIt workspace (Linux)](#getting-started-ros-2--moveit-workspace-linux)
- [Getting started: Python calibration tooling](#getting-started-python-calibration-tooling)
- [Getting started: fus_targeting_gui](#getting-started-fus_targeting_gui)
- [Typical workflow, in order](#typical-workflow-in-order)
- [Key concepts and safety notes](#key-concepts-and-safety-notes)
- [Full project history](#full-project-history)

## Hardware

- **Cyton Gamma 1500** arm, connected over USB-to-serial (Dynamixel bus,
  1,000,000 baud, Protocol 1.0). Motor IDs 0-6 are the arm joints
  (`shoulder_roll`, `shoulder_pitch`, `shoulder_yaw`, `elbow_pitch`,
  `elbow_yaw`, `wrist_pitch`, `wrist_roll`); motor 7 is the gripper and is
  excluded from the calibrated IK chain.
- **NDI Polaris Spectra** optical tracker, connected over its own
  USB-to-serial adapter, with two passive marker rigid bodies (`.rom`
  geometry files): one mounted on the arm's end effector (the "moving"
  tool) and one fixed relative to the work area (the "fixed" tool). See
  `references/marker_mount.stl` for the moving-marker mounting bracket.
- A skull or phantom target, with a segmented surface mesh (STL) for
  `fus_targeting_gui` to load.

## Repository layout

- **`src/`** — core motor control (`dynamixel_motor.{h,cpp}`) and the
  per-joint calibration table (`robot_calibration.{h,cpp}`) that every
  motor-facing program in this repo goes through for safety clamping and
  tick/radian conversion. `src/archive/` holds retired modules kept
  locally for reference (not tracked in git).
- **`tests/`** — one standalone hardware-in-the-loop program per file
  (tick/radian checks, home-pose move, NDI single-tool diagnostic,
  multi-joint backlash test). Not a unit test suite; each is built and run
  individually against the physical robot and/or tracker. `tests/archive/`
  holds retired tools.
- **`calibration/`** — the kinematic-calibration pipeline:
  - `collection/` — the C++ tools that actually talk to the hardware and
    collect data: `ndi_capture_and_validate.cpp` (the main NDI capture and
    `--validate` tool, the source of every calibration dataset in this
    project) and `record_hand_poses.cpp` (records hand-posed joint
    configurations for later NDI capture).
  - `current/` — the active Python fitting/validation scripts for the
    deployed 48-parameter model (`calibrate_kinematics.py`,
    `final_deployment_fit.py`, `deployed_model_predictions.py`, and a few
    others). This is what you touch to refit or re-validate the
    calibration.
  - `data/` — the real, tracked dataset the deployed model was fit on
    (`deployed_model_training_dataset_374pose.csv`). See
    `calibration/data/README.md`.
  - `archive/` — superseded diagnostic scripts and the earlier 60-param
    research model, kept locally for reference (not tracked in git).
- **`ros/`** — the ROS 2 + MoveIt 2 workspace. `src/` holds the real
  packages:
  - `cyton_description` — URDF/xacro and meshes for the arm.
  - `cyton_hardware` — the `ros2_control` hardware interface plugin
    bridging the Dynamixel servos to ROS 2/MoveIt.
  - `cyton_moveit_config` — MoveIt 2 configuration (SRDF, kinematics,
    planning, controllers) and the main `demo.launch.py`.
  - `cyton_bringup` — the single top-level launch entry point
    (`bringup.launch.py`), a thin wrapper around `cyton_moveit_config`'s
    launch file.
  - `cyton_trac_ik_kinematics_plugin` — an alternative TRAC-IK-based IK
    solver (KDL is the default; this is newer and less validated).
  - `cyton_ndi_capture` — `ndi_measure`, a standalone NDI measurement tool
    for checking MoveIt-commanded poses against independent tracker
    ground truth.
  - `cyton_pose_commander` — drives the arm through a CSV of tick-domain
    joint targets via MoveIt, with a plan-preview-then-confirm workflow.
  - `cyton_accuracy_check` — `run_accuracy_check`, a combined
    move-then-measure program that commands the arm via MoveIt and
    captures each pose's NDI measurement in one synchronized loop.
  - `fus_targeting_gui` — the skull-mesh point-picking targeting
    application. See its own `README.md` for setup and usage.

  Loose files at the top level of `ros/` are real collected accuracy
  data; see `ros/archive/README.md` for the superseded ones.
- **`external/`** — vendored dependencies as git submodules:
  `DynamixelSDK` (populated), `ndicapi` (populated), `trac_ik` (only
  needed if you plan to build `cyton_trac_ik_kinematics_plugin`; empty in
  many checkouts, since a plain `git submodule update --init --recursive`
  does not populate it, see [Software prerequisites](#software-prerequisites)).
- **`references/`** — the robot's URDF (`cyton_gamma_1500_trac_ik.urdf`)
  and the moving-marker mounting bracket CAD (`marker_mount.stl`).
- **`pid_tuning/`, `notes/`, `trash/`, `understanding/`** — local-only
  material, gitignored and not part of the tracked repo. Never needed to
  build or run anything here.

## Software prerequisites

**Windows (native C++ build):**
- Visual Studio 2022 Build Tools, with the "Desktop development with C++"
  workload (provides MSVC and NMake).
- CMake.

**Linux (ROS 2 workspace):**
- ROS 2 (Jazzy or later) with MoveIt 2 installed.
- `colcon`.
- `ros-<distro>-joint-state-broadcaster` and
  `ros-<distro>-joint-trajectory-controller` (not always installed by
  default; without these, `ros2_control`'s controllers fail to load).

**Both machines:**
- Git, with submodule support.
- [`uv`](https://docs.astral.sh/uv/) for the Python calibration scripts
  (or plain `pip`, using `pyproject.toml`'s dependency list: `numpy`,
  `scipy`).

**Submodules:** only `external/ndicapi` has a real `.gitmodules` entry.
`external/DynamixelSDK` and `external/trac_ik` are submodule references
with no `.gitmodules` entry, so `git submodule update --init --recursive`
will not populate them on a fresh clone. If either is missing, populate it
manually (clone `ROBOTIS-GIT/DynamixelSDK` or `traclabs/trac_ik` into the
corresponding `external/` directory at the commit this repo's git tree
references). The root `CMakeLists.txt` and `ros/src/cyton_hardware`'s
`CMakeLists.txt` both guard on `external/DynamixelSDK` existing (a
`FATAL_ERROR` if it is missing); NDI and TRAC-IK targets are guarded more
gently and are simply skipped if their corresponding `external/`
directory is not populated.

## Getting started: native C++ build (Windows)

Build from a **Developer Command Prompt for VS 2022**, or source
`vcvars64.bat` first in another shell:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd build
nmake -f Makefile
```

To (re)configure the build directory from scratch, or after adding/moving
a source file referenced in `CMakeLists.txt`:

```
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Debug
```

Every hardware-facing tool has its own device path and baud rate as
constants near the top of its `.cpp` file (Dynamixel bus on `COM4`, NDI
tracker on `COM3`, by convention in this project) — update these to match
your machine before running anything against real hardware.

There is no unit test framework. `tests/*.cpp` and
`calibration/collection/*.cpp` are standalone, hardware-in-the-loop
programs, each built into `build/<name>.exe` and run individually against
the physical robot and/or tracker, typically with interactive
Enter-to-proceed prompts between motion steps.

## Getting started: ROS 2 / MoveIt workspace (Linux)

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

Bring the whole stack up (safe, hardware-free default — nothing moves):

```bash
ros2 launch cyton_bringup bringup.launch.py
```

Against real hardware (arm connected, powered, and clear to move):

```bash
ros2 launch cyton_bringup bringup.launch.py hardware_type:=real serial_port:=/dev/ttyUSB0
```

See `ros/src/cyton_bringup/launch/bringup.launch.py`'s own docstring for
every other launch argument (backlash compensation, uncalibrated A/B
comparison, TRAC-IK vs. KDL).

## Getting started: Python calibration tooling

```bash
uv run --with numpy --with scipy python calibration/current/calibrate_kinematics.py --selftest
```

`--selftest` runs against synthetic ground-truth data and needs no
hardware — run this before trusting the script on real data. To refit
against the real deployed dataset:

```bash
uv run python calibration/current/final_deployment_fit.py
```

## Getting started: fus_targeting_gui

With `move_group` already running (see above), in a second terminal:

```bash
ros2 launch fus_targeting_gui targeting_gui.launch.py
```

Or run the node directly: `ros2 run fus_targeting_gui targeting_gui`. See
`ros/src/fus_targeting_gui/README.md` for the app's architecture and
`ros/src/fus_targeting_gui/config/default_config.yaml` for the one file
that would need editing to point this at a different, already-calibrated
robot arm.

## Typical workflow, in order

1. **Read-only sanity check.** `read_motor_positions.exe` connects, prints
   every motor's current position, and disconnects. Safe to run at any
   time, including while the arm is being hand-posed.
2. **Home pose.** `test_home_pose.exe` enables torque and holds the arm at
   its current pose — a basic "does the arm respond at all" check.
3. **NDI connectivity.** `test_ndi_single_tool.exe` confirms the tracker
   can see one marker before attempting a real capture session.
4. **Calibration capture.** `ndi_capture_and_validate.exe` (built from
   `calibration/collection/`) drives the arm through a pose list and
   captures paired NDI measurements — this is the tool behind every
   calibration dataset in this project. `--quick-test` and `--validate`
   are its other modes; see its own header comment for the full CLI.
5. **Refit or validate the calibration**, using the scripts in
   `calibration/current/`, against the data `ndi_capture_and_validate`
   collected.
6. **Bring up ROS 2 / MoveIt** (`cyton_bringup`), with `hardware_type`
   left at its safe `mock_components` default until you are ready to move
   the real arm.
7. **Drive the arm through MoveIt** — via RViz directly, `cyton_pose_commander`
   for a CSV of joint targets, or `fus_targeting_gui` for the real
   click-a-point-on-the-skull workflow.

## Key concepts and safety notes

- **`src/robot_calibration.cpp`'s `jointCalibrations` table is the single
  source of truth** for per-joint zero points, gear-ratio scale, and
  safety tick limits. Every motor-facing program goes through it — joint
  limits are never duplicated per file.
- **`elbow_yaw` (motor 4) is deliberately locked** to a narrow ~4-degree
  window in production. It is the arm's single worst-measured backlash
  joint; excluding it from motion planning measurably improved the other
  6 joints' calibration accuracy. This is a real, permanent 6-DOF-not-7
  trade-off, not a bug.
- **Backlash compensation** (`moveJointSafely`/`moveJointsSafely` in
  `dynamixel_motor.cpp`) is on by default for every blocking move: a joint
  that would otherwise arrive by decreasing its tick value first
  overshoots below the target, so every move finishes by increasing,
  converting direction-dependent backlash into an ordinary, correctable
  constant offset.
- **Torque is persistent servo-firmware state.** Killing a process
  (including Ctrl+C, which none of these programs catch) does not drop
  torque — the arm keeps holding its last commanded position. Torque only
  releases via an explicit disable call.
- **Windows console "Mark" mode looks exactly like a hang.** Clicking into
  a PowerShell/cmd window while text is scrolling pauses the process
  entirely until you press Esc or right-click to deselect. Check for this
  before assuming a real freeze.
- **Accuracy is a relative, not absolute, measurement.** Every NDI-based
  accuracy figure in this project measures the moving marker's pose
  relative to the fixed marker. The joint-level calibration corrections
  transfer directly to a real deployment; the fitted base-frame transform
  does not, since it is anchored to the calibration rig, not the robot's
  true mounting location.
- **Only run `git commit`/`git push` yourself.** This is a standing
  project convention — changes get staged and handed back for you to
  commit, never committed automatically.

## Full project history

**`CLAUDE.md`** is this project's detailed, continuously-updated decision
log — every real bug found, every calibration result, every dead end
ruled out, and why. It is long, but it is the authoritative source for
"why is this the way it is" on anything not obvious from the code itself.
