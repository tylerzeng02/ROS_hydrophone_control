# cyton_setup

C++17 control/integration codebase for a **Cyton Gamma 1500** 7-DOF
Dynamixel-servo robot arm, with an **NDI Polaris Spectra** optical tracker
integration used to kinematically calibrate the arm against real,
independently-measured ground truth. Includes a ROS 2 + MoveIt 2
integration (`ros/`) for real motion planning and execution on the
physical hardware.

The application driving this work: using the arm to position a hydrophone
probe against a skull, to validate that an ultrasound helmet is actually
directing energy where intended — an alternative to gantry-style
positioning systems, trading some raw precision for reach into positions
a gantry can't access.

## Build (native, C++ hardware tools)

Requires an MSVC developer environment on `PATH` (NMake Makefiles
generator) — build from a **Developer Command Prompt for VS 2022**, or
source `vcvars64.bat` first in another shell:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd build
nmake -f Makefile
```

To (re)configure the build directory from scratch:

```
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Debug
```

There's no unit test framework — `tests/*.cpp` are standalone,
hardware-in-the-loop programs, each built and run individually against
the physical robot/tracker.

## Layout

- **`src/`** — core motor control (`dynamixel_motor.{h,cpp}`) and the
  per-joint calibration table (`robot_calibration.{h,cpp}`) every
  motor-facing program goes through.
- **`tests/`** — one standalone hardware-in-the-loop program per file
  (tick/radian checks, NDI capture, calibration data collection, etc.) —
  not a unit test suite. I-gain/PID-tuning-specific tools live in
  `pid_tuning/tests/` instead, not here.
- **`pid_tuning/`** — everything related to the shoulder_pitch I-gain
  investigation, consolidated into one place: `tests/` (C++ tools),
  `scripts/` (Python analysis + PowerShell sweep automation), `data/`
  (collected pose sets and sweep results). See `pid_tuning/README.md`.
- **`calibration/`** — the Python kinematic-calibration pipeline
  (`current/` is the active fitting/analysis code for the deployed
  48-param model; `archive/` holds ~110 one-off diagnostic scripts plus
  the superseded 60-param model in `archive/60param_model/`).
- **`ros/`** — the ROS 2 + MoveIt 2 workspace (`src/` holds the real
  packages: hardware interface, MoveIt config, NDI capture/accuracy-check
  tools, and `fus_targeting_gui` — the skull-mesh point-picking targeting
  application, see that package's own README; loose files at the top
  level are real collected accuracy data, see `ros/archive/README.md` for
  the superseded ones).
- **`external/`** — vendored dependencies (DynamixelSDK, ndicapi,
  trac_ik) as git submodule links.
- **`references/`** — the robot's URDF and related reference docs.

`trash/` and `understanding/` exist locally but are gitignored (not part
of the tracked repo) — the former is discarded material kept around only
as a safety net, the latter a presentation-specific reference package;
neither is needed to build or run anything here.

## Full project history

**`CLAUDE.md`** is this project's detailed, continuously-updated decision
log — every real bug found, every calibration result, every dead end
ruled out, and why. It's long, but it's the authoritative source for
"why is this the way it is" on anything not obvious from the code itself.
