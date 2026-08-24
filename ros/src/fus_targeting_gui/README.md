# fus_targeting_gui

Standalone GUI for transcranial focused ultrasound (tFUS) acoustic
characterization: load a skull mesh, click a point on its surface, and
send the robot's end effector there via MoveIt (plan, preview, confirm,
execute), with every result logged to CSV.

Each picked target has its own **Alignment Parameters**, editable at any
time (re-selecting an earlier target restores its own saved values):

- **Standoff distance** — how far the probe stops short of the picked
  point, measured along the approach axis (not simply along the surface
  normal, so this stays exact even when tilted).
- **Tilt from normal** — angle of the approach axis away from the surface
  normal (0 = straight-on).
- **Tilt azimuth** — which direction around the normal the tilt leans.
- **Probe roll** — rotation of the probe about its own approach axis.

"Reset to Surface Normal" zeros tilt/azimuth/roll in one click. A live
preview (yellow shaft + red/green/blue orientation frame) updates in the
3D view as these are adjusted, before anything is planned or executed.

**Robot-agnostic by design.** This package contains no robot-specific
logic anywhere in its `.py` files — planning group, base/tip frame, and
joint names all come from `config/default_config.yaml`. Pointing this at
a different, already-calibrated MoveIt-configured arm should only ever
require editing that one file's `robot:` block.

This is the first pure-Python (`ament_python`) package in this workspace
— everything else under `ros/src/*` is C++ (`ament_cmake`) using
`rclcpp`/`MoveGroupInterface` directly. This package talks to MoveIt via
[`pymoveit2`](https://github.com/AndrejOrsula/pymoveit2) instead, since
`moveit_py` isn't installed here — `pymoveit2` wraps the same standard
`/move_action` (`moveit_msgs/action/MoveGroup`) interface
`MoveGroupInterface` itself calls internally, so it works with any
MoveIt-configured robot without compiled bindings.

## Setup

**1. `pymoveit2` — clone it into this workspace, don't pip install it.**
Despite `moveit_bridge.py`'s docstring referencing its documented pip
install, `pymoveit2`'s current repo layout (`CMakeLists.txt`/`package.xml`
at the root, no `setup.py`/`pyproject.toml`) is an `ament_cmake` ROS 2
package, not a pip-installable one:

```
cd ~/dev/cyton_setup/ros/src
git clone https://github.com/AndrejOrsula/pymoveit2.git
```

It'll get built automatically the next time you `colcon build` this
workspace (declared as an `exec_depend` in `package.xml`, so colcon builds
it first). It has its own rosdep-resolvable dependencies — covered by the
`rosdep install` below.

**2. Everything else** — third-party PyPI packages (`PySide6`, `pyvista`,
`pyvistaqt`, `PyYAML`) that colcon/rosdep don't manage for `ament_python`
packages, plus `tf_transformations` via rosdep. From this package's
directory:

```
rosdep install --from-paths . --ignore-src -r -y
pip install -r requirements.txt
```

If `pip install` fails with `externally-managed-environment` (PEP 668, on
newer Ubuntu), use a venv that can still see the system-installed ROS
packages:

```
python3 -m venv --system-site-packages ~/fus_gui_venv
source ~/fus_gui_venv/bin/activate
pip install -r requirements.txt
```

Then re-activate that venv (`source ~/fus_gui_venv/bin/activate`) any time
you want to run the GUI, alongside sourcing the workspace's own
`install/setup.bash`. `colcon build` itself doesn't need the venv active.

**3. Build:**

```
colcon build --packages-select pymoveit2 fus_targeting_gui
source install/setup.bash
```

## Running

`move_group` has to already be up (this package only *talks* to MoveIt,
it doesn't launch it):

```
# terminal 1
ros2 launch cyton_bringup bringup.launch.py          # mock_components is fine, no hardware needed
# terminal 2
ros2 launch fus_targeting_gui targeting_gui.launch.py
```

Or run the node directly: `ros2 run fus_targeting_gui targeting_gui`.

`mesh_view.py` is also runnable standalone with zero ROS dependency, to
test mesh loading/point-picking in isolation:

```
python3 -m fus_targeting_gui.mesh_view /path/to/mesh.stl [scale]
```

## Layout

- `geometry_utils.py` — pure-numpy direction/rotation math (tilt-off-axis,
  look-at basis with roll) shared by `mesh_view.py`'s preview and
  `registration.py`'s actual pose computation. No ROS dependency, kept
  separate from `registration.py` specifically so `mesh_view.py` doesn't
  have to import `tf_transformations`/`geometry_msgs` to draw a preview.
- `mesh_view.py` — mesh loading + rendering + surface point-picking +
  live alignment preview (PyVista/`pyvistaqt`). No ROS dependency.
- `registration.py` — converts a picked mesh-local point+normal (plus
  per-target tilt/azimuth/roll/standoff) into a target `geometry_msgs/Pose`
  in the robot's base frame. Isolated behind the `Registration` interface
  (`FixedPoseRegistration` is the only implementation today — a fixed,
  hand-specified mesh-to-`base_frame` pose, the same one
  `ros/publish_skull_marker.py` already uses to place this mesh in RViz)
  so a future registration scheme (e.g. NDI-fiducial-based) can be swapped
  in without touching `mesh_view.py` or `moveit_bridge.py`.
- `moveit_bridge.py` — thin `pymoveit2.MoveIt2` wrapper, built entirely
  from `RobotConfig`. Splits plan and execute into two calls so the GUI
  can show a preview and require operator confirmation before anything
  moves.
- `config.py` — loads `config.yaml` into the small typed structures every
  other module reads through.
- `main_window.py` — wires the above together: mesh view, target list,
  Plan/Execute buttons, per-session CSV log
  (`fus_targeting_session_<timestamp>.csv`).
- `node_entry.py` — `ros2 run` entry point (`rclpy.init`, builds the
  bridge, runs the Qt event loop).

## Known open item

`registration.py`'s `quaternion_looking_along()` assumes the end
effector's local **+Z** axis is the one that should point along the probe's
approach direction — this depends on how the probe is physically mounted
on `virtual_endeffector`, which couldn't be verified from this repo alone.
Confirm against the real mount (e.g. compare a planned pose in RViz
against the physical probe orientation) before trusting a computed
orientation on real hardware.
