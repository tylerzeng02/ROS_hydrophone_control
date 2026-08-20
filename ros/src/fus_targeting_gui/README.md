# fus_targeting_gui

Standalone GUI for transcranial focused ultrasound (tFUS) acoustic
characterization: load a skull mesh, click a point on its surface, and
send the robot's end effector there via MoveIt (plan, preview, confirm,
execute), with every result logged to CSV.

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

Colcon/rosdep don't manage third-party PyPI packages for `ament_python`
packages — install this package's GUI/MoveIt-client dependencies directly
into the Python environment `ros2 run` uses:

```
pip install -r requirements.txt
```

Then build normally:

```
colcon build --packages-select fus_targeting_gui
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

- `mesh_view.py` — mesh loading + rendering + surface point-picking
  (PyVista/`pyvistaqt`). No ROS dependency.
- `registration.py` — converts a picked mesh-local point+normal into a
  target `geometry_msgs/Pose` in the robot's base frame. Isolated behind
  the `Registration` interface (`FixedPoseRegistration` is the only
  implementation today — a fixed, hand-specified mesh-to-`base_frame`
  pose, the same one `ros/publish_skull_marker.py` already uses to place
  this mesh in RViz) so a future registration scheme (e.g. NDI-fiducial-
  based) can be swapped in without touching `mesh_view.py` or
  `moveit_bridge.py`.
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
