# trac_ik_lib (vendored, modified)

Origin: `trac_ik_lib`, part of [HIRO-group/trac_ik](https://github.com/HIRO-group/trac_ik)
(originally traclabs/trac_ik), BSD-licensed.

Only the ROS-independent core IK solver files are vendored here
(`kdl_tl.cpp`/`.hpp`, `nlopt_ik.cpp`/`.hpp`, `trac_ik.cpp`/`.hpp`,
`dual_quaternion.h`, `math3d.h`). `trac_ik_ros.cpp`/`.hpp` (the ROS1/catkin
wrapper) is not included, since `cyton_trac_ik_kinematics_plugin` is its
own ROS2-native wrapper around this core solver and never needed it.

**Modified from upstream:** `trac_ik.cpp`, `trac_ik.hpp`, `nlopt_ik.cpp`,
and `kdl_tl.cpp` all have local changes removing ROS1/catkin dependencies
so this core solver builds standalone in a ROS2 package. These files are
no longer in sync with, and should not be assumed equivalent to, upstream
`trac_ik_lib`.

These files used to be compiled directly from a `trac_ik` git submodule
at `external/trac_ik`, so the patched version only ever existed in one
local checkout and was never committed anywhere git could track it,
meaning a fresh clone of this repo could not actually build this plugin.
Vendoring the already-patched files directly here fixes that, and is a
better fit than a maintained fork: this is no longer meaningfully "the
same as upstream plus tracking", it is a working, patched fork of a
handful of files.
