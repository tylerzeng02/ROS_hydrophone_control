#!/usr/bin/env bash
#
# One-time setup for the Linux/ROS 2 side of this project. Populates the two
# submodules that have no .gitmodules entry, fixes the FTDI serial-latency
# and dialout-group issues that block real hardware control, installs the
# ROS 2 controller packages MoveIt needs to execute (not just plan) a
# trajectory, and runs the first build.
#
# This script does NOT install ROS 2 or MoveIt 2 themselves. Those are
# large, versioned installs with their own official instructions, and this
# script only checks that they are already present.
#
# Usage: ./scripts/setup_linux.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== Checking prerequisites =="

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1"
        echo "$2"
        exit 1
    fi
}

require_cmd git   "Install git before continuing."
require_cmd cmake "Install cmake (e.g. 'sudo apt install cmake')."
require_cmd colcon "Install colcon (e.g. 'sudo apt install python3-colcon-common-extensions')."

if [ -z "${ROS_DISTRO:-}" ]; then
    echo "ROS_DISTRO is not set. Source your ROS 2 install first, e.g.:"
    echo "  source /opt/ros/jazzy/setup.bash"
    exit 1
fi
echo "Found ROS 2 distro: $ROS_DISTRO"

echo
echo "== Populating submodules =="

# ndicapi has a proper .gitmodules entry.
git submodule update --init external/ndicapi

# DynamixelSDK and trac_ik are submodule references with no .gitmodules
# entry, so a plain submodule update cannot bring them in. Clone them
# directly and pin to the exact commit already recorded in this repo's
# tree, so the checkout matches what the rest of the code was built
# against.
populate_bare_submodule() {
    local path="$1"
    local url="$2"
    local pinned_commit
    pinned_commit="$(git ls-tree HEAD "$path" | awk '{print $3}')"

    if [ -n "$(ls -A "$path" 2>/dev/null)" ]; then
        echo "$path is already populated, skipping."
        return
    fi

    echo "Cloning $url into $path (pinned to $pinned_commit)..."
    rm -rf "$path"
    git clone "$url" "$path"
    (cd "$path" && git checkout "$pinned_commit")
}

populate_bare_submodule "external/DynamixelSDK" "https://github.com/ROBOTIS-GIT/DynamixelSDK"
populate_bare_submodule "external/trac_ik" "https://github.com/traclabs/trac_ik"

echo
echo "== Serial port access (dialout group) =="

if id -nG "$USER" | grep -qw dialout; then
    echo "$USER is already in the dialout group."
else
    echo "Adding $USER to the dialout group (needed to open the arm and tracker's serial ports)."
    sudo usermod -aG dialout "$USER"
    echo "NOTE: this only takes effect in a NEW terminal (or after 'newgrp dialout'), not this one."
fi

echo
echo "== FTDI latency timer (needed for real hardware control) =="

UDEV_RULE_PATH="/etc/udev/rules.d/99-ftdi-latency.rules"
UDEV_RULE_CONTENT='ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"'

if [ -f "$UDEV_RULE_PATH" ] && grep -qF "$UDEV_RULE_CONTENT" "$UDEV_RULE_PATH"; then
    echo "FTDI latency-timer udev rule already installed."
else
    echo "Installing FTDI latency-timer udev rule (fixes the control loop missing its cycle budget)."
    echo "$UDEV_RULE_CONTENT" | sudo tee "$UDEV_RULE_PATH" >/dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo "NOTE: unplug and replug the arm's and tracker's USB adapters for this to take effect."
fi

echo
echo "== ROS 2 controller packages =="

MISSING_PKGS=()
for pkg in "ros-${ROS_DISTRO}-joint-state-broadcaster" "ros-${ROS_DISTRO}-joint-trajectory-controller"; do
    if ! dpkg -l "$pkg" >/dev/null 2>&1; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [ ${#MISSING_PKGS[@]} -eq 0 ]; then
    echo "joint_state_broadcaster and joint_trajectory_controller are already installed."
else
    echo "Installing: ${MISSING_PKGS[*]}"
    sudo apt update
    sudo apt install -y "${MISSING_PKGS[@]}"
fi

echo
echo "== fus_targeting_gui Python dependencies =="

PYTHON_DEPS=(PySide6 pyvista pyvistaqt)
MISSING_PY=()
for dep in "${PYTHON_DEPS[@]}"; do
    if ! python3 -c "import ${dep}" >/dev/null 2>&1; then
        MISSING_PY+=("$dep")
    fi
done

if [ ${#MISSING_PY[@]} -eq 0 ]; then
    echo "PySide6, pyvista, and pyvistaqt are already installed."
else
    echo "Installing: ${MISSING_PY[*]}"
    pip3 install --user "${MISSING_PY[@]}"
fi

# pymoveit2 is a ROS 2 source package (built by colcon), not a pip package,
# so it needs to live under ros/src/ like the other packages.
if [ ! -d "ros/src/pymoveit2" ]; then
    echo "Cloning pymoveit2 into ros/src/ (required by fus_targeting_gui's MoveIt bridge)."
    git clone https://github.com/AndrejOrsula/pymoveit2.git ros/src/pymoveit2
else
    echo "pymoveit2 already present in ros/src/."
fi

echo
echo "== Building the native C++ library and test programs =="
cmake -S . -B build
cmake --build build

echo
echo "== Building the ROS 2 workspace =="
(cd ros && colcon build)

echo
echo "== Done =="
echo "Open a new terminal (so the dialout group membership applies), then:"
echo "  source /opt/ros/${ROS_DISTRO}/setup.bash"
echo "  cd ros && source install/setup.bash"
echo "  ros2 launch cyton_bringup bringup.launch.py"
