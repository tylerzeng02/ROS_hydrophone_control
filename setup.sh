#!/usr/bin/env bash
# Repo-level setup: everything after ./setup_system.sh that does not need
# sudo. Safe to rerun. Already-completed steps are skipped or left to
# colcon/cmake's own incremental build, so a rerun after a small change is
# fast, not a full rebuild from scratch.
#
# Requires ROS 2 Jazzy already installed and sourceable (this script
# sources /opt/ros/jazzy/setup.bash itself if ROS_DISTRO isn't already
# set, so you do not need to source it yourself first).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

JOBS="${JOBS:-$(nproc)}"

echo "==> [1/6] Git submodules (DynamixelSDK, ndicapi, trac_ik)..."
git submodule update --init --recursive

echo "==> [2/6] pymoveit2 (not vendored, cloned separately)..."
if [[ -d ros/src/pymoveit2/.git ]]; then
  echo "    Already present, skipping clone."
else
  git clone https://github.com/AndrejOrsula/pymoveit2.git ros/src/pymoveit2
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
  echo "==> Sourcing /opt/ros/jazzy/setup.bash (ROS_DISTRO was not already set)..."
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
fi

echo "==> [3/6] ROS-level dependencies (rosdep)..."
if ! command -v rosdep &> /dev/null; then
  echo "    rosdep not found. Install it and run 'sudo rosdep init' once, then rerun this script."
  exit 1
fi
rosdep update
rosdep install --from-paths ros/src --ignore-src -r -y

echo "==> [4/6] fus_targeting_gui's Python dependencies..."
REQUIREMENTS=ros/src/fus_targeting_gui/requirements.txt
if pip install -r "$REQUIREMENTS" 2>/tmp/pip_install_err.log; then
  :
elif grep -q "externally-managed-environment" /tmp/pip_install_err.log; then
  echo "    System pip is externally managed. Using a venv at ~/fus_gui_venv instead."
  if [[ ! -d ~/fus_gui_venv ]]; then
    python3 -m venv --system-site-packages ~/fus_gui_venv
  fi
  # shellcheck disable=SC1091
  source ~/fus_gui_venv/bin/activate
  pip install -r "$REQUIREMENTS"
  echo "    Remember to 'source ~/fus_gui_venv/bin/activate' before running the GUI later."
else
  cat /tmp/pip_install_err.log
  exit 1
fi

echo "==> [5/6] Building the ROS workspace (colcon, $JOBS parallel workers)..."
(
  cd ros
  python3 -m colcon build --parallel-workers "$JOBS"
)

echo "==> [6/6] Building the native C++ tools (cmake, $JOBS parallel jobs)..."
cmake -S . -B build
cmake --build build -j"$JOBS"

cat <<EOF

==> Setup complete.

Next steps:
  source ros/install/setup.bash
  ros2 launch cyton_bringup bringup.launch.py   # mock hardware, nothing moves
  uv run python calibration/current/calibrate_kinematics.py --selftest
EOF
