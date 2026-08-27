#!/usr/bin/env bash
# One-time SYSTEM-level setup for this repo. Requires sudo. Review this
# file before running it, it modifies system packages, group membership,
# and udev rules.
#
# Does NOT install ROS 2 itself -- follow ROS's own install docs first:
# https://docs.ros.org/en/jazzy/Installation.html
#
# Usage:
#   ./setup_system.sh            # base packages only (simulation-only use)
#   ./setup_system.sh --hardware # also configure real-hardware serial access

set -euo pipefail

echo "==> Installing build tools and ROS 2 packages this repo needs..."
sudo apt update
sudo apt install -y \
  build-essential cmake git \
  ros-jazzy-ros2-control \
  ros-jazzy-joint-state-broadcaster \
  ros-jazzy-joint-trajectory-controller

if [[ "${1:-}" == "--hardware" ]]; then
  echo "==> Configuring real-hardware serial access..."

  if ! groups "$USER" | grep -q '\bdialout\b'; then
    sudo usermod -aG dialout "$USER"
    echo "    Added $USER to the dialout group. Log out and back in for this"
    echo "    to take effect (a new shell alone is not enough)."
  else
    echo "    $USER is already in the dialout group."
  fi

  UDEV_RULE=/etc/udev/rules.d/99-ftdi-latency.rules
  if [[ ! -f "$UDEV_RULE" ]]; then
    echo 'ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"' \
      | sudo tee "$UDEV_RULE" > /dev/null
    sudo udevadm control --reload-rules
    echo "    Installed $UDEV_RULE (fixes the FTDI adapter's default 16ms"
    echo "    latency timer, which is too slow for this project's control loop)."
  else
    echo "    $UDEV_RULE already exists, leaving it as-is."
  fi
fi

echo "==> System setup done. Next: ./setup.sh"
