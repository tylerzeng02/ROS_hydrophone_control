#!/usr/bin/env python3

"""Standalone visual mockup of the fus_targeting_gui layout.

WHY THIS EXISTS
---------------

This is a *design reference*, not the real application. It recreates the
layout/structure of fus_targeting_gui by hand, from scratch, using nothing
but PySide6. It has:

  - no dependency on ROS 2 / MoveIt,
  - no dependency on the real Cyton hardware,
  - no working button/signal logic at all.

Every widget you see is real and laid out the same way as the eventual
production tool, so someone can read this file top to bottom and see
exactly how the GUI is put together (which container holds what, how
controls are grouped, what gets disabled until a device is "connected",
etc.) and adapt the same pattern for their own instrument-control GUI.

REQUIREMENTS
------------

Just PySide6:

  pip install PySide6
  python3 mockup_targeting_gui.py

DESIGN PATTERN SUMMARIZED
-------------------------

The window is split into three columns, each a self-contained panel:

1. Targeting panel (left): everything about the mesh and target picking.
   Load mesh, pick and preview targets, predefined points, search-area
   grid generation, alignment parameters (standoff, tilt, azimuth, roll).

2. Motion panel (middle): everything about the robot arm and planning.
   Connect to MoveIt, plan to target, preview trajectory, execute move,
   result details (actual vs. target, plan quality).

3. Log panel (right): a single scrolling read-only text feed shared by
   both panels, so the operator always has one place to look for status,
   errors, and warnings regardless of which panel raised them.

Within each panel, controls that only make sense once a device is
connected start disabled. This stops an operator from, for example,
planning a trajectory when the robot isn't available. That's a small but
important UX pattern worth copying: gate device-dependent controls behind
the connection state, don't just let them throw errors when clicked.

Each target can have per-target alignment parameters (saved per pick), and
a CSV log tracks the full history: mesh point, intended pose, planned
trajectory quality, actual achieved position (post-execution).
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class MeshPreviewPlaceholder(QWidget):
    """Placeholder for the live 3D mesh + point-picking view.

    The real GUI embeds a pyvista QtInteractor here (mesh rendering +
    surface picking + live preview of target point + approach vector).
    Wiring up the real mesh picker (pyvista) is a separate concern from
    the *layout*, so this just paints a static axes-and-cube sketch to
    show where that view lives and roughly how much space it should get.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(300)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("white"))

        margin = 20
        w, h = self.width() - margin, self.height() - margin

        # Axes + labels
        painter.setPen(QPen(QColor("black"), 1))
        painter.drawLine(margin, h, w, h)
        painter.drawLine(margin, h, margin, 10)
        painter.drawText(margin - 15, h + 15, "X")
        painter.drawText(w + 5, h + 15, "Y")
        painter.drawText(margin - 30, 15, "Z")

        # A sketched cube (representing a mesh), purely decorative.
        painter.setPen(QPen(QColor("#888"), 1.5))
        cx, cy = (margin + w) // 2, (10 + h) // 2
        size = 60
        # Front face
        painter.drawRect(cx - size // 2, cy - size // 2, size, size)
        # Back face (offset to look 3D)
        painter.drawRect(
            cx - size // 2 + 20, cy - size // 2 - 20, size, size
        )
        # Connecting edges
        for (x, y) in [
            (cx - size // 2, cy - size // 2),
            (cx + size // 2, cy - size // 2),
            (cx + size // 2, cy + size // 2),
            (cx - size // 2, cy + size // 2),
        ]:
            painter.drawLine(x, y, x + 20, y - 20)

        painter.drawText(
            margin + 5,
            h - 5,
            "Click to pick a target point on the mesh...",
        )


class TrajectoryPreviewPlaceholder(QWidget):
    """Placeholder for the planned trajectory visualization.

    The real GUI shows trajectory quality metrics and a simplified view of
    the planned arm configuration at key points. Here we just show axes +
    a sketched path.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(150)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("white"))

        margin = 20
        w, h = self.width() - margin, self.height() - margin

        # Axes
        painter.setPen(QPen(QColor("black"), 1))
        painter.drawLine(margin, h, w, h)
        painter.drawLine(margin, h, margin, 10)

        # A sketched sample trajectory (path from start to target).
        painter.setPen(QPen(QColor("#2a7fff"), 2))
        import math

        points = [
            (margin + i, h - int(40 * math.sin(i / 40)))
            for i in range(0, w - margin, 4)
        ]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            painter.drawLine(x1, y1, x2, y2)

        # Start/end markers
        painter.setPen(QPen(QColor("green"), 4))
        if points:
            painter.drawPoint(int(points[0][0]), int(points[0][1]))
        painter.setPen(QPen(QColor("red"), 4))
        if points:
            painter.drawPoint(int(points[-1][0]), int(points[-1][1]))

        painter.setFont(painter.font())
        painter.setPen(QPen(QColor("black"), 1))
        painter.drawText(margin + 5, 25, "Planned trajectory (preview)")


# ========================================================================== #
# Left panel: Mesh targeting controls
# ========================================================================== #


def build_targeting_panel() -> QGroupBox:
    """Targeting/mesh controls: load mesh, pick targets, manage alignment."""

    panel = QGroupBox("Mesh Targeting")
    layout = QVBoxLayout(panel)

    # --- Mesh loading ----
    mesh_group = QGroupBox("Mesh")
    mesh_layout = QVBoxLayout(mesh_group)

    mesh_row = QHBoxLayout()
    mesh_row.addWidget(QLabel("Mesh file:"))
    mesh_row.addStretch()
    mesh_row.addWidget(QPushButton("Load..."))
    mesh_layout.addLayout(mesh_row)

    mesh_status = QLabel("(no mesh loaded)")
    mesh_status.setStyleSheet("color: grey;")
    mesh_layout.addWidget(mesh_status)

    layout.addWidget(mesh_group)

    # --- 3D mesh view + picking ----
    layout.addWidget(QLabel("Surface picking:"))
    layout.addWidget(MeshPreviewPlaceholder())

    # --- Alignment parameters (standoff, tilt, azimuth, roll) ----
    # These are per-target: each pick captures whatever these controls
    # currently read, and re-selecting an earlier target restores its
    # own saved values here. Editing them while a target is selected
    # live-recomputes that target's pose.

    alignment_group = QGroupBox("Alignment Parameters")
    alignment_form = QFormLayout(alignment_group)

    standoff_spin = QDoubleSpinBox()
    standoff_spin.setRange(0.0, 300.0)
    standoff_spin.setDecimals(1)
    standoff_spin.setValue(10.0)
    standoff_spin.setSuffix(" mm")
    alignment_form.addRow("Standoff distance:", standoff_spin)

    tilt_spin = QDoubleSpinBox()
    tilt_spin.setRange(0.0, 89.0)
    tilt_spin.setDecimals(1)
    tilt_spin.setSuffix(" deg")
    alignment_form.addRow("Tilt from normal:", tilt_spin)

    azimuth_spin = QDoubleSpinBox()
    azimuth_spin.setRange(0.0, 359.9)
    azimuth_spin.setWrapping(True)
    azimuth_spin.setDecimals(1)
    azimuth_spin.setSuffix(" deg")
    alignment_form.addRow("Tilt azimuth:", azimuth_spin)

    roll_spin = QDoubleSpinBox()
    roll_spin.setRange(-180.0, 180.0)
    roll_spin.setDecimals(1)
    roll_spin.setSuffix(" deg")
    alignment_form.addRow("Probe roll:", roll_spin)

    reset_button = QPushButton("Reset to Surface Normal")
    alignment_form.addRow(reset_button)

    layout.addWidget(alignment_group)

    # --- Target list (all picked/predefined points) ----
    # Tabs: "Picked Points" (user-picked on the mesh) and "Predefined Points"
    # (loaded from a CSV file ahead of time, e.g. from CAD)

    targets_tabs = QTabWidget()

    picked_tab = QWidget()
    picked_layout = QVBoxLayout(picked_tab)
    picked_list = QListWidget()
    for i in range(3):  # dummy entries
        item = QListWidgetItem(f"Target {i}: ({-0.17 + i*0.02:.3f}, ...)")
        picked_list.addItem(item)
    picked_layout.addWidget(picked_list)
    targets_tabs.addTab(picked_tab, "Picked Points")

    predefined_tab = QWidget()
    predefined_layout = QVBoxLayout(predefined_tab)
    predefined_list = QListWidget()
    predefined_list.setPlaceholderText("(no predefined points loaded)")
    predefined_layout.addWidget(predefined_list)
    targets_tabs.addTab(predefined_tab, "Predefined Points")

    layout.addWidget(targets_tabs)

    # --- Search area workflow (auto-generate a grid of targets) ----
    search_group = QGroupBox("Search Area")
    search_layout = QVBoxLayout(search_group)

    search_buttons = QHBoxLayout()
    search_buttons.addWidget(QPushButton("Start Boundary"))
    search_buttons.addWidget(QPushButton("Finish Area"))
    search_buttons.addStretch()
    search_layout.addLayout(search_buttons)

    spacing_row = QHBoxLayout()
    spacing_row.addWidget(QLabel("Grid spacing:"))
    spacing_spin = QDoubleSpinBox()
    spacing_spin.setRange(1.0, 100.0)
    spacing_spin.setDecimals(1)
    spacing_spin.setValue(5.0)
    spacing_spin.setSuffix(" mm")
    spacing_row.addWidget(spacing_spin)
    spacing_row.addStretch()
    search_layout.addLayout(spacing_row)

    search_group.setEnabled(False)  # only enabled when mesh is loaded
    layout.addWidget(search_group)

    # All targeting controls disabled until a mesh is loaded
    # (mesh_group itself stays enabled so user can load one)
    for widget in (alignment_group, targets_tabs, search_group):
        widget.setEnabled(False)

    return panel


# ========================================================================== #
# Middle panel: Robot arm + motion planning
# ========================================================================== #


def build_motion_panel() -> QGroupBox:
    """Motion/planning controls: connect to MoveIt, plan/preview/execute."""

    panel = QGroupBox("Robot Arm (Cyton)")
    layout = QVBoxLayout(panel)

    # --- Connection ----
    # This is the gate: everything below starts disabled until "Connect" is
    # clicked and succeeds. In the real app, clicking "Connect" spawns the
    # MoveIt bridge node and waits for it to report ready.

    connect_button = QPushButton("Connect to MoveIt")
    layout.addWidget(connect_button)

    # --- Status indicator ----
    status_row = QHBoxLayout()
    status_row.addWidget(QLabel("Status:"))
    status_label = QLabel("disconnected")
    status_label.setStyleSheet("color: red; font-weight: bold;")
    status_row.addWidget(status_label)
    status_row.addStretch()
    layout.addLayout(status_row)

    # Everything below starts disabled until connected
    enabled_when_connected = []

    # --- Target selection (which point from the targeting panel) ----
    target_group = QGroupBox("Target")
    target_layout = QVBoxLayout(target_group)

    target_row = QHBoxLayout()
    target_row.addWidget(QLabel("Selected target:"))
    target_label = QLabel("(none)")
    target_label.setStyleSheet("color: grey;")
    target_row.addWidget(target_label)
    target_row.addStretch()
    target_layout.addLayout(target_row)

    target_coords = QLabel("Mesh point: -, Approach: -")
    target_coords.setStyleSheet("color: grey; font-size: 10px;")
    target_layout.addWidget(target_coords)

    target_group.setEnabled(False)
    enabled_when_connected.append(target_group)
    layout.addWidget(target_group)

    # --- Planning + preview ----
    plan_group = QGroupBox("Planning")
    plan_layout = QVBoxLayout(plan_group)

    plan_row = QHBoxLayout()
    plan_button = QPushButton("Plan to Target")
    plan_button.setEnabled(False)
    plan_row.addWidget(plan_button)

    plan_status = QLabel("")
    plan_status.setStyleSheet("color: grey;")
    plan_row.addWidget(plan_status)
    plan_row.addStretch()
    plan_layout.addLayout(plan_row)

    # Trajectory preview
    plan_layout.addWidget(QLabel("Trajectory preview:"))
    plan_layout.addWidget(TrajectoryPreviewPlaceholder())

    # Plan quality metrics
    metrics_form = QFormLayout()
    for label in ("Duration", "Smoothness", "Collisions"):
        metrics_form.addRow(label + ":", QLabel("-"))
    plan_layout.addLayout(metrics_form)

    plan_group.setEnabled(False)
    enabled_when_connected.append(plan_group)
    layout.addWidget(plan_group)

    # --- Execution ----
    exec_group = QGroupBox("Execution")
    exec_layout = QVBoxLayout(exec_group)

    exec_row = QHBoxLayout()
    exec_button = QPushButton("Execute Move")
    exec_button.setEnabled(False)
    exec_row.addWidget(exec_button)

    exec_status = QLabel("")
    exec_status.setStyleSheet("color: grey;")
    exec_row.addWidget(exec_status)
    exec_row.addStretch()
    exec_layout.addLayout(exec_row)

    # Results: actual vs. intended position
    results_form = QFormLayout()
    for label in ("Target position (mm)", "Actual position (mm)", "Error"):
        results_form.addRow(label + ":", QLabel("-"))
    exec_layout.addLayout(results_form)

    exec_group.setEnabled(False)
    enabled_when_connected.append(exec_group)
    layout.addWidget(exec_group)

    layout.addStretch()

    # Store references so a real handler can enable/disable them on connect
    panel._enabled_when_connected = enabled_when_connected

    return panel


# ========================================================================== #
# Right panel: Shared status/error log
# ========================================================================== #


def build_log_panel() -> QGroupBox:
    """Shared status/error/warning log visible to both left and right panels."""

    panel = QGroupBox("Log")
    layout = QVBoxLayout(panel)

    log = QPlainTextEdit()
    log.setReadOnly(True)
    log.setPlaceholderText(
        "Status, warnings, and error messages from both panels appear here.\n"
        "Scroll up to see earlier events."
    )

    # Dummy log entries for mockup purposes
    log.setPlainText(
        "[12:34:56] Mesh loaded: /home/temp/CT5_DICOM_skull_mesh_decimated.stl\n"
        "[12:34:57] Target 1 picked: (-0.170, 0.020, 0.150) mm\n"
        "[12:35:01] MoveIt bridge status: waiting for connection...\n"
    )

    layout.addWidget(log)

    return panel


# ========================================================================== #
# Main window
# ========================================================================== #


class TargetingMockupWindow(QMainWindow):
    """Top-level window: three panels side by side.

    Left (targeting): mesh and targeting controls, fixed width.
    Middle (motion): robot and planning controls, fixed width.
    Right (log): status and error log, takes remaining space.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "FUS Targeting -- design mockup (non-functional)"
        )
        self.resize(1800, 1000)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Left panel: targeting
        targeting_panel = build_targeting_panel()
        targeting_panel.setMaximumWidth(500)
        layout.addWidget(targeting_panel)

        # Middle panel: motion
        motion_panel = build_motion_panel()
        motion_panel.setMaximumWidth(500)
        layout.addWidget(motion_panel)

        # Right panel: log (takes remaining space)
        layout.addWidget(build_log_panel())


def main() -> int:
    app = QApplication(sys.argv)
    window = TargetingMockupWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
