"""Wires MeshView (picking) + MoveItBridge (planning/execution) together,
with the plan-preview-then-confirm flow and a per-session CSV log."""

import csv
import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from .config import AppConfig
from .mesh_view import MeshView
from .moveit_bridge import MoveItBridge


class _CallThread(QThread):
    """Runs one blocking callable off the GUI thread and emits `result`
    when done. Used for both plan_to_pose() and execute(), since both
    block on ROS service or action calls and would otherwise freeze the
    GUI."""

    result = Signal(object)

    def __init__(self, fn, *args, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._args = args

    def run(self):
        self.result.emit(self._fn(*self._args))


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, moveit_bridge: MoveItBridge, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FUS Targeting")
        self._config = config
        self._bridge = moveit_bridge

        self._pending_pose = None      # most recently picked-and-computed target pose
        self._planned_trajectory = None
        self._worker_thread = None
        self._csv_path = None
        self._csv_writer = None
        self._csv_file = None

        self.mesh_view = MeshView()
        self.mesh_view.point_picked.connect(self._on_point_picked)

        self.target_list = QListWidget()
        self.target_list.setMaximumWidth(320)

        self.status_label = QLabel("Load a mesh to begin.")
        self.status_label.setWordWrap(True)
        self._bridge.status.connect(self.status_label.setText)

        self.load_mesh_button = QPushButton("Load Mesh...")
        self.load_mesh_button.clicked.connect(self._on_load_mesh_clicked)

        self.plan_button = QPushButton("Plan to Selected Point")
        self.plan_button.setEnabled(False)
        self.plan_button.clicked.connect(self._on_plan_clicked)

        self.execute_button = QPushButton("Execute Planned Move")
        self.execute_button.setEnabled(False)
        self.execute_button.clicked.connect(self._on_execute_clicked)

        side_panel = QVBoxLayout()
        side_panel.addWidget(self.load_mesh_button)
        side_panel.addWidget(QLabel("Picked targets (mesh-local mm):"))
        side_panel.addWidget(self.target_list)
        side_panel.addWidget(self.plan_button)
        side_panel.addWidget(self.execute_button)
        side_panel.addWidget(self.status_label)
        side_panel.addStretch(1)

        side_widget = QWidget()
        side_widget.setLayout(side_panel)
        side_widget.setMaximumWidth(340)

        layout = QHBoxLayout()
        layout.addWidget(self.mesh_view, stretch=1)
        layout.addWidget(side_widget)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self._start_csv_log()
        self.mesh_view.load_mesh(config.mesh.default_path, scale=config.mesh.scale)

    def _start_csv_log(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = Path(f"fus_targeting_session_{timestamp}.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "target_index", "mesh_point_x_mm", "mesh_point_y_mm", "mesh_point_z_mm",
            "mesh_normal_x", "mesh_normal_y", "mesh_normal_z",
            "target_pose_x_m", "target_pose_y_m", "target_pose_z_m",
            "target_pose_qx", "target_pose_qy", "target_pose_qz", "target_pose_qw",
            "plan_result", "execute_result",
        ])
        self.status_label.setText(f"Logging to {self._csv_path}")

    def _on_point_picked(self, point, normal):
        pose = self._config.registration.mesh_point_to_target_pose(
            point, normal, self._config.targeting.standoff_mm
        )
        index = self.target_list.count()
        item = QListWidgetItem(
            f"#{index}: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}) mm"
        )
        item.setData(1000, {"point": point, "normal": normal, "pose": pose})
        self.target_list.addItem(item)
        self.target_list.setCurrentItem(item)

        self._pending_pose = pose
        self._planned_trajectory = None
        self.plan_button.setEnabled(True)
        self.execute_button.setEnabled(False)

    def _on_load_mesh_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open skull mesh", "", "Mesh files (*.stl *.ply *.obj);;All files (*)"
        )
        if path:
            self.mesh_view.load_mesh(path, scale=self._config.mesh.scale)
            self.status_label.setText(f"Loaded {path}")

    def _on_plan_clicked(self):
        item = self.target_list.currentItem()
        if item is None:
            QMessageBox.warning(self, "No target selected", "Pick a point on the mesh first.")
            return
        data = item.data(1000)
        self._pending_pose = data["pose"]

        self.plan_button.setEnabled(False)
        self._worker_thread = _CallThread(self._bridge.plan_to_pose, self._pending_pose)
        self._worker_thread.result.connect(self._on_plan_result)
        self._worker_thread.start()

    def _on_plan_result(self, trajectory):
        self.plan_button.setEnabled(True)
        self._planned_trajectory = trajectory
        if trajectory is None:
            QMessageBox.critical(self, "Planning failed", "See status message for details.")
            return

        self.execute_button.setEnabled(True)
        QMessageBox.information(
            self, "Plan ready",
            "A plan was found. Review it (RViz's MotionPlanning display shows the "
            "preview automatically, same as this project's other MoveIt tools) before "
            "pressing Execute."
        )

    def _on_execute_clicked(self):
        if self._planned_trajectory is None:
            return
        confirm = QMessageBox.question(
            self, "Confirm execution",
            "Execute this planned move on the real arm now?",
        )
        if confirm != QMessageBox.Yes:
            return

        self.execute_button.setEnabled(False)
        self._worker_thread = _CallThread(self._bridge.execute, self._planned_trajectory)
        self._worker_thread.result.connect(self._on_execute_result)
        self._worker_thread.start()

    def _on_execute_result(self, success):
        self.execute_button.setEnabled(True)
        item = self.target_list.currentItem()
        if item is not None:
            data = item.data(1000)
            point, normal, pose = data["point"], data["normal"], data["pose"]
            self._csv_writer.writerow([
                self.target_list.currentRow(),
                point[0], point[1], point[2],
                normal[0], normal[1], normal[2],
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
                "ok" if self._planned_trajectory is not None else "failed",
                "ok" if success else "failed",
            ])
            self._csv_file.flush()

        if not success:
            QMessageBox.critical(self, "Execution failed", "See status message for details.")

    def closeEvent(self, event):
        if self._csv_file is not None:
            self._csv_file.close()
        self._bridge.shutdown()
        super().closeEvent(event)
