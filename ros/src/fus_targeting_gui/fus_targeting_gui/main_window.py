"""Wires MeshView (picking) + MoveItBridge (planning/execution) together,
with an alignment-parameter panel (standoff distance, tilt off the surface
normal, azimuth around it, and probe roll), a "Predefined Points" list for
selecting points marked out ahead of time (e.g. from CAD) instead of
free-hand picking, a "Search Area" workflow for auto-generating a grid of
surface-snapped targets across a boundary you pick, a plan-preview-then-
confirm flow, and a per-session CSV log."""

import csv
import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from .config import AppConfig
from .mesh_view import MeshView
from .moveit_bridge import MoveItBridge
from .predefined_points import load_predefined_points_csv
from .search_area import generate_scan_grid

_TARGET_DATA_ROLE = 1000


class _CallThread(QThread):
    """Runs one blocking callable off the GUI thread, emits `result` when
    done. Used for both plan_to_pose() and execute() -- both block on
    ROS service/action calls and would otherwise freeze the GUI."""

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

        self._planned_trajectory = None
        self._worker_thread = None
        self._csv_path = None
        self._csv_writer = None
        self._csv_file = None
        self._syncing_params = False  # guards against param-sync feedback loops

        self._area_mode = False
        self._area_boundary = []  # list of (point, normal) while defining an area
        self._area_counter = 0
        self._area_worker_thread = None
        self._pending_area_spacing_mm = None

        self.mesh_view = MeshView()
        self.mesh_view.point_picked.connect(self._on_mesh_point_picked)

        self.target_list = QListWidget()
        self.target_list.setMaximumWidth(320)
        self.target_list.currentItemChanged.connect(self._on_target_selected)

        # Alignment parameters -- these are per-target: each pick captures
        # whatever these controls currently read, and re-selecting an
        # earlier target restores its own saved values here (see
        # _on_target_selected). Editing them while a target is selected
        # live-recomputes that target's pose and preview.
        self.standoff_spin = QDoubleSpinBox()
        self.standoff_spin.setRange(0.0, 300.0)
        self.standoff_spin.setDecimals(1)
        self.standoff_spin.setSuffix(" mm")
        self.standoff_spin.setValue(config.targeting.standoff_mm)
        self.standoff_spin.setToolTip(
            "Distance the probe stops short of the picked point, measured "
            "along the (possibly tilted) approach axis."
        )

        self.tilt_spin = QDoubleSpinBox()
        self.tilt_spin.setRange(0.0, 89.0)
        self.tilt_spin.setDecimals(1)
        self.tilt_spin.setSuffix(" deg")
        self.tilt_spin.setToolTip(
            "Angle of the approach axis away from the surface normal "
            "(0 = straight-on)."
        )

        self.azimuth_spin = QDoubleSpinBox()
        self.azimuth_spin.setRange(0.0, 359.9)
        self.azimuth_spin.setWrapping(True)
        self.azimuth_spin.setDecimals(1)
        self.azimuth_spin.setSuffix(" deg")
        self.azimuth_spin.setToolTip(
            "Direction of the tilt around the surface normal (0-360). "
            "Has no effect while tilt is 0."
        )

        self.roll_spin = QDoubleSpinBox()
        self.roll_spin.setRange(-180.0, 180.0)
        self.roll_spin.setDecimals(1)
        self.roll_spin.setSuffix(" deg")
        self.roll_spin.setToolTip("Rotation of the probe about its own approach axis.")

        for spin in (self.standoff_spin, self.tilt_spin, self.azimuth_spin, self.roll_spin):
            spin.valueChanged.connect(self._on_params_changed)

        self.reset_alignment_button = QPushButton("Reset to Surface Normal")
        self.reset_alignment_button.clicked.connect(self._on_reset_alignment_clicked)

        params_form = QFormLayout()
        params_form.addRow("Standoff distance:", self.standoff_spin)
        params_form.addRow("Tilt from normal:", self.tilt_spin)
        params_form.addRow("Tilt azimuth:", self.azimuth_spin)
        params_form.addRow("Probe roll:", self.roll_spin)
        params_group = QGroupBox("Alignment Parameters")
        params_layout = QVBoxLayout()
        params_layout.addLayout(params_form)
        params_layout.addWidget(self.reset_alignment_button)
        params_group.setLayout(params_layout)

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

        # Predefined points -- an alternative to free-hand picking, for
        # points marked out ahead of time (e.g. exported from CAD). Only
        # populated if config.mesh.predefined_points_path is set.
        self.predefined_list = QListWidget()
        self.predefined_list.setMaximumWidth(320)
        self.predefined_list.itemClicked.connect(self._on_predefined_point_selected)
        predefined_group = QGroupBox("Predefined Points")
        predefined_layout = QVBoxLayout()
        predefined_layout.addWidget(self.predefined_list)
        predefined_group.setLayout(predefined_layout)

        # Search area -- pick >=3 boundary points on the mesh, then
        # generate a grid of surface-snapped targets across that region at
        # a chosen spacing. See _on_finish_area_clicked() / search_area.py.
        self.area_spacing_spin = QDoubleSpinBox()
        self.area_spacing_spin.setRange(0.5, 100.0)
        self.area_spacing_spin.setDecimals(1)
        self.area_spacing_spin.setSuffix(" mm")
        self.area_spacing_spin.setValue(config.targeting.default_search_area_spacing_mm)

        self.start_area_button = QPushButton("Start Defining Search Area")
        self.start_area_button.clicked.connect(self._on_start_area_clicked)
        self.finish_area_button = QPushButton("Finish Area (Generate Grid)")
        self.finish_area_button.setEnabled(False)
        self.finish_area_button.clicked.connect(self._on_finish_area_clicked)
        self.cancel_area_button = QPushButton("Cancel")
        self.cancel_area_button.setEnabled(False)
        self.cancel_area_button.clicked.connect(self._on_cancel_area_clicked)

        self.area_status_label = QLabel(
            "Click “Start”, then click ≥ 3 boundary points on the mesh."
        )
        self.area_status_label.setWordWrap(True)

        area_form = QFormLayout()
        area_form.addRow("Grid spacing:", self.area_spacing_spin)
        area_buttons_layout = QHBoxLayout()
        area_buttons_layout.addWidget(self.start_area_button)
        area_buttons_layout.addWidget(self.finish_area_button)
        area_buttons_layout.addWidget(self.cancel_area_button)
        area_group = QGroupBox("Search Area")
        area_layout = QVBoxLayout()
        area_layout.addLayout(area_form)
        area_layout.addLayout(area_buttons_layout)
        area_layout.addWidget(self.area_status_label)
        area_group.setLayout(area_layout)

        side_panel = QVBoxLayout()
        side_panel.addWidget(self.load_mesh_button)
        side_panel.addWidget(QLabel("Picked targets (mesh-local mm):"))
        side_panel.addWidget(self.target_list)
        side_panel.addWidget(params_group)
        side_panel.addWidget(self.plan_button)
        side_panel.addWidget(self.execute_button)
        side_panel.addWidget(self.status_label)
        side_panel.addWidget(predefined_group)
        side_panel.addWidget(area_group)
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
        self._load_predefined_points()

    # ---- shared target-adding pipeline (manual pick / predefined / search area) ----

    def _current_params(self):
        return dict(
            standoff_mm=self.standoff_spin.value(),
            tilt_deg=self.tilt_spin.value(),
            azimuth_deg=self.azimuth_spin.value(),
            roll_deg=self.roll_spin.value(),
        )

    def _set_params_silently(self, params):
        self._syncing_params = True
        self.standoff_spin.setValue(params["standoff_mm"])
        self.tilt_spin.setValue(params["tilt_deg"])
        self.azimuth_spin.setValue(params["azimuth_deg"])
        self.roll_spin.setValue(params["roll_deg"])
        self._syncing_params = False

    def _compute_pose(self, point, normal, params):
        return self._config.registration.mesh_point_to_target_pose(
            point, normal,
            standoff_mm=params["standoff_mm"], tilt_deg=params["tilt_deg"],
            azimuth_deg=params["azimuth_deg"], roll_deg=params["roll_deg"],
        )

    def _add_target(self, point, normal, source="manual"):
        """Adds one target to the Picked Targets list using whatever the
        Alignment Parameters panel currently reads -- the single entry
        point every target-creation path (free-hand click, predefined-
        point selection, search-area grid generation) funnels through, so
        they all get identical pose computation, CSV logging, and
        plan/execute handling."""
        params = self._current_params()
        pose = self._compute_pose(point, normal, params)

        index = self.target_list.count()
        # point/normal are in the mesh's own local frame, already scaled to
        # meters by MeshView.load_mesh -- *1000 here for a human-readable
        # mm label; the underlying data stays in meters (see registration.py,
        # which needs it in the same units as base_frame).
        item = QListWidgetItem(
            f"#{index} [{source}]: "
            f"({point[0]*1000:.1f}, {point[1]*1000:.1f}, {point[2]*1000:.1f}) mm"
        )
        item.setData(
            _TARGET_DATA_ROLE,
            {"point": point, "normal": normal, "pose": pose, "source": source, **params},
        )
        self.target_list.addItem(item)
        self.target_list.setCurrentItem(item)  # triggers _on_target_selected
        return item

    def _on_mesh_point_picked(self, point, normal):
        if self._area_mode:
            self._on_area_boundary_point_picked(point, normal)
        else:
            self._add_target(point, normal, source="manual")

    # ---- predefined points ----

    def _load_predefined_points(self):
        path = self._config.mesh.predefined_points_path
        if not path:
            return
        try:
            points = load_predefined_points_csv(path, self._config.mesh.scale)
        except (OSError, KeyError, ValueError) as e:
            self.status_label.setText(f"Failed to load predefined points ({path}): {e}")
            return

        self.predefined_list.clear()
        marker_positions = []
        for i, p in enumerate(points):
            normal = p.normal_local
            if normal is None:
                normal = self.mesh_view.nearest_surface_normal(p.point_local)
            label = p.label if p.label else f"CAD-{i}"
            item = QListWidgetItem(
                f"{label}: ({p.point_local[0]*1000:.1f}, {p.point_local[1]*1000:.1f}, "
                f"{p.point_local[2]*1000:.1f}) mm"
            )
            item.setData(_TARGET_DATA_ROLE, {"point": p.point_local, "normal": normal, "label": label})
            self.predefined_list.addItem(item)
            marker_positions.append(p.point_local)

        self.mesh_view.show_predefined_points(marker_positions)

    def _on_predefined_point_selected(self, item):
        data = item.data(_TARGET_DATA_ROLE)
        self._add_target(data["point"], data["normal"], source=f"CAD:{data['label']}")

    # ---- search area ----

    def _on_start_area_clicked(self):
        self._area_mode = True
        self._area_boundary = []
        self.mesh_view.clear_area_boundary_preview()
        self.area_status_label.setText(
            "Search area: 0 boundary point(s) picked (need ≥ 3). Click points on the mesh."
        )
        self.start_area_button.setEnabled(False)
        self.finish_area_button.setEnabled(False)
        self.cancel_area_button.setEnabled(True)

    def _on_cancel_area_clicked(self):
        self._area_mode = False
        self._area_boundary = []
        self.mesh_view.clear_area_boundary_preview()
        self.area_status_label.setText(
            "Click “Start”, then click ≥ 3 boundary points on the mesh."
        )
        self.start_area_button.setEnabled(True)
        self.finish_area_button.setEnabled(False)
        self.cancel_area_button.setEnabled(False)

    def _on_area_boundary_point_picked(self, point, normal):
        self._area_boundary.append((point, normal))
        n = len(self._area_boundary)
        self.area_status_label.setText(f"Search area: {n} boundary point(s) picked (need ≥ 3).")
        self.finish_area_button.setEnabled(n >= 3)
        self.mesh_view.update_area_boundary_preview([p for p, _ in self._area_boundary])

    def _on_finish_area_clicked(self):
        if len(self._area_boundary) < 3:
            return
        boundary_points = [p for p, _ in self._area_boundary]
        boundary_normals = [n for _, n in self._area_boundary]
        spacing_mm = self.area_spacing_spin.value()

        try:
            grid_points, plane_normal = generate_scan_grid(
                boundary_points, boundary_normals, spacing_mm
            )
        except ValueError as e:
            QMessageBox.critical(self, "Search area failed", str(e))
            return

        # Raycasting every grid point against the mesh (mesh.ray_trace())
        # can take long enough on a large/fine area to freeze the GUI if
        # done synchronously here -- run it on a worker thread instead,
        # same pattern already used for plan()/execute(). A separate
        # _area_worker_thread attribute (not self._worker_thread) so this
        # can't collide with a Plan/Execute click landing mid-computation.
        self._pending_area_spacing_mm = spacing_mm
        self.start_area_button.setEnabled(False)
        self.finish_area_button.setEnabled(False)
        self.cancel_area_button.setEnabled(False)
        self.area_status_label.setText(
            f"Generating grid ({len(grid_points)} candidate point(s)) and raycasting onto "
            "the surface -- may take a few seconds for a large/fine area; GUI stays responsive."
        )
        self._area_worker_thread = _CallThread(
            self._raycast_area_grid, grid_points, plane_normal
        )
        self._area_worker_thread.result.connect(self._on_area_grid_raycasted)
        self._area_worker_thread.start()

    def _raycast_area_grid(self, grid_points, plane_normal):
        """Runs on the worker thread started by _on_finish_area_clicked().
        Pure geometry queries against self._mesh (ray_trace/find_closest_point)
        -- no Qt widget or VTK-actor/interactor calls here, which must only
        ever happen back on the GUI thread (see _on_area_grid_raycasted)."""
        results = []
        skipped = 0
        for grid_point in grid_points:
            result = self.mesh_view.raycast_onto_surface(grid_point, plane_normal)
            if result is None:
                skipped += 1
            else:
                results.append(result)
        return results, skipped

    def _on_area_grid_raycasted(self, outcome):
        results, skipped = outcome
        self._area_counter += 1
        area_label = f"area{self._area_counter}"
        for surface_point, surface_normal in results:
            self._add_target(surface_point, surface_normal, source=area_label)
        added = len(results)

        self.mesh_view.clear_area_boundary_preview()
        self._area_mode = False
        self._area_boundary = []
        self.start_area_button.setEnabled(True)
        self.finish_area_button.setEnabled(False)
        self.cancel_area_button.setEnabled(False)
        self.area_status_label.setText(
            f"{area_label}: added {added} target(s), {skipped} grid point(s) missed the surface."
        )
        QMessageBox.information(
            self, "Search area complete",
            f"Added {added} target(s) at ~{self._pending_area_spacing_mm:.1f}mm spacing "
            f"({skipped} grid point(s) fell outside the mesh and were skipped).\n\n"
            "Each was added to Picked Targets with the current Alignment Parameters "
            "applied uniformly -- review and Plan/Execute them one at a time as usual."
        )

    # ---- alignment-parameter editing / selection sync ----

    def _on_target_selected(self, current, _previous=None):
        if current is None:
            return
        data = current.data(_TARGET_DATA_ROLE)
        self._planned_trajectory = None
        self.plan_button.setEnabled(True)
        self.execute_button.setEnabled(False)

        self._set_params_silently(data)
        self.mesh_view.update_target_preview(
            data["point"], data["normal"], data["tilt_deg"], data["azimuth_deg"],
            data["roll_deg"], data["standoff_mm"],
        )

    def _on_params_changed(self, _value=None):
        if self._syncing_params:
            return
        item = self.target_list.currentItem()
        if item is None:
            return
        data = item.data(_TARGET_DATA_ROLE)
        params = self._current_params()
        data.update(params)
        data["pose"] = self._compute_pose(data["point"], data["normal"], params)
        item.setData(_TARGET_DATA_ROLE, data)

        self.mesh_view.update_target_preview(
            data["point"], data["normal"], params["tilt_deg"], params["azimuth_deg"],
            params["roll_deg"], params["standoff_mm"],
        )
        # Parameters changed -> any existing plan for this target is stale.
        self._planned_trajectory = None
        self.execute_button.setEnabled(False)

    def _on_reset_alignment_clicked(self):
        self._set_params_silently(
            {"standoff_mm": self.standoff_spin.value(), "tilt_deg": 0.0,
             "azimuth_deg": 0.0, "roll_deg": 0.0}
        )
        self._on_params_changed()

    # ---- mesh / plan / execute / CSV (unchanged from before) ----

    def _start_csv_log(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = Path(f"fus_targeting_session_{timestamp}.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "target_index", "source", "mesh_point_x_mm", "mesh_point_y_mm", "mesh_point_z_mm",
            "mesh_normal_x", "mesh_normal_y", "mesh_normal_z",
            "standoff_mm", "tilt_deg", "azimuth_deg", "roll_deg",
            "target_pose_x_m", "target_pose_y_m", "target_pose_z_m",
            "target_pose_qx", "target_pose_qy", "target_pose_qz", "target_pose_qw",
            "plan_result", "execute_result",
        ])
        self.status_label.setText(f"Logging to {self._csv_path}")

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
        pose = item.data(_TARGET_DATA_ROLE)["pose"]

        self.plan_button.setEnabled(False)
        self._worker_thread = _CallThread(self._bridge.plan_to_pose, pose)
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
            data = item.data(_TARGET_DATA_ROLE)
            point, normal, pose = data["point"], data["normal"], data["pose"]
            self._csv_writer.writerow([
                self.target_list.currentRow(), data["source"],
                point[0] * 1000, point[1] * 1000, point[2] * 1000,
                normal[0], normal[1], normal[2],
                data["standoff_mm"], data["tilt_deg"], data["azimuth_deg"], data["roll_deg"],
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
