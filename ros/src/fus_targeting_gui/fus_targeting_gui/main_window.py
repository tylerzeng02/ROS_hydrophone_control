"""Wires MeshView (picking) + MoveItBridge (planning/execution) together.

Layout follows the lab's existing fus_positioner tool (Velmex gantry +
PicoScope), adapted for the Cyton arm + MoveIt + a clickable 3D skull mesh
instead of a gantry + flat XYZ jog:

- Center: the 3D mesh view -- kept large (the primary interactive surface,
  unlike the mockup's small fixed trace-plot widget).
- "Oscilloscope (PicoScope)" panel: same role/position as the real tool --
  UI layout only for now (no PicoScope driver wired up yet, deliberately;
  see TracePlot's docstring). Its "Registration" sub-panel IS real: fits a
  mesh-local -> base_frame rigid transform from point pairs (Kabsch/SVD,
  see point_registration.py) instead of trusting FixedPoseRegistration's
  hand-typed config.yaml guess.
- "Targeting" panel: everything about DEFINING a target -- mesh loading,
  predefined points, search area, alignment parameters, the target list.
- "Motion" panel: everything about MOVING to the selected target -- plan/
  execute and a live preview/status readout.
- "Log" panel: one shared, timestamped, scrolling read-only feed for
  status/warning/error messages from every panel.

Note on the mockup's "Connect" device-gating pattern: not reproduced for
MoveIt, because it doesn't map onto this codebase -- MoveItBridge/pymoveit2
connect synchronously in __init__ (see node_entry.py), before MainWindow
even exists; there's no real "not yet connected" state for a button to
gate. It IS reproduced (disabled controls) for the Oscilloscope panel,
since that one's a genuine "not wired up yet" state."""

import csv
import datetime
import math
import threading
from pathlib import Path

from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSlider, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .config import AppConfig
from .mesh_view import MeshView
from .moveit_bridge import MoveItBridge
from .point_registration import fit_rigid_transform
from .predefined_points import load_predefined_points_csv
from .registration import rotation_matrix_to_rpy
from .search_area import generate_scan_grid, generate_volume_probing_points

_TARGET_DATA_ROLE = 1000
_LEVEL_COLORS = {"info": "#000000", "warning": "#b36b00", "error": "#c0392b"}

# Named starting points for 3D Volume Probing (see _on_finish_area_clicked()
# and search_area.generate_volume_probing_points()), each just pre-filling
# the height/depth-spacing controls, not a fixed shape file -- the user can
# still hand-adjust either value afterward before clicking Finish Area.
# height_pct=100 reaches exactly the skull's own centroid; lower values
# stop short of it, scaled to each surface point's own distance to center.
_VOLUME_PRESETS = {
    "Full Depth (Brim to Centre)": {"height_pct": 100.0, "depth_spacing_mm": 10.0},
    "Mid-Depth Block": {"height_pct": 50.0, "depth_spacing_mm": 8.0},
    "Shallow Surface Layer": {"height_pct": 20.0, "depth_spacing_mm": 5.0},
}


def _labeled_readout(label: str, unit: str = "") -> tuple[QWidget, QLabel]:
    """"Name ......... value unit" readout row (e.g. 'Peak Voltage: - Vp'),
    matching the Oscilloscope panel's mockup pattern. Returns the row
    widget plus the value QLabel for a future real wiring to update."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QLabel(label))
    layout.addStretch()
    value = QLabel(f"- {unit}".strip())
    value.setStyleSheet("color: grey;")
    layout.addWidget(value)
    return row, value


def _coordinate_table() -> QTableWidget:
    """3-column (x, y, z) table in mm, used for the Registration panel's
    paired point lists."""
    table = QTableWidget(0, 3)
    table.setHorizontalHeaderLabels(["x (mm)", "y (mm)", "z (mm)"])
    table.horizontalHeader().setStretchLastSection(True)
    return table


class TracePlot(QWidget):
    """Placeholder for the live oscilloscope-trace plot -- PicoScope
    integration isn't wired up yet (deliberately, per current scope; see
    this file's own header comment). This just sketches a static axes +
    curve so the layout shows where a real plotting widget (pyqtgraph,
    matplotlib, QtCharts...) will eventually go, same as the lab's
    existing fus_positioner mockup does for the same widget."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(110)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("white"))

        margin = 40
        w, h = self.width() - margin, self.height() - margin

        painter.setPen(QPen(QColor("black"), 1))
        painter.drawLine(margin, 10, margin, h)
        painter.drawLine(margin, h, w, h)
        painter.drawText(4, h // 2, "Volts")
        painter.drawText(w // 2, self.height() - 4, "Time (ms)")

        painter.setPen(QPen(QColor("#2a7fff"), 2))
        points = [
            (margin + i, h - 40 - int(30 * math.sin(i / 12)))
            for i in range(0, max(w - margin, 0), 4)
        ]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            painter.drawLine(x1, y1, x2, y2)


class Logger(QPlainTextEdit):
    """Shared, timestamped, read-only status/warning/error feed -- see this
    file's own header comment for why it exists. QPlainTextEdit has no
    setTextColor() (that's QTextEdit only); color-coding here goes through
    appendHtml() with an inline <span> instead."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)  # cap growth over a long session
        self.setPlaceholderText(
            "Status, warnings, and error messages from every panel appear here."
        )

    def log(self, message: str, level: str = "info"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        color = _LEVEL_COLORS.get(level, _LEVEL_COLORS["info"])
        self.appendHtml(f'<span style="color:{color}">[{timestamp}] {message}</span>')


class _CallThread(QThread):
    """Runs one blocking callable off the GUI thread, emits `result` when
    done. Used for plan()/execute(), area-grid raycasting, and the
    registration panel's live end-effector pose lookup -- all block long
    enough to freeze the GUI if run synchronously."""

    result = Signal(object)

    def __init__(self, fn, *args, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._args = args

    def run(self):
        self.result.emit(self._fn(*self._args))


class MainWindow(QMainWindow):
    # Emitted from the search-area raycast worker thread (see
    # _raycast_area_grid()) so the status label can show progress instead
    # of one static "may take a few seconds" message for the whole
    # operation. Measured at ~35ms/raycast on the full-res skull mesh, so
    # a large or fine area can take a minute or more, and with no progress
    # indication that is indistinguishable from a hang. Signal emission
    # from a worker thread is safe: Qt auto-queues delivery to the
    # receiving GUI-thread slot.
    area_progress = Signal(int, int)  # (current, total)

    # Emitted from the "Simulate All Targets" worker thread (see
    # _simulate_all_targets()) so its progress reaches the log safely.
    # That method must never touch Logger/QListWidget directly from a
    # worker thread.
    simulate_progress = Signal(int, int)  # (current, total)

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
        self._pending_volume_probing = None  # set only when 3D Volume Probing is checked

        self._last_picked_point = None  # most recent raw mesh-local pick, for Registration
        self._registration_pairs = []  # list of (mesh_point_local, robot_point_base_frame)
        self._registration_worker_thread = None
        self._collision_worker_thread = None
        self._simulate_worker_thread = None
        self._simulate_poses = None  # set while a simulate run is active; used by "Try Again"
        # Pause/cancel signaling for the simulate worker thread. Plain
        # Events, not a Qt mechanism, since the worker thread must be able
        # to block on these directly (see _simulate_checkpoint()) without
        # round-tripping through the Qt event loop. should_run starts set
        # (SET = keep going, CLEAR = hold at the next checkpoint); toggled
        # by the Pause/Resume button. cancel_requested starts clear; set by
        # the Cancel button and never cleared mid-run, only reset at the
        # start of the next simulate run.
        self._simulate_should_run = threading.Event()
        self._simulate_should_run.set()
        self._simulate_cancel_requested = threading.Event()
        self._fitted_rotation = None
        self._fitted_translation = None

        self._build_ui()

        self.area_progress.connect(self._on_area_progress)
        self.simulate_progress.connect(self._on_simulate_progress)
        self._bridge.status.connect(self._on_bridge_status)
        self._start_csv_log()
        self.mesh_view.load_mesh(config.mesh.default_path, scale=config.mesh.scale)
        self._load_predefined_points()
        self.logger.log("MoveIt bridge initialized.", "info")
        self._publish_skull_collision_object()

    # ---- layout ----

    def _build_ui(self):
        self.mesh_view = MeshView()
        self.mesh_view.point_picked.connect(self._on_mesh_point_picked)

        # Created before _build_motion_panel() since that panel embeds it
        # directly as a section, rather than Log getting its own tab.
        self.logger = Logger()

        instrument_panel = self._build_instrument_panel()
        targeting_panel = self._build_targeting_panel()
        motion_panel = self._build_motion_panel()

        # Tabbed side panel, matching RViz's own MotionPlanning display:
        # one narrow dock switching between tool groups instead of several
        # fixed-width panels competing with the 3D view for space. Each
        # panel keeps its own QGroupBox title from before (a little
        # redundant next to the tab label, but the safer, less invasive
        # change than stripping every panel-builder's outer framing).
        # Targeting is first/default since it is the main workflow, the
        # same way RViz defaults to its first tab.
        self.side_tabs = QTabWidget()
        # No setMaximumWidth: fixing a max would fight the splitter below,
        # which is the whole point of making this draggable. A floor is
        # still worth keeping so it can't be dragged down to nothing.
        self.side_tabs.setMinimumWidth(250)
        self.side_tabs.addTab(targeting_panel, "Targeting")
        self.side_tabs.addTab(instrument_panel, "Oscilloscope")
        self.side_tabs.addTab(motion_panel, "Motion")

        mesh_group = QGroupBox("3D View (click to pick a target)")
        mesh_layout = QVBoxLayout()
        mesh_layout.addWidget(self.mesh_view)
        mesh_group.setLayout(mesh_layout)

        # QSplitter (not a plain QHBoxLayout) so the boundary between the
        # 3D view and the side tabs is a draggable handle, not a fixed
        # split. setStretchFactor keeps the 3D view absorbing extra space
        # when the window itself is resized, while setSizes only sets the
        # initial split; the user can drag it to anything afterward.
        # 902/494 and the 1400x906 window size below match the layout the
        # user settled on as the preferred default (measured directly via
        # xwininfo on 2026-08-26).
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(mesh_group)
        splitter.addWidget(self.side_tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([902, 494])

        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(splitter)
        self.setCentralWidget(central)
        self.resize(1400, 906)

    def _build_instrument_panel(self) -> QWidget:
        """Oscilloscope (PicoScope) panel: UI layout only (no driver wired
        up yet; see TracePlot's docstring), except Registration, which is
        real. Matches the lab's existing fus_positioner tool's panel so
        the two feel like siblings. Plain QWidget, not a titled QGroupBox:
        this becomes a side_tabs tab's content directly (see _build_ui()),
        and the tab's own label already names it, so an inner title here
        would just duplicate that text and misalign against the other tabs'
        own differently-sized inner content."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        connect_button = QPushButton("Connect")
        connect_button.setEnabled(False)
        connect_button.setToolTip("PicoScope integration not wired up yet.")
        layout.addWidget(connect_button)

        acquisition_group = QGroupBox("Oscilloscope Control")
        acquisition_layout = QGridLayout(acquisition_group)
        acquisition_layout.addWidget(QLabel("Capture Length"), 0, 0)
        capture_length = QSpinBox()
        capture_length.setRange(1, 1_000_000)
        capture_length.setValue(10_000)
        capture_length.setSuffix(" samples")
        acquisition_layout.addWidget(capture_length, 0, 1)
        acquisition_layout.addWidget(QLabel("Averages"), 0, 2)
        averages = QSpinBox()
        averages.setRange(1, 1000)
        averages.setValue(4)
        averages.setSuffix(" waveforms")
        acquisition_layout.addWidget(averages, 0, 3)
        acquisition_layout.addWidget(QLabel("Input Range (Vp)"), 1, 0)
        input_range = QSpinBox()
        input_range.setRange(1, 1000)
        input_range.setValue(20)
        acquisition_layout.addWidget(input_range, 1, 1)
        acquisition_layout.addWidget(QLabel("Sampling Freq. (MHz)"), 1, 2)
        acquisition_layout.addWidget(QLineEdit(), 1, 3)
        acquisition_group.setEnabled(False)
        layout.addWidget(acquisition_group)

        results_group = QGroupBox("Scan Results")
        results_layout = QVBoxLayout(results_group)
        for label_text, unit in (("Peak Voltage", "Vp"), ("Focal Size", "mm")):
            row, _value_label = _labeled_readout(label_text, unit)
            results_layout.addWidget(row)
        results_group.setEnabled(False)
        layout.addWidget(results_group)

        layout.addWidget(TracePlot())
        capture_button = QPushButton("Capture Trace")
        capture_button.setEnabled(False)
        layout.addWidget(capture_button)

        layout.addWidget(self._build_registration_group())
        layout.addStretch(1)
        return panel

    def _build_registration_group(self) -> QGroupBox:
        """REAL functionality (unlike the rest of the Oscilloscope panel):
        fits a mesh-local -> base_frame rigid transform from point pairs.
        Workflow: click a mesh point (also adds it as a normal target --
        harmless), physically position the arm's probe against the
        corresponding real landmark (e.g. via RViz), then "Add Point" here
        to pair the two. See point_registration.py for the actual math."""
        group = QGroupBox("Registration")
        layout = QVBoxLayout()

        buttons_row = QHBoxLayout()
        self.add_registration_point_button = QPushButton("Add")
        self.add_registration_point_button.setToolTip(
            "Pair the most recently clicked mesh point with the arm's live end-effector position."
        )
        self.add_registration_point_button.clicked.connect(
            self._on_add_registration_point_clicked
        )
        self.remove_registration_point_button = QPushButton("Remove")
        self.remove_registration_point_button.setToolTip("Remove the selected point pair.")
        self.remove_registration_point_button.clicked.connect(
            self._on_remove_registration_point_clicked
        )
        self.clear_registration_points_button = QPushButton("Clear")
        self.clear_registration_points_button.setToolTip("Clear all point pairs.")
        self.clear_registration_points_button.clicked.connect(
            self._on_clear_registration_points_clicked
        )
        for b in (
            self.add_registration_point_button, self.remove_registration_point_button,
            self.clear_registration_points_button,
        ):
            buttons_row.addWidget(b)
        layout.addLayout(buttons_row)

        tables_row = QHBoxLayout()
        mesh_col = QVBoxLayout()
        mesh_col.addWidget(QLabel("Mesh Coordinates (mm)"))
        self.mesh_points_table = _coordinate_table()
        self.mesh_points_table.setMaximumHeight(110)
        mesh_col.addWidget(self.mesh_points_table)
        robot_col = QVBoxLayout()
        robot_col.addWidget(QLabel("Robot Coordinates (mm)"))
        self.robot_points_table = _coordinate_table()
        self.robot_points_table.setMaximumHeight(110)
        robot_col.addWidget(self.robot_points_table)
        tables_row.addLayout(mesh_col)
        tables_row.addLayout(robot_col)
        layout.addLayout(tables_row)

        footer_row = QHBoxLayout()
        self.show_matrix_button = QPushButton("Show matrix")
        self.show_matrix_button.setEnabled(False)
        self.show_matrix_button.clicked.connect(self._on_show_matrix_clicked)
        self.save_registration_button = QPushButton("Save")
        self.save_registration_button.setEnabled(False)
        self.save_registration_button.clicked.connect(self._on_save_registration_clicked)
        footer_row.addWidget(self.show_matrix_button)
        footer_row.addStretch()
        footer_row.addWidget(self.save_registration_button)
        layout.addLayout(footer_row)

        group.setLayout(layout)
        return group

    def _build_clipping_group(self) -> QGroupBox:
        """Sliders that progressively cut away part of the mesh (top/
        bottom/right/left, in the mesh's own local frame -- see
        MeshView.set_clip_fraction()'s own docstring on why "right"/"left"
        aren't independently verified against true anatomical sides), so
        obstructing geometry can be moved out of the way when picking a
        target underneath/behind it. The actual re-clip (measured ~175ms
        on the real mesh) fires on slider RELEASE, not every drag tick --
        the percentage label still updates live so dragging feels
        responsive even though the 3D geometry only updates once you let go."""
        group = QGroupBox("Clipping (cut away obstructing geometry)")
        layout = QVBoxLayout()

        self.clip_sliders = {}
        self.clip_value_labels = {}
        for axis, title in (
            ("top", "Cut from top"), ("bottom", "Cut from bottom"),
            ("right", "Cut from right"), ("left", "Cut from left"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(title))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 95)
            slider.setValue(0)
            value_label = QLabel("0%")
            value_label.setMinimumWidth(36)
            slider.valueChanged.connect(
                lambda v, lbl=value_label: lbl.setText(f"{v}%")
            )
            slider.sliderReleased.connect(
                lambda ax=axis, sl=slider: self._on_clip_slider_released(ax, sl)
            )
            row.addWidget(slider)
            row.addWidget(value_label)
            layout.addLayout(row)
            self.clip_sliders[axis] = slider
            self.clip_value_labels[axis] = value_label

        self.reset_clipping_button = QPushButton("Reset Clipping")
        self.reset_clipping_button.clicked.connect(self._on_reset_clipping_clicked)
        layout.addWidget(self.reset_clipping_button)

        group.setLayout(layout)
        return group

    def _on_clip_slider_released(self, axis, slider):
        fraction = slider.value() / 100.0
        self.mesh_view.set_clip_fraction(axis, fraction)
        self.logger.log(f"Clipping: {axis} set to {slider.value()}%.", "info")

    def _on_reset_clipping_clicked(self):
        for axis, slider in self.clip_sliders.items():
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
            self.clip_value_labels[axis].setText("0%")
        self.mesh_view.reset_clipping()
        self.logger.log("Clipping reset.", "info")

    def _build_targeting_panel(self) -> QWidget:
        """Everything about DEFINING a target: mesh loading, predefined
        points, search area, alignment parameters, the target list. Plain
        QWidget, not a titled QGroupBox: see _build_instrument_panel()'s
        docstring for why (this becomes a side_tabs tab's content
        directly, whose own label already names it)."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.load_mesh_button = QPushButton("Load Mesh...")
        self.load_mesh_button.clicked.connect(self._on_load_mesh_clicked)
        layout.addWidget(self.load_mesh_button)

        layout.addWidget(self._build_clipping_group())

        layout.addWidget(QLabel("Picked targets (mesh-local mm):"))
        self.target_list = QListWidget()
        self.target_list.setMaximumHeight(140)
        self.target_list.currentItemChanged.connect(self._on_target_selected)
        layout.addWidget(self.target_list)

        target_buttons_row = QHBoxLayout()
        self.remove_target_button = QPushButton("Remove")
        self.remove_target_button.setToolTip("Remove the selected target from the list.")
        self.remove_target_button.clicked.connect(self._on_remove_target_clicked)
        self.clear_targets_button = QPushButton("Clear All")
        self.clear_targets_button.clicked.connect(self._on_clear_targets_clicked)
        target_buttons_row.addWidget(self.remove_target_button)
        target_buttons_row.addWidget(self.clear_targets_button)
        layout.addLayout(target_buttons_row)

        self.standoff_spin = QDoubleSpinBox()
        self.standoff_spin.setRange(0.0, 300.0)
        self.standoff_spin.setDecimals(1)
        self.standoff_spin.setSuffix(" mm")
        self.standoff_spin.setValue(self._config.targeting.standoff_mm)
        self.standoff_spin.setToolTip(
            "Distance the probe stops short of the picked point, measured "
            "along the (possibly tilted) approach axis."
        )

        self.tilt_spin = QDoubleSpinBox()
        self.tilt_spin.setRange(0.0, 89.0)
        self.tilt_spin.setDecimals(1)
        self.tilt_spin.setSuffix(" deg")
        self.tilt_spin.setToolTip(
            "Angle of the approach axis away from the surface normal (0 = straight-on)."
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

        self.reset_alignment_button = QPushButton("Reset to Normal")
        self.reset_alignment_button.setToolTip(
            "Zero tilt, azimuth, and roll: straight-on approach along the surface normal."
        )
        self.reset_alignment_button.clicked.connect(self._on_reset_alignment_clicked)

        params_form = QFormLayout()
        params_form.addRow("Standoff distance:", self.standoff_spin)
        params_form.addRow("Tilt from normal:", self.tilt_spin)
        params_form.addRow("Tilt azimuth:", self.azimuth_spin)
        params_form.addRow("Probe roll:", self.roll_spin)
        self.enforce_orientation_checkbox = QCheckBox("Enforce orientation on approach")
        self.enforce_orientation_checkbox.setChecked(self._config.targeting.enforce_orientation)
        self.enforce_orientation_checkbox.setToolTip(
            "When checked, Plan requires the arm to actually arrive at the computed "
            "Tilt/Azimuth/Roll orientation above, not just the position. A reachability "
            "scan found only 3/15 points plannable with this on vs. 10/15 with it off "
            "(elbow_yaw's narrow locked window makes an exact-angle approach much "
            "harder to satisfy), so expect more failed plans with this checked. "
            "Takes effect on the next Plan click, no restart needed."
        )
        self.enforce_orientation_checkbox.toggled.connect(self._on_enforce_orientation_toggled)

        params_group = QGroupBox("Alignment Parameters")
        params_layout = QVBoxLayout()
        params_layout.addLayout(params_form)
        params_layout.addWidget(self.reset_alignment_button)
        params_layout.addWidget(self.enforce_orientation_checkbox)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        self.predefined_list = QListWidget()
        self.predefined_list.setMaximumHeight(100)
        self.predefined_list.itemClicked.connect(self._on_predefined_point_selected)
        predefined_group = QGroupBox("Predefined Points")
        predefined_layout = QVBoxLayout()
        predefined_layout.addWidget(self.predefined_list)
        predefined_group.setLayout(predefined_layout)
        layout.addWidget(predefined_group)

        self.area_spacing_spin = QDoubleSpinBox()
        self.area_spacing_spin.setRange(0.5, 100.0)
        self.area_spacing_spin.setDecimals(1)
        self.area_spacing_spin.setSuffix(" mm")
        self.area_spacing_spin.setValue(self._config.targeting.default_search_area_spacing_mm)

        # 3D Volume Probing: extends the same boundary-picked footprint
        # into a column of points per surface location, reaching inward
        # toward the skull's own centroid instead of staying on the
        # surface. See search_area.generate_volume_probing_points() for
        # why the columns are scaled per-point rather than a fixed depth.
        self.volume_probing_checkbox = QCheckBox("3D Volume Probing")
        self.volume_probing_checkbox.setToolTip(
            "Extend the search area inward from the surface toward the skull's "
            "centre, instead of a flat surface scan. Height and Depth Spacing "
            "below become active when this is checked."
        )
        self.volume_probing_checkbox.toggled.connect(self._on_volume_probing_toggled)

        self.volume_preset_combo = QComboBox()
        self.volume_preset_combo.addItem("Custom")
        for name in _VOLUME_PRESETS:
            self.volume_preset_combo.addItem(name)
        self.volume_preset_combo.setEnabled(False)
        self.volume_preset_combo.currentTextChanged.connect(self._on_volume_preset_selected)

        self.volume_height_spin = QDoubleSpinBox()
        self.volume_height_spin.setRange(0.0, 100.0)
        self.volume_height_spin.setDecimals(0)
        self.volume_height_spin.setSuffix(" %")
        self.volume_height_spin.setValue(100.0)
        self.volume_height_spin.setEnabled(False)
        self.volume_height_spin.setToolTip(
            "How far each column reaches toward the skull's centre, as a percentage "
            "of that column's own surface-to-centre distance. 100% reaches exactly "
            "the centre; lower values stop short of it. Editing this switches the "
            "Preset above to Custom."
        )
        self.volume_height_spin.valueChanged.connect(self._on_volume_param_edited)

        self.volume_depth_spacing_spin = QDoubleSpinBox()
        self.volume_depth_spacing_spin.setRange(0.5, 100.0)
        self.volume_depth_spacing_spin.setDecimals(1)
        self.volume_depth_spacing_spin.setSuffix(" mm")
        self.volume_depth_spacing_spin.setValue(10.0)
        self.volume_depth_spacing_spin.setEnabled(False)
        self.volume_depth_spacing_spin.setToolTip(
            "Spacing between points along each inward column. Editing this "
            "switches the Preset above to Custom."
        )
        self.volume_depth_spacing_spin.valueChanged.connect(self._on_volume_param_edited)

        self.start_area_button = QPushButton("Start Area")
        self.start_area_button.setToolTip("Start defining a search area by clicking boundary points on the mesh.")
        self.start_area_button.clicked.connect(self._on_start_area_clicked)
        self.finish_area_button = QPushButton("Finish Area")
        self.finish_area_button.setEnabled(False)
        self.finish_area_button.setToolTip("Finish the boundary and generate a target grid inside it.")
        self.finish_area_button.clicked.connect(self._on_finish_area_clicked)
        self.cancel_area_button = QPushButton("Cancel")
        self.cancel_area_button.setEnabled(False)
        self.cancel_area_button.clicked.connect(self._on_cancel_area_clicked)
        self.undo_area_point_button = QPushButton("Undo Point")
        self.undo_area_point_button.setEnabled(False)
        self.undo_area_point_button.setToolTip(
            "Removes only the most recently placed boundary point -- unlike "
            "Cancel, keeps the rest of the boundary you've already picked."
        )
        self.undo_area_point_button.clicked.connect(self._on_undo_area_point_clicked)

        self.area_status_label = QLabel(
            'Click "Start", then click ≥ 3 boundary points on the mesh.'
        )
        self.area_status_label.setWordWrap(True)

        area_form = QFormLayout()
        area_form.addRow("Grid spacing:", self.area_spacing_spin)
        area_form.addRow(self.volume_probing_checkbox)
        area_form.addRow("Preset:", self.volume_preset_combo)
        area_form.addRow("Height:", self.volume_height_spin)
        area_form.addRow("Depth spacing:", self.volume_depth_spacing_spin)
        area_buttons_layout = QHBoxLayout()
        area_buttons_layout.addWidget(self.start_area_button)
        area_buttons_layout.addWidget(self.finish_area_button)
        area_buttons_layout.addWidget(self.cancel_area_button)
        area_buttons_row2 = QHBoxLayout()
        area_buttons_row2.addWidget(self.undo_area_point_button)
        area_group = QGroupBox("Search Area")
        area_layout = QVBoxLayout()
        area_layout.addLayout(area_form)
        area_layout.addLayout(area_buttons_layout)
        area_layout.addLayout(area_buttons_row2)
        area_layout.addWidget(self.area_status_label)
        area_group.setLayout(area_layout)
        layout.addWidget(area_group)

        layout.addStretch(1)
        return panel

    def _build_motion_panel(self) -> QWidget:
        """Everything about MOVING to the currently selected target. Plain
        QWidget, not a titled QGroupBox: see _build_instrument_panel()'s
        docstring for why (this becomes a side_tabs tab's content
        directly, whose own label already names it)."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.status_label = QLabel("Load a mesh to begin.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.preview_label = QLabel("Select a target to preview.")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        self.plan_button = QPushButton("Plan to Point")
        self.plan_button.setToolTip("Plan a trajectory to the selected target.")
        self.plan_button.setEnabled(False)
        self.plan_button.clicked.connect(self._on_plan_clicked)
        layout.addWidget(self.plan_button)

        self.execute_button = QPushButton("Execute Plan")
        self.execute_button.setToolTip("Execute the currently planned move.")
        self.execute_button.setEnabled(False)
        self.execute_button.clicked.connect(self._on_execute_clicked)
        layout.addWidget(self.execute_button)

        self.simulate_all_button = QPushButton("Simulate All")
        self.simulate_all_button.setToolTip(
            "Plans and executes every Picked Target in sequence (mock hardware): "
            "watch the arm visit each one live in RViz's MotionPlanning display."
        )
        self.simulate_all_button.clicked.connect(self._on_simulate_all_clicked)
        layout.addWidget(self.simulate_all_button)

        # Only meaningful while a simulate run is active; disabled otherwise
        # (see _start_simulate_worker()/_on_simulate_all_finished()).
        simulate_controls = QHBoxLayout()
        self.simulate_pause_button = QPushButton("Pause")
        self.simulate_pause_button.setEnabled(False)
        self.simulate_pause_button.setToolTip(
            "Holds the run after the current plan/execute step finishes, before "
            "starting the next target. Click again (now labeled Resume) to continue."
        )
        self.simulate_pause_button.clicked.connect(self._on_simulate_pause_clicked)
        simulate_controls.addWidget(self.simulate_pause_button)

        self.simulate_cancel_button = QPushButton("Cancel")
        self.simulate_cancel_button.setEnabled(False)
        self.simulate_cancel_button.setToolTip(
            "Stops the run. Also requests an immediate stop of whatever the arm "
            "is currently executing, not just a hold before the next target."
        )
        self.simulate_cancel_button.clicked.connect(self._on_simulate_cancel_clicked)
        simulate_controls.addWidget(self.simulate_cancel_button)
        layout.addLayout(simulate_controls)

        # A section within this tab, not its own tab (self.logger is built
        # in _build_ui() before this method runs, specifically for this).
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.logger)
        layout.addWidget(log_group, stretch=1)
        return panel

    # ---- shared status/log plumbing ----

    def _on_bridge_status(self, text: str):
        self.status_label.setText(text)
        self.logger.log(text, "info")
        # Also mirror to stdout. The GUI window can't always be brought to
        # front for a screenshot (no window-management tool available in
        # every environment this runs in), so this is what makes bridge
        # status visible via the process's own redirected log file too.
        print(f"[status] {text}", flush=True)

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

    def _add_target(self, point, normal, source="manual", refresh_markers=True,
                     standoff_mm_override=None):
        """Adds one target to the Picked Targets list using whatever the
        Alignment Parameters panel currently reads -- the single entry
        point every target-creation path (free-hand click, predefined-
        point selection, search-area grid generation) funnels through, so
        they all get identical pose computation, CSV logging, and
        plan/execute handling.

        refresh_markers=False lets a bulk caller (e.g. search-area/volume-
        probing, which can add many targets in one loop) skip
        _refresh_target_markers() per item and call it once after the
        loop instead. That call rebuilds the marker list from every
        target currently in the list, so calling it once per item in a
        large bulk add is O(N^2) in the target count -- harmless at the
        scale a single manual click or a modest 2D grid produces, but a
        real, user-visible freeze once 3D volume probing multiplies a
        surface grid into many points per column.

        standoff_mm_override replaces the Alignment Parameters panel's
        global standoff for this one target. mesh_point_to_target_pose()
        always backs the target away from `point` along the outward
        normal by the standoff distance, which is correct for a surface
        point (stay outside, don't touch the bone) but wrong for a 3D
        volume-probing point already sitting at a deliberately-computed
        interior depth: applying the usual outward standoff on top would
        push the actual commanded target back toward, or past, the
        surface, undoing the depth the Height/Depth-spacing controls just
        set up. Volume-probing callers pass 0.0 here for exactly that
        reason; every other caller leaves this None and gets the normal,
        globally-configured standoff.
        """
        params = self._current_params()
        if standoff_mm_override is not None:
            params = dict(params, standoff_mm=standoff_mm_override)
        pose = self._compute_pose(point, normal, params)

        index = self.target_list.count()
        # point/normal are in the mesh's own local frame, already scaled to
        # meters by MeshView.load_mesh. The *1000 here is only for a
        # human-readable mm label; the underlying data stays in meters
        # (see registration.py, which needs it in the same units as
        # base_frame).
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
        if refresh_markers:
            self._refresh_target_markers()
        return item

    def _refresh_target_markers(self):
        """Redraws persistent numbered markers on the mesh for every target
        currently in the list -- see MeshView.update_target_markers()."""
        points, labels = [], []
        for i in range(self.target_list.count()):
            data = self.target_list.item(i).data(_TARGET_DATA_ROLE)
            points.append(data["point"])
            labels.append(f"#{i}")
        self.mesh_view.update_target_markers(points, labels)

    def _on_remove_target_clicked(self):
        row = self.target_list.currentRow()
        if row < 0:
            return
        self.target_list.takeItem(row)
        self._refresh_target_markers()

    def _on_clear_targets_clicked(self):
        if self.target_list.count() == 0:
            return
        confirm = QMessageBox.question(
            self, "Clear all targets", "Remove all picked targets?"
        )
        if confirm != QMessageBox.Yes:
            return
        self.target_list.clear()
        self._refresh_target_markers()
        self.mesh_view.clear_target_preview()

    def _on_mesh_point_picked(self, point, normal):
        self._last_picked_point = point
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
            self.logger.log(f"Failed to load predefined points ({path}): {e}", "error")
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
        self.logger.log(f"Loaded {len(points)} predefined point(s) from {path}.", "info")

    def _on_predefined_point_selected(self, item):
        data = item.data(_TARGET_DATA_ROLE)
        self.logger.log(f"Selected predefined point: {data['label']}", "info")
        self._last_picked_point = data["point"]
        self._add_target(data["point"], data["normal"], source=f"CAD:{data['label']}")

    # ---- search area ----

    def _on_volume_probing_toggled(self, checked: bool):
        self.volume_preset_combo.setEnabled(checked)
        self.volume_height_spin.setEnabled(checked)
        self.volume_depth_spacing_spin.setEnabled(checked)

    def _on_volume_preset_selected(self, name: str):
        preset = _VOLUME_PRESETS.get(name)
        if preset is None:  # "Custom", or a spurious signal during addItem()
            return
        self._syncing_params = True
        self.volume_height_spin.setValue(preset["height_pct"])
        self.volume_depth_spacing_spin.setValue(preset["depth_spacing_mm"])
        self._syncing_params = False

    def _on_volume_param_edited(self, _value=None):
        # Hand-editing height or depth spacing means whatever preset was
        # selected no longer describes the current values; guarded by
        # _syncing_params so applying a preset's own values above doesn't
        # immediately flip the combo back to "Custom".
        if self._syncing_params:
            return
        self.volume_preset_combo.setCurrentText("Custom")

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
        self.undo_area_point_button.setEnabled(False)

    def _on_cancel_area_clicked(self):
        self._area_mode = False
        self._area_boundary = []
        self.mesh_view.clear_area_boundary_preview()
        self.area_status_label.setText(
            'Click "Start", then click ≥ 3 boundary points on the mesh.'
        )
        self.start_area_button.setEnabled(True)
        self.finish_area_button.setEnabled(False)
        self.cancel_area_button.setEnabled(False)
        self.undo_area_point_button.setEnabled(False)

    def _on_undo_area_point_clicked(self):
        """Removes only the most recently placed boundary point, unlike
        Cancel which discards the whole in-progress boundary -- lets you
        fix one mis-click without redoing everything else."""
        if not self._area_boundary:
            return
        self._area_boundary.pop()
        n = len(self._area_boundary)
        self.area_status_label.setText(f"Search area: {n} boundary point(s) picked (need ≥ 3).")
        self.finish_area_button.setEnabled(n >= 3)
        self.undo_area_point_button.setEnabled(n > 0)
        self.mesh_view.update_area_boundary_preview([p for p, _ in self._area_boundary])

    def _on_area_boundary_point_picked(self, point, normal):
        self._area_boundary.append((point, normal))
        n = len(self._area_boundary)
        self.area_status_label.setText(f"Search area: {n} boundary point(s) picked (need ≥ 3).")
        self.finish_area_button.setEnabled(n >= 3)
        self.undo_area_point_button.setEnabled(True)
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

        # Captured here, on the GUI thread, rather than re-read once the
        # worker thread finishes: the checkbox/spin values could otherwise
        # change mid-raycast on a large/slow area, and _on_area_grid_raycasted()
        # (also GUI-thread, but a later event) should apply exactly the
        # settings the user had when they clicked Finish Area.
        self._pending_volume_probing = None
        if self.volume_probing_checkbox.isChecked():
            centroid = self.mesh_view.get_mesh_centroid()
            if centroid is None:
                QMessageBox.critical(self, "Search area failed", "No mesh loaded.")
                return
            self._pending_volume_probing = {
                "centroid": centroid,
                "height_fraction": self.volume_height_spin.value() / 100.0,
                "depth_spacing_mm": self.volume_depth_spacing_spin.value(),
            }

        # Raycasting every grid point against the mesh (mesh.ray_trace())
        # can take long enough on a large/fine area to freeze the GUI if
        # done synchronously here, so it runs on a worker thread instead,
        # the same pattern already used for plan()/execute(). A separate
        # _area_worker_thread attribute (not self._worker_thread) so this
        # can't collide with a Plan/Execute click landing mid-computation.
        self._pending_area_spacing_mm = spacing_mm
        self.start_area_button.setEnabled(False)
        self.finish_area_button.setEnabled(False)
        self.cancel_area_button.setEnabled(False)
        self.undo_area_point_button.setEnabled(False)
        # ~35ms/raycast measured directly against the full-res skull mesh,
        # giving an estimate instead of a vague "a few seconds" that can
        # be wrong by over a minute on a large or fine area.
        estimated_s = len(grid_points) * 0.035
        self.area_status_label.setText(
            f"Raycasting {len(grid_points)} candidate point(s) onto the surface "
            f"(~{estimated_s:.0f}s estimated)... GUI stays responsive."
        )
        self.logger.log(
            f"Search area: raycasting {len(grid_points)} candidate point(s) at "
            f"{spacing_mm:.1f}mm spacing (~{estimated_s:.0f}s estimated)...", "info"
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
        ever happen back on the GUI thread (see _on_area_grid_raycasted).
        Emits area_progress periodically -- see that signal's own comment
        for why (~35ms/raycast measured on the real mesh, so this step
        alone can take well over a minute on a large/fine area)."""
        total = len(grid_points)
        results = []
        skipped = 0
        for i, grid_point in enumerate(grid_points):
            result = self.mesh_view.raycast_onto_surface(grid_point, plane_normal)
            if result is None:
                skipped += 1
            else:
                results.append(result)
            self.area_progress.emit(i + 1, total)
        return results, skipped

    def _on_area_progress(self, current, total):
        self.area_status_label.setText(
            f"Raycasting onto the surface: {current}/{total} point(s)..."
        )

    def _on_area_grid_raycasted(self, outcome):
        results, skipped = outcome
        self._area_counter += 1
        area_label = f"area{self._area_counter}"

        volume_probing = self._pending_volume_probing
        if volume_probing is not None:
            # Only the surface anchor points came from raycasting; every
            # point further down each column is a pure geometric
            # extension toward the centroid (see
            # generate_volume_probing_points()'s own docstring for why it
            # needs no raycast of its own).
            targets = generate_volume_probing_points(
                results,
                volume_probing["centroid"],
                volume_probing["height_fraction"],
                volume_probing["depth_spacing_mm"],
            )
            volume_note = (
                f", extended to a 3D volume ({volume_probing['height_fraction']*100:.0f}% "
                f"toward centre at {volume_probing['depth_spacing_mm']:.1f}mm depth spacing)"
            )
        else:
            targets = results
            volume_note = ""

        # Volume-probing points already sit at a deliberately-computed
        # interior depth; the usual outward standoff would push them back
        # toward the surface instead (see _add_target()'s docstring).
        # Plain 2D surface points keep the normal, panel-configured standoff.
        standoff_override = 0.0 if volume_probing is not None else None
        for point, normal in targets:
            self._add_target(
                point, normal, source=area_label, refresh_markers=False,
                standoff_mm_override=standoff_override,
            )
        if targets:
            self._refresh_target_markers()
        added = len(targets)

        self.mesh_view.clear_area_boundary_preview()
        self._area_mode = False
        self._area_boundary = []
        self.start_area_button.setEnabled(True)
        self.finish_area_button.setEnabled(False)
        self.cancel_area_button.setEnabled(False)
        self.area_status_label.setText(
            f"{area_label}: added {added} target(s){volume_note}, "
            f"{skipped} grid point(s) missed the surface."
        )
        self.logger.log(
            f"{area_label}: added {added} target(s){volume_note}, "
            f"{skipped} grid point(s) missed the surface.",
            "warning" if skipped else "info",
        )
        QMessageBox.information(
            self, "Search area complete",
            f"Added {added} target(s) at ~{self._pending_area_spacing_mm:.1f}mm surface "
            f"spacing{volume_note} "
            f"({skipped} grid point(s) fell outside the mesh and were skipped).\n\n"
            "Each was added to Picked Targets with the current Alignment Parameters "
            "applied uniformly: review and Plan/Execute them one at a time as usual."
        )

    # ---- registration (mesh <-> robot point-pair fit) ----

    def _on_add_registration_point_clicked(self):
        if self._last_picked_point is None:
            QMessageBox.warning(
                self, "No mesh point picked",
                "Click a point on the mesh first (or select a Predefined Point)."
            )
            return
        self.add_registration_point_button.setEnabled(False)
        self._registration_worker_thread = _CallThread(
            self._bridge.get_current_end_effector_pose
        )
        self._registration_worker_thread.result.connect(self._on_registration_robot_point_ready)
        self._registration_worker_thread.start()

    def _on_registration_robot_point_ready(self, robot_point):
        self.add_registration_point_button.setEnabled(True)
        if robot_point is None:
            QMessageBox.critical(
                self, "Failed to read arm position",
                "Could not look up the arm's current position -- see the Log panel."
            )
            return
        mesh_point = self._last_picked_point
        self._registration_pairs.append((mesh_point, robot_point))
        self._refresh_registration_tables()
        self.logger.log(
            f"Registration: added point pair #{len(self._registration_pairs) - 1} "
            f"(mesh {tuple(round(v*1000, 2) for v in mesh_point)}mm <-> "
            f"robot {tuple(round(v*1000, 2) for v in robot_point)}mm).", "info",
        )

    def _on_remove_registration_point_clicked(self):
        row = self.mesh_points_table.currentRow()
        if row < 0 or row >= len(self._registration_pairs):
            return
        del self._registration_pairs[row]
        self._refresh_registration_tables()

    def _on_clear_registration_points_clicked(self):
        self._registration_pairs = []
        self._refresh_registration_tables()

    def _refresh_registration_tables(self):
        n = len(self._registration_pairs)
        self.mesh_points_table.setRowCount(n)
        self.robot_points_table.setRowCount(n)
        for i, (mesh_pt, robot_pt) in enumerate(self._registration_pairs):
            for col, val in enumerate(mesh_pt):
                self.mesh_points_table.setItem(i, col, QTableWidgetItem(f"{val*1000:.2f}"))
            for col, val in enumerate(robot_pt):
                self.robot_points_table.setItem(i, col, QTableWidgetItem(f"{val*1000:.2f}"))
        enough = n >= 3
        self.show_matrix_button.setEnabled(enough)
        self.save_registration_button.setEnabled(enough and self._fitted_rotation is not None)

    def _on_show_matrix_clicked(self):
        mesh_points = [p[0] for p in self._registration_pairs]
        robot_points = [p[1] for p in self._registration_pairs]
        try:
            R, t, rmse = fit_rigid_transform(mesh_points, robot_points)
        except ValueError as e:
            QMessageBox.critical(self, "Registration fit failed", str(e))
            return

        self._fitted_rotation = R
        self._fitted_translation = t
        self.save_registration_button.setEnabled(True)

        roll, pitch, yaw = rotation_matrix_to_rpy(R)
        rmse_mm = rmse * 1000.0
        self.logger.log(
            f"Registration fit: RMSE={rmse_mm:.3f}mm over {len(self._registration_pairs)} "
            f"pair(s).", "warning" if rmse_mm > 2.0 else "info",
        )
        QMessageBox.information(
            self, "Fitted registration",
            f"Fitted transform ({len(self._registration_pairs)} point pairs):\n\n"
            f"Translation (xyz_m): [{t[0]:.5f}, {t[1]:.5f}, {t[2]:.5f}]\n"
            f"Rotation (rpy_rad): [{roll:.5f}, {pitch:.5f}, {yaw:.5f}]\n\n"
            f"RMSE: {rmse_mm:.3f} mm\n\n"
            "Not applied automatically -- click Save to write it to a file, then "
            "copy those values into config.yaml's registration: block yourself."
        )

    def _on_save_registration_clicked(self):
        if self._fitted_rotation is None:
            QMessageBox.warning(self, "Nothing to save", "Click “Show matrix” first.")
            return
        roll, pitch, yaw = rotation_matrix_to_rpy(self._fitted_rotation)
        t = self._fitted_translation
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"fitted_registration_{timestamp}.yaml")
        with open(out_path, "w") as f:
            f.write(
                "# Fitted mesh-local -> base_frame registration, from the GUI's "
                "Registration panel (point_registration.fit_rigid_transform).\n"
                "# NOT automatically applied -- copy these values into "
                "config/default_config.yaml's registration: block yourself.\n"
                "registration:\n"
                f"  xyz_m: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]\n"
                f"  rpy_rad: [{roll:.6f}, {pitch:.6f}, {yaw:.6f}]\n"
            )
        self.logger.log(f"Saved fitted registration to {out_path}", "info")
        QMessageBox.information(
            self, "Saved",
            f"Wrote fitted registration to {out_path}.\n\n"
            "Copy these values into config.yaml's registration: block to actually use them."
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
        idx = self.target_list.row(current)
        self.preview_label.setText(
            f"Target #{idx} [{data['source']}]: "
            f"({data['point'][0]*1000:.1f}, {data['point'][1]*1000:.1f}, "
            f"{data['point'][2]*1000:.1f}) mm. Ready to plan."
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

    def _on_enforce_orientation_toggled(self, checked: bool):
        # self._config.targeting is the exact object MoveItBridge.__init__()
        # was given (see node_entry.py), not a copy, and plan_to_pose()
        # reads targeting.enforce_orientation fresh on every call rather
        # than caching it, so mutating it here takes effect on the very
        # next Plan click with no restart needed.
        self._config.targeting.enforce_orientation = checked
        self.logger.log(
            f"Orientation enforcement {'enabled' if checked else 'disabled'} "
            f"(takes effect on the next Plan).",
            "info",
        )

    # ---- mesh / plan / execute / CSV ----

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
        if not path:
            return
        try:
            self.mesh_view.load_mesh(path, scale=self._config.mesh.scale)
        except Exception as e:  # noqa: BLE001 -- report to the GUI, don't crash it
            self.logger.log(f"Failed to load mesh: {e}", "error")
            QMessageBox.critical(self, "Load failed", f"Could not load mesh: {e}")
            return
        self.status_label.setText(f"Loaded {path}")
        self.logger.log(f"Loaded mesh: {path}", "info")
        self.target_list.clear()
        self._refresh_target_markers()
        # load_mesh() already reset clip fractions internally. Keep the
        # slider widgets themselves in sync so they don't show a stale
        # position for the new mesh.
        for axis, slider in self.clip_sliders.items():
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
            self.clip_value_labels[axis].setText("0%")
        self._publish_skull_collision_object()

    def _publish_skull_collision_object(self):
        """Registers the currently-loaded mesh as a real MoveIt collision
        object so the planner actually avoids it -- without this, move_group
        has no knowledge the skull exists at all (confirmed directly: this
        codebase never touched PlanningScene/CollisionObject before this
        method existed, and a target placed inside the mesh's own volume
        planned "successfully" with no awareness anything was there).

        Deliberately uses MeshView.get_collision_mesh_data(), which always
        reads the FULL, UNCLIPPED original mesh -- clipping sliders are a
        picking/viewing convenience, not a claim the physical skull has
        less material there, so collision awareness must not follow them.
        Runs on a worker thread (the publish is a blocking service call)."""
        vertices_local, triangles = self.mesh_view.get_collision_mesh_data()
        if vertices_local is None:
            return
        vertices_base = self._config.registration.transform_points_to_base_frame(vertices_local)
        self.logger.log(
            f"Publishing skull collision object ({len(triangles)} triangles)...", "info"
        )
        self._collision_worker_thread = _CallThread(
            self._bridge.set_skull_collision_object, vertices_base, triangles
        )
        self._collision_worker_thread.result.connect(self._on_skull_collision_published)
        self._collision_worker_thread.start()

    def _on_skull_collision_published(self, success):
        self.logger.log(
            "Skull collision object active -- planning will avoid it." if success
            else "Failed to publish skull collision object -- see status above.",
            "info" if success else "error",
        )

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
            self._prompt_retry_or_cancel_planning()
            return

        self.execute_button.setEnabled(True)
        QMessageBox.information(
            self, "Plan ready",
            "A plan was found. Review it (RViz's MotionPlanning display shows the "
            "preview automatically, same as this project's other MoveIt tools) before "
            "pressing Execute."
        )

    def _prompt_retry_or_cancel_planning(self):
        """Shown after plan_to_pose() exhausts its own automatic retries
        (including a collision retreat, if one was needed; see
        MoveItBridge.plan_to_pose()). "Try Again" re-issues the same
        planning request from scratch, since a fresh attempt can still
        succeed on a different random seed even after 5 prior failures."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Planning failed")
        box.setText(
            "Planning failed after automatic retries. See the status log for details."
        )
        retry_button = box.addButton("Try Again", QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() == retry_button:
            self._on_plan_clicked()

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

    # ---- simulate all targets (plan + execute the whole list in sequence) ----

    def _on_simulate_all_clicked(self):
        count = self.target_list.count()
        if count == 0:
            QMessageBox.warning(self, "No targets", "Pick at least one target first.")
            return
        confirm = QMessageBox.question(
            self, "Simulate all targets",
            f"Plan and execute all {count} target(s) in sequence?\n\n"
            "Watch the arm move through each one in RViz's MotionPlanning display. "
            "Stops if any target fails to plan or execute, with a Try Again option "
            "to retry just that target."
        )
        if confirm != QMessageBox.Yes:
            return

        # Capture plain pose data on the GUI thread BEFORE starting the
        # worker. _simulate_all_targets() must never touch QListWidget
        # (or Logger, or any other Qt widget) directly from a worker
        # thread. Kept on self so a later "Try Again" click can resume
        # from the same poses without re-reading the list.
        poses = [self.target_list.item(i).data(_TARGET_DATA_ROLE)["pose"] for i in range(count)]
        self._simulate_poses = poses
        self._start_simulate_worker(poses, start_index=0)

    def _start_simulate_worker(self, poses, start_index):
        self.simulate_all_button.setEnabled(False)
        self.plan_button.setEnabled(False)
        self.execute_button.setEnabled(False)
        self.simulate_pause_button.setEnabled(True)
        self.simulate_pause_button.setText("Pause")
        self.simulate_cancel_button.setEnabled(True)
        self._simulate_should_run.set()
        self._simulate_cancel_requested.clear()
        if start_index == 0:
            self.logger.log(f"Simulating {len(poses)} target(s) in sequence...", "info")
        else:
            self.logger.log(f"Resuming from target #{start_index}...", "info")
        self._simulate_worker_thread = _CallThread(
            self._simulate_all_targets, poses, start_index
        )
        self._simulate_worker_thread.result.connect(self._on_simulate_all_finished)
        self._simulate_worker_thread.start()

    def _simulate_checkpoint(self) -> bool:
        """Call from the simulate worker thread at each safe point between
        motions (never mid-plan or mid-execute, both of which are bounded
        calls that either finish or fail on their own). Blocks here while
        paused. Does not itself stop an already-executing trajectory --
        that is Cancel's separate, immediate self._bridge.cancel_execution()
        call, since interrupting motion mid-flight needs a real controller-
        level stop request, not just skipping the next loop iteration.

        Returns:
            False if cancellation was requested (caller should stop and
            return); True otherwise.
        """
        self._simulate_should_run.wait()
        return not self._simulate_cancel_requested.is_set()

    def _simulate_all_targets(self, poses, start_index=0):
        """Runs on a worker thread. Plans then immediately executes each
        pose from start_index onward (mock hardware), stopping at the
        first failure or cancellation. start_index lets a "Try Again"/
        "Resume" click continue from that point instead of restarting the
        whole sequence. Progress goes out via simulate_progress (a Signal,
        safe to emit from a worker thread); self._bridge's own
        plan_to_pose()/execute() already report their own status the same
        safe way.

        Returns:
            (targets_completed, final_status).
        """
        total = len(poses)
        for i in range(start_index, total):
            if not self._simulate_checkpoint():
                return i, "cancelled"
            self.simulate_progress.emit(i, total)
            trajectory = self._bridge.plan_to_pose(poses[i])
            if trajectory is None:
                return i, "plan_failed"
            if not self._simulate_checkpoint():
                return i, "cancelled"
            success = self._bridge.execute(trajectory)
            if not success:
                return i, "execute_failed"
        return total, "all_ok"

    def _on_simulate_progress(self, current, total):
        self.logger.log(f"Simulate: target #{current} of {total}...", "info")

    def _on_simulate_pause_clicked(self):
        if self._simulate_should_run.is_set():
            self._simulate_should_run.clear()
            self.simulate_pause_button.setText("Resume")
            self.logger.log("Pausing after the current step finishes...", "info")
        else:
            self._simulate_should_run.set()
            self.simulate_pause_button.setText("Pause")
            self.logger.log("Resuming.", "info")

    def _on_simulate_cancel_clicked(self):
        self._simulate_cancel_requested.set()
        self._simulate_should_run.set()  # wake the worker if it is currently paused
        self._bridge.cancel_execution()
        self.simulate_cancel_button.setEnabled(False)
        self.logger.log("Cancel requested -- stopping after the current step.", "info")

    def _on_simulate_all_finished(self, outcome):
        self.simulate_all_button.setEnabled(True)
        self.plan_button.setEnabled(self.target_list.currentItem() is not None)
        self.simulate_pause_button.setEnabled(False)
        self.simulate_pause_button.setText("Pause")
        self.simulate_cancel_button.setEnabled(False)
        completed, status = outcome
        if status == "all_ok":
            self.logger.log(f"Simulated all {completed} target(s) successfully.", "info")
            QMessageBox.information(
                self, "Simulation complete",
                f"Successfully planned and executed all {completed} target(s)."
            )
            self._simulate_poses = None
            return

        if status == "cancelled":
            self.logger.log(f"Simulation cancelled at target #{completed}.", "info")
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Information)
            box.setWindowTitle("Simulation cancelled")
            box.setText(
                f"Cancelled at target #{completed}. "
                f"{completed} target(s) completed successfully before this."
            )
            resume_button = box.addButton("Resume", QMessageBox.AcceptRole)
            box.addButton("Stop", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is resume_button:
                self._start_simulate_worker(self._simulate_poses, start_index=completed)
            else:
                self._simulate_poses = None
            return

        reason = "planning" if status == "plan_failed" else "execution"
        self.logger.log(
            f"Simulation stopped at target #{completed} ({reason} failed).", "error"
        )
        # A plain QMessageBox.critical() only offers Ok. Building the box
        # directly adds a Try Again button that resumes from this exact
        # target, and pops the same dialog again on a repeat failure, so
        # the user can retry as many times as they want.
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Simulation stopped")
        box.setText(
            f"Stopped at target #{completed}: {reason} failed. "
            f"{completed} target(s) completed successfully before this."
        )
        try_again_button = box.addButton("Try Again", QMessageBox.AcceptRole)
        box.addButton("Stop", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is try_again_button:
            self._start_simulate_worker(self._simulate_poses, start_index=completed)
        else:
            self._simulate_poses = None

    def closeEvent(self, event):
        if self._csv_file is not None:
            self._csv_file.close()
        self._bridge.shutdown()
        super().closeEvent(event)
