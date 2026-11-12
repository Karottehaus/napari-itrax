from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Literal, cast

import napari
import numpy as np
import pandas as pd
from napari.layers import Image as ImageLayer
from PIL import Image
from qtpy.QtCore import QSettings, Qt
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..analysis import DEFAULT_LOG_RATIOS, DEFAULT_RATIOS
from ..images import find_optical_image, photo_scale
from ..models import INVALID_POINTS, Settings
from ..workflow import (
    ExportOptions,
    PreparedRun,
    ProcessingOptions,
    ProcessingResult,
    create_template,
    export_selection,
    prepare_run,
    process_run,
    quality_table,
)
from .common import PathEdit, WorkerWidget, combo, optional_number


class ProcessingWidget(WorkerWidget):
    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.preferences = QSettings("napari-itrax", "processing")
        self.result: ProcessingResult | None = None
        self.photo_layer: ImageLayer | None = None
        self.tabs = QTabWidget()
        QVBoxLayout(self.body).addWidget(self.tabs)
        input_page, selection_page, export_page = QWidget(), QWidget(), QWidget()
        self.tabs.addTab(input_page, "Input / settings")
        self.tabs.addTab(selection_page, "Elements")
        self.tabs.addTab(export_page, "Export")
        form = QFormLayout(input_page)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.input_path = PathEdit(directory=True)
        self.depth_path = PathEdit(pattern="Excel (*.xlsx *.xlsm)")
        self.output_path = PathEdit(directory=True)
        form.addRow("Scan folder", self.input_path)
        form.addRow("True Depth", self.depth_path)
        form.addRow(self.button("Create True Depth template…", self.template))
        form.addRow("Output folder", self.output_path)
        self.tube = combo(["", "Cr", "Rh", "Mo", "Ag"])
        self.voltage, self.current, self.exposure = QLineEdit(), QLineEdit(), QLineEdit()
        form.addRow("Tube (document.txt takes precedence)", self.tube)
        form.addRow("Voltage (kV)", self.voltage)
        form.addRow("Current (mA)", self.current)
        form.addRow("Exposure (s)", self.exposure)
        self.operator = QLineEdit(str(self.preferences.value("operator", "")))
        form.addRow("Operator", self.operator)
        self.flags: dict[str, QCheckBox] = {}
        for key, text, checked in (
                ("replicates", "Include XRF replicates", False),
                ("high_resolution", "High-resolution photos", False),
                ("ruler_images", "Photos with depth rulers", True),
                ("argon_images", "Photos with Ar counts", True),
                ("error_log", "Write log file", False),
        ):
            control = QCheckBox(text)
            control.setChecked(checked)
            self.flags[key] = control
            form.addRow(control)
        self.invalid = combo(list(INVALID_POINTS))
        self.suspect = QLineEdit("1.5")
        self.file_format = combo(["xlsx", "csv"])
        self.core_workbooks = QCheckBox("One workbook per scan with treatment tabs (xlsx)")
        form.addRow("Invalid points", self.invalid)
        form.addRow("Suspect: Ar × median (empty to disable)", self.suspect)
        form.addRow("File format", self.file_format)
        form.addRow(self.core_workbooks)
        self.process_button = self.button("Process scans", self.process)
        form.addRow(self.process_button)
        self.scans = combo([])
        form.addRow("Photo preview", self.scans)
        form.addRow(self.button("Show selected photo", self.show_photo))

        layout = QVBoxLayout(selection_page)
        legend = QLabel(
            "Select elements for normalization and CLR. Zeros: zero counts; CV: relative variation; "
            "Noise and Repeat: estimated noise share; Overlap: neighboring peak and log correlation. "
            "Dimmed rows meet the original quality flags: ≥5% zeros, ≥50% noise/repeat, "
            "overlap r ≥0.8, air, or tube signal."
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)
        self.elements = QTableWidget(0, 6)
        self.elements.setHorizontalHeaderLabels(
            ["Element", "Zeros %", "CV %", "Noise %", "Repeat %", "Overlap"]
        )
        layout.addWidget(self.elements)
        buttons = QHBoxLayout()
        for text, action in (
                ("All", partial(self.select_all, True)),
                ("None", partial(self.select_all, False)),
                ("Load…", self.load_selection),
                ("Save…", self.save_selection),
                ("Last selection", self.last_selection),
        ):
            buttons.addWidget(self.button(text, action))
        layout.addLayout(buttons)
        export_form = QFormLayout(export_page)
        self.ratios = QLineEdit(str(self.preferences.value("ratios", DEFAULT_RATIOS)))
        self.log_ratios = QLineEdit(str(self.preferences.value("log_ratios", DEFAULT_LOG_RATIOS)))
        self.inc_coh = QCheckBox("Include inc/coh ratio")
        self.detail = QCheckBox("50 cm detail pages for long cores")
        self.detail.setChecked(self.preferences.value("detail", False, type=bool))
        export_form.addRow("Ratios (e.g. Ca/Ti)", self.ratios)
        export_form.addRow("Log ratios (e.g. Ca/Ti)", self.log_ratios)
        export_form.addRow(self.inc_coh)
        export_form.addRow(self.detail)
        self.export_button = self.button("Write element files and plots", self.export)
        export_form.addRow(self.export_button)
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)

    def options(self) -> ProcessingOptions:
        if not all((self.input_path.text(), self.depth_path.text(), self.output_path.text())):
            raise ValueError("Choose a scan folder, True Depth file, and output folder.")
        settings: Settings = dict(
            voltage=optional_number(self.voltage.text()),
            current=optional_number(self.current.text()),
            time=optional_number(self.exposure.text()),
        )
        if self.tube.currentText():
            settings["tube"] = self.tube.currentText()
        return ProcessingOptions(
            Path(self.input_path.text()),
            Path(self.depth_path.text()),
            Path(self.output_path.text()),
            settings=settings,
            replicates=self.flags["replicates"].isChecked(),
            high_resolution=self.flags["high_resolution"].isChecked(),
            ruler_images=self.flags["ruler_images"].isChecked(),
            argon_images=self.flags["argon_images"].isChecked(),
            error_log=self.flags["error_log"].isChecked(),
            invalid_points=cast(Literal["empty", "keep", "leave out"], self.invalid.currentText()),
            suspect_factor=optional_number(self.suspect.text()),
            file_format=cast(Literal["xlsx", "csv"], self.file_format.currentText()),
            core_workbooks=self.core_workbooks.isChecked(),
            operator=self.operator.text().strip(),
        )

    def process(self) -> None:
        self.start_work(partial(prepare_run, self.options()), self.prepared, "Validating input…")

    def prepared(self, prepared: PreparedRun) -> None:
        exists = Path(prepared.folders.root).exists()
        if (
                exists
                and QMessageBox.question(
            self,
            "Replace results?",
            f"Delete and replace the existing results folder?\n{prepared.folders.root}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
                != QMessageBox.StandardButton.Yes
        ):
            self.status.setText("Cancelled; existing results retained.")
            return
        self.result = None
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.start_work(
            partial(process_run, prepared, overwrite=exists, progress=self.progress_changed.emit),
            self.processed,
            "Processing scans…",
        )

    def processed(self, result: ProcessingResult) -> None:
        self.result = result
        self.preferences.setValue("operator", self.operator.text())
        table = quality_table(result)
        self.elements.setRowCount(len(table))
        for row, values in enumerate(table.itertuples(index=False, name=None)):
            overlap = values[5]
            poor = (
                    values[1] >= 5
                    or any(pd.notna(v) and v >= 50 for v in values[3:5])
                    or (overlap is not None and (overlap[1] is None or overlap[1] >= 0.8))
            )
            for column, value in enumerate(values):
                if column == 5:
                    text = (
                        "" if value is None else value[0] + ("" if value[1] is None else f" {value[1]:.2f}")
                    )
                elif column == 0:
                    text = str(value)
                else:
                    text = "—" if pd.isna(value) else f"{value:.0f}"
                item = QTableWidgetItem(text)
                if poor:
                    item.setForeground(
                        self.palette().brush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
                    )
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                if column == 0:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                self.elements.setItem(row, column, item)
        self.elements.resizeColumnsToContents()
        self.scans.clear()
        for index, scan in enumerate(result.prepared.scans):
            if scan.rep_id == 0:
                self.scans.addItem(scan.label, index)
        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(2, True)
        self.tabs.setCurrentIndex(1)
        self.status.setText(
            "Scans processed. Select elements for export.\n" + "\n".join(result.prepared.notes)
        )

    def template(self) -> None:
        if not self.input_path.text():
            raise ValueError("Choose a scan folder first.")
        path, _ = QFileDialog.getSaveFileName(
            self, "Create True Depth template", "TrueDepth.xlsx", "Excel (*.xlsx)"
        )
        if path:
            self.start_work(
                partial(create_template, Path(self.input_path.text()), Path(path)),
                partial(self.template_created, path),
                "Creating template…",
            )

    def template_created(self, path: str, unused: None) -> None:
        self.depth_path.edit.setText(path)
        self.status.setText("Template created. Fill in StartID (true depth in mm) before processing.")

    def selected(self) -> list[str]:
        return [
            self.elements.item(i, 0).text()
            for i in range(self.elements.rowCount())
            if self.elements.item(i, 0).checkState() == Qt.CheckState.Checked
        ]

    def select_all(self, checked: bool) -> None:
        for i in range(self.elements.rowCount()):
            self.elements.item(i, 0).setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )

    def apply_selection(self, names: list[str]) -> None:
        available = {self.elements.item(i, 0).text() for i in range(self.elements.rowCount())}
        for i in range(self.elements.rowCount()):
            item = self.elements.item(i, 0)
            item.setCheckState(Qt.CheckState.Checked if item.text() in names else Qt.CheckState.Unchecked)
        missing = set(names) - available
        self.status.setText(
            "Selection loaded." + (f" Unavailable: {', '.join(sorted(missing))}" if missing else "")
        )

    def last_selection(self) -> None:
        self.apply_selection(str(self.preferences.value("selection", "")).splitlines())

    def load_selection(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load element selection", "", "Text (*.txt)")
        if path:
            self.apply_selection(
                [v.strip() for v in Path(path).read_text(encoding="utf-8-sig").splitlines() if v.strip()]
            )

    def save_selection(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save element selection", "elements.txt", "Text (*.txt)")
        if path:
            Path(path).write_text("\n".join(self.selected()) + "\n", encoding="utf-8")

    def export(self) -> None:
        if self.result is None:
            raise ValueError("Process scans first.")
        options = ExportOptions(
            self.selected(),
            self.inc_coh.isChecked(),
            self.ratios.text(),
            self.log_ratios.text(),
            self.detail.isChecked(),
        )
        self.start_work(
            partial(export_selection, self.result, options), self.exported, "Writing tables and plots…"
        )

    def exported(self, notes: list[str]) -> None:
        self.preferences.setValue("selection", "\n".join(self.selected()))
        self.preferences.setValue("ratios", self.ratios.text())
        self.preferences.setValue("log_ratios", self.log_ratios.text())
        self.preferences.setValue("detail", self.detail.isChecked())
        self.status.setText("Export complete.\n" + "\n".join(notes))

    def show_photo(self) -> None:
        if self.result is None or self.scans.currentIndex() < 0:
            raise ValueError("Process scans and select a photo first.")
        scan = self.result.prepared.scans[self.scans.currentData()]
        path = find_optical_image(scan.folder)
        if path is None:
            raise ValueError(f"No optical photo found for {scan.label}.")
        with Image.open(path) as image:
            data = np.asarray(image.convert("RGB").transpose(Image.Transpose.ROTATE_180))
        calibration = photo_scale(scan, data.shape[1])
        scale = (calibration[1], calibration[1]) if calibration else (1.0, 1.0)
        translate = (0.0, calibration[0]) if calibration else (0.0, 0.0)
        if self.photo_layer is not None and self.photo_layer in self.viewer.layers:
            self.viewer.layers.remove(self.photo_layer)
        self.photo_layer = self.viewer.add_image(
            data,
            rgb=True,
            name=f"{scan.label} ({'mm' if calibration else 'pixels; uncalibrated'})",
            scale=scale,
            translate=translate,
            metadata={"source": path, "depth_axis": 1},
        )
