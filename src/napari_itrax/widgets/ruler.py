from __future__ import annotations

from functools import partial
from pathlib import Path

import napari
import numpy as np
from napari.layers import Image as ImageLayer
from napari.layers import Points
from napari.utils.events import Event
from PIL import Image
from qtpy.QtCore import Qt
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHeaderView,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
)

from napari_itrax.imaging import ruler
from napari_itrax.workflows.ruler import PhotoJob, export_rulers
from napari_itrax.imaging.ruler import measured_scale
from napari_itrax.widgets.common import PathEdit, WorkerWidget, combo, optional_number


class RulerWidget(WorkerWidget):
    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.photos: list[Path] = []
        self.measurements: dict[Path, tuple[str, float]] = {}
        self.image_layer: ImageLayer | None = None
        self.markers: Points | None = None
        self.preview_path: Path | None = None
        form = QFormLayout(self.body)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow(self.button("Choose photos…", self.choose_photos))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Photo", "Top depth (cm)", "ITRAX document"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        form.addRow(self.table)
        form.addRow(self.button("Choose document for selected photo…", self.choose_document))
        form.addRow(self.button("Choose document for all photos…", self.common_document))
        form.addRow(self.button("Find documents automatically", self.auto_documents))
        self.core_top = combo(["left", "right", "top", "bottom"])
        self.mode = combo(["dpi", "pixel", "length", "document", "measured"])
        self.value = QLineEdit()
        form.addRow("Core top in original photo", self.core_top)
        form.addRow("Scale source", self.mode)
        form.addRow("DPI / mm per pixel / full photo length (mm)", self.value)
        self.reference_length = QLineEdit()
        form.addRow("Distance between markers (mm)", self.reference_length)
        form.addRow(self.button("Show selected photo and markers", self.preview))
        form.addRow(self.button("Use measurement for selected photo", self.measure_selected))
        form.addRow(self.button("Use measurement for all photos", self.measure_all))
        self.core_depth = QCheckBox("Core-depth ruler")
        self.core_depth.setChecked(True)
        self.photo_depth = QCheckBox("Photo-depth ruler")
        self.photo_depth.setChecked(True)
        self.vertical = QCheckBox("Export upright, with rulers on the right")
        form.addRow(self.core_depth)
        form.addRow(self.photo_depth)
        form.addRow(self.vertical)
        self.output = PathEdit(directory=True)
        form.addRow("Output folder", self.output)
        form.addRow(self.button("Export ruler photos", self.export))

    def choose_photos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose core photos", "", "Images (*.tif *.tiff *.jpg *.jpeg *.png *.bmp)"
        )
        if paths:
            self.set_photos([Path(p) for p in paths])

    def set_photos(self, paths: list[Path]) -> None:
        self.disconnect_markers()
        self.photos = paths
        self.measurements.clear()
        self.preview_path = None
        self.table.setRowCount(len(paths))
        for row, path in enumerate(paths):
            item = QTableWidgetItem(str(path))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 0, item)
            self.table.setItem(row, 1, QTableWidgetItem(""))
            self.table.setItem(row, 2, QTableWidgetItem(ruler.find_document(str(path)) or ""))
        if paths:
            self.table.selectRow(0)
            self.output.edit.setText(str(paths[0].parent / "photos with ruler"))
            dpi = ruler.image_dpi(str(paths[0]))
            self.value.setText("" if dpi is None else str(dpi))

    def selected_row(self) -> int:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.photos):
            raise ValueError("Select a photo first.")
        return row

    def choose_document(self) -> None:
        row = self.selected_row()
        path, _ = QFileDialog.getOpenFileName(self, "Choose ITRAX document", "", "Text (*.txt)")
        if path:
            self.table.item(row, 2).setText(path)

    def common_document(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose common ITRAX document", "", "Text (*.txt)")
        if path:
            for row in range(len(self.photos)):
                self.table.item(row, 2).setText(path)

    def auto_documents(self) -> None:
        for row, path in enumerate(self.photos):
            self.table.item(row, 2).setText(ruler.find_document(str(path)) or "")

    def disconnect_markers(self) -> None:
        if self.markers is not None:
            self.markers.events.data.disconnect(self.marker_changed)

    def preview(self) -> None:
        path = self.photos[self.selected_row()]
        with Image.open(path) as image:
            data = np.asarray(image.convert("RGB"))
        self.disconnect_markers()
        for layer in (self.markers, self.image_layer):
            if layer is not None and layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self.preview_path = path
        self.image_layer = self.viewer.add_image(
            data, rgb=True, name=f"{path.name} (pixels)", metadata={"source": str(path)}
        )
        h, w = data.shape[:2]
        points = (
            [[h / 2, w / 4], [h / 2, 3 * w / 4]]
            if self.core_top.currentText() in ("left", "right")
            else [[h / 4, w / 2], [3 * h / 4, w / 2]]
        )
        self.markers = self.viewer.add_points(
            points,
            name="ITRAX calibration markers",
            size=12,
            features={"endpoint": ["start", "end"]},
            text="endpoint",
        )
        self.markers.mode = "select"
        self.markers.events.data.connect(self.marker_changed)
        self.status.setText(
            "Move the two markers along a known reference length, then enter its distance in mm."
        )

    def marker_changed(self, event: Event) -> None:
        self.status.setText("Markers changed. Apply the measurement to update calibration.")

    def calibration(self) -> float:
        if self.markers is None or self.image_layer is None or self.preview_path is None:
            raise ValueError("Show a photo and place its markers first.")
        if self.markers not in self.viewer.layers or self.image_layer not in self.viewer.layers:
            raise ValueError("The calibration layers were removed. Show the photo again.")
        if len(self.markers.data) != 2:
            raise ValueError("Calibration requires exactly two markers.")
        length = optional_number(self.reference_length.text())
        if length is None:
            raise ValueError("Enter the reference length in mm.")
        # Use public layer transforms: camera zoom and independent layer transforms do not affect the result.
        pixels = [self.image_layer.world_to_data(self.markers.data_to_world(p)) for p in self.markers.data]
        axis = 1 if self.core_top.currentText() in ("left", "right") else 0
        return measured_scale(length, pixels[0][axis], pixels[1][axis])

    def measure_selected(self) -> None:
        path = self.photos[self.selected_row()]
        if path != self.preview_path:
            raise ValueError("Show the selected photo before measuring it.")
        self.apply_measurement([path])

    def measure_all(self) -> None:
        self.apply_measurement(self.photos)

    def apply_measurement(self, paths: list[Path]) -> None:
        scale = self.calibration()
        orientation = self.core_top.currentText()
        for path in paths:
            self.measurements[path] = (orientation, scale)
        self.mode.setCurrentText("measured")
        if self.image_layer is not None and self.markers is not None:
            world = [self.markers.data_to_world(p) for p in self.markers.data]
            pixels = np.asarray([self.image_layer.world_to_data(p) for p in world])
            self.image_layer.scale = (scale, scale)
            self.markers.scale = self.image_layer.scale
            self.markers.translate = self.image_layer.translate
            self.markers.rotate = self.image_layer.rotate
            self.markers.shear = self.image_layer.shear
            self.markers.affine = self.image_layer.affine
            self.markers.data = pixels
            self.image_layer.name = f"{self.preview_path.name} (mm)" if self.preview_path else "Photo (mm)"
        self.status.setText(
            f"Applied {scale:.6g} mm/pixel to {len(paths)} photo(s)."
            + (" Batch calibration assumes the same camera scale for every photo." if len(paths) > 1 else "")
        )

    def jobs(self) -> list[PhotoJob]:
        jobs: list[PhotoJob] = []
        orientation, mode = self.core_top.currentText(), self.mode.currentText()
        for row, path in enumerate(self.photos):
            with Image.open(path) as image:
                length = image.width if orientation in ("left", "right") else image.height
            if mode == "document":
                document = self.table.item(row, 2).text().strip()
                if not document:
                    raise ValueError(f"Choose an ITRAX document for {path.name}.")
                scale = ruler.document_scale(str(path), length, document)
            elif mode == "measured":
                measurement = self.measurements.get(path)
                if measurement is None or measurement[0] != orientation:
                    raise ValueError(f"Calibrate {path.name} for the selected core orientation.")
                scale = measurement[1]
            else:
                scale = ruler.mm_per_pixel(mode, optional_number(self.value.text()), length)
            if scale is None:
                raise ValueError(f"A positive scale is required for {path.name}.")
            depth = optional_number(self.table.item(row, 1).text()) if self.core_depth.isChecked() else None
            if self.core_depth.isChecked() and depth is None:
                raise ValueError(
                    f"Enter the core top depth for {path.name}, or disable its core-depth ruler."
                )
            jobs.append(
                PhotoJob(
                    path, scale, orientation, depth, self.photo_depth.isChecked(), self.vertical.isChecked()
                )
            )
        return jobs

    def export(self) -> None:
        if not self.output.text():
            raise ValueError("Choose an output folder.")
        self.start_work(
            partial(export_rulers, self.jobs(), Path(self.output.text()), self.progress_changed.emit),
            self.exported,
            "Writing ruler photos…",
        )

    def exported(self, paths: list[Path]) -> None:
        self.status.setText(f"Exported {len(paths)} ruler photos to {self.output.text()}.")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is None:
            self.disconnect_markers()
        super().closeEvent(event)
