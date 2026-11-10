from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from napari.qt.threading import FunctionWorker, create_worker
from qtpy.QtCore import Signal
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

T = TypeVar("T")


class PathEdit(QWidget):
    def __init__(self, *, directory: bool = False, save: bool = False, pattern: str = "") -> None:
        super().__init__()
        self.directory, self.save, self.pattern = directory, save, pattern
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.browse = QPushButton("Browse…")
        layout.addWidget(self.edit)
        layout.addWidget(self.browse)
        self.browse.clicked.connect(self.choose)

    def choose(self) -> None:
        if self.directory:
            path = QFileDialog.getExistingDirectory(self, "Choose folder", self.edit.text())
        elif self.save:
            path, _ = QFileDialog.getSaveFileName(self, "Save file", self.edit.text(), self.pattern)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choose file", self.edit.text(), self.pattern)
        if path:
            self.edit.setText(path)

    def text(self) -> str:
        return self.edit.text().strip()


def combo(values: list[str]) -> QComboBox:
    control = QComboBox()
    control.addItems(values)
    return control


class WorkerWidget(QWidget):
    progress_changed = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.worker: FunctionWorker | None = None
        self.layout_root = QVBoxLayout(self)
        self.body = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.body)
        self.layout_root.addWidget(scroll)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.layout_root.addWidget(self.progress)
        self.layout_root.addWidget(self.status)
        self.progress_changed.connect(self.set_progress)

    def set_progress(self, value: float) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(round(value * 100))

    def button(self, text: str, action: Callable[[], None]) -> QPushButton:
        button = QPushButton(text)

        def clicked(checked: bool = False) -> None:
            try:
                action()
            except Exception as error:
                self.show_error(error)

        button.clicked.connect(clicked)
        return button

    def show_error(self, error: Exception) -> None:
        self.status.setText(f"{type(error).__name__}: {error}")
        QMessageBox.critical(self, "ITRAX", str(error))

    def start_work(self, work: Callable[[], T], done: Callable[[T], None], message: str) -> None:
        if self.worker is not None:
            raise RuntimeError("An operation is already running.")
        self.body.setEnabled(False)
        self.status.setText(message)
        self.progress.setRange(0, 0)
        worker = create_worker(work, _start_thread=False, _ignore_errors=True)
        self.worker = worker

        def returned(value: T) -> None:
            # Clear before done(), which may schedule the next stage.
            self.worker = None
            self.body.setEnabled(True)
            self.set_progress(1)
            try:
                done(value)
            except Exception as error:
                self.show_error(error)

        def failed(error: Exception) -> None:
            self.worker = None
            self.body.setEnabled(True)
            self.set_progress(0)
            self.show_error(error)

        worker.returned.connect(returned)
        worker.errored.connect(failed)
        worker.start()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None:
            self.status.setText("Wait for the current operation to finish before closing.")
            event.ignore()
        else:
            super().closeEvent(event)


def optional_number(text: str) -> float | None:
    return float(text.strip().replace(",", ".")) if text.strip() else None
