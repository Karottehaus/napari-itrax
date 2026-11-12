from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import pandas as pd

Photo = tuple[str, float, float]

ScanPhoto = tuple[str, str, str, Photo | None]

ArgonCurve = tuple[pd.Series | None, ...]


class Settings(TypedDict, total=False):
    voltage: float | None
    current: float | None
    time: float | None
    tube: str


class Document(Settings, total=False):
    sample: str
    section: str
    user: str
    date: str
    hour: int
    start: float
    stop: float
    optical_start: float
    optical_end: float


class RunInfo(TypedDict, total=False):
    version: str
    data_folder: str
    td_file: str
    scans: int
    cores: int
    measurements: int
    left_out: int
    settings: str
    file_format: str
    core_workbooks: bool
    suspect_factor: float | None
    selected: list[str] | None
    operator: str
    measured: list[tuple[str, int | None]]
    invalid_points: str
    ratios: list[tuple[str, str]]
    log_ratios: list[tuple[str, str]]


GROUP_COLUMN = "_core_group"  # internal: which core a row belongs to (for the zero replacement)

SCAN_COLUMN = "_scan"  # internal: which scan a row comes from (for the per-scan norm/clr files)

INTERNAL_COLUMNS = [GROUP_COLUMN, SCAN_COLUMN]

INVALID_EMPTY, INVALID_KEEP, INVALID_LEAVE_OUT = "empty", "keep", "leave out"

INVALID_POINTS = {
    INVALID_EMPTY: "rows kept, values empty",
    INVALID_KEEP: "rows kept with values",
    INVALID_LEAVE_OUT: "rows left out",
}


class ITRAXpyError(Exception):
    """A problem with the input that the user can fix. The message is shown as-is in the GUI."""


@dataclass
class ScanFolder:
    """A folder containing an ITRAX results file."""

    folder: str
    results_file: str
    names: list[str]  # folder names from the input folder (first) down to the scan folder (last)
    paths: list[str]  # the matching absolute paths
    document: Document  # values read from document.txt (may be empty)

    @property
    def display(self) -> str:
        return "/".join(self.names)


@dataclass
class Scan:
    core: str  # 'Core' name of the matched True Depth row
    td_row: pd.Series
    section: int | None  # section number, if the core has several sections
    rep_id: int  # 0 = main scan, >0 = replicate
    folder: str  # folder holding the results file and optical image
    results_file: str
    label: str  # base name of the per-scan output table
    core_dir: str  # folder whose name matched the core (searched for high-res images)
    document: Document
    display: str
    settings: Settings = field(default_factory=Settings)
    photo: Photo | None = None  # (saved LR image, depth of its top, depth of its bottom) in mm


@dataclass
class OutputFolders:
    root: str
    xrf: str
    reps: str
    images_lr: str
    images_hr: str
    images_ruler: str
    images_ar: str


ScanResult = tuple[Scan, pd.DataFrame, list[str]]


@dataclass
class SelectionTables:
    """Calculated export tables and diagnostics, independent of file format."""

    meta: pd.DataFrame
    counts: pd.DataFrame
    selected: pd.DataFrame
    normalised: pd.DataFrame
    clr: pd.DataFrame
    per_scatter: pd.DataFrame
    per_kcps: pd.DataFrame
    ratios: pd.DataFrame
    notes: list[str]
