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
