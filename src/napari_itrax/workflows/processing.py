from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from napari_itrax import __version__
from napari_itrax.reporting import plots, readme_pdf
from napari_itrax.analysis import (
    DEFAULT_LOG_RATIOS,
    DEFAULT_RATIOS,
    all_zero,
    calculate_selection,
    check_depths,
    combine,
    cv_percent,
    flag_suspect,
    is_element,
    main_rows,
    noise_percent,
    overlaps,
    parse_ratio_pairs,
    repeat_percent,
    zeroes_percent,
)
from napari_itrax.reporting.exports import (
    output_folders,
    prepare_output_folders,
    write_run_info,
    write_scan_tables,
    write_selection_tables,
)
from napari_itrax.imaging.images import save_scan_images, scan_photos
from napari_itrax.models import (
    GROUP_COLUMN,
    INVALID_POINTS,
    OutputFolders,
    RunInfo,
    Scan,
    ScanResult,
    Settings,
)
from napari_itrax.readers import check_scans, find_scan_folders, load_true_depth, read_scans
from napari_itrax.scans import assign_scans, resolve_settings, true_depth_template

log = logging.getLogger("itraxpy")


@dataclass(frozen=True)
class ProcessingOptions:
    input_folder: Path
    true_depth: Path
    output_folder: Path
    settings: Settings = field(default_factory=Settings)
    replicates: bool = False
    high_resolution: bool = False
    ruler_images: bool = True
    argon_images: bool = True
    invalid_points: Literal["empty", "keep", "leave out"] = "empty"
    suspect_factor: float | None = 1.5
    file_format: Literal["xlsx", "csv"] = "xlsx"
    core_workbooks: bool = False
    operator: str = ""
    error_log: bool = False


@dataclass
class PreparedRun:
    options: ProcessingOptions
    scans: list[Scan]
    true_depth: pd.DataFrame
    folders: OutputFolders
    label: str
    settings: str
    tube: str
    notes: list[str]
    tables: list[ScanResult]
    combined: pd.DataFrame
    elements: list[str]
    meta_columns: list[str]


@dataclass
class ProcessingResult:
    prepared: PreparedRun
    combined: pd.DataFrame
    elements: list[str]
    meta_columns: list[str]
    info: RunInfo


@dataclass(frozen=True)
class ExportOptions:
    selected: list[str]
    include_inc_coh: bool = False
    ratios: str = DEFAULT_RATIOS
    log_ratios: str = DEFAULT_LOG_RATIOS
    detail_pages: bool = False


def prepare_run(options: ProcessingOptions) -> PreparedRun:
    """Read and validate inputs without changing the output directory."""
    if options.invalid_points not in INVALID_POINTS:
        raise ValueError("Unknown invalid-point policy.")
    if options.file_format not in ("xlsx", "csv"):
        raise ValueError("File format must be xlsx or csv.")
    if options.suspect_factor is not None and (
            not math.isfinite(options.suspect_factor) or options.suspect_factor <= 0
    ):
        raise ValueError("The suspect factor must be finite and positive, or empty.")
    for key, value in (
            ("voltage", options.settings.get("voltage")),
            ("current", options.settings.get("current")),
            ("time", options.settings.get("time")),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be finite and positive, or empty.")
    notes: list[str] = []
    depth, raw = load_true_depth(str(options.true_depth), notes)
    found = find_scan_folders(str(options.input_folder), notes)
    scans = assign_scans(found, depth, options.replicates, notes)
    settings, tube = resolve_settings(scans, options.settings, notes)
    check_scans(scans)
    tables = read_scans(scans, notes)
    if options.suspect_factor is not None:
        flag_suspect(tables, options.suspect_factor, notes)
    check_depths(tables, notes)
    combined, elements, meta = combine(tables, options.invalid_points, notes)
    label = options.input_folder.resolve().name
    folders = output_folders(str(options.output_folder), label)
    target = Path(folders.root).resolve()
    sources = [options.input_folder.resolve(), options.true_depth.resolve()]
    if any(source == target or target in source.parents for source in sources):
        raise ValueError("The results directory must not contain the input data or True Depth file.")
    return PreparedRun(
        options, scans, raw, folders, label, settings, tube, notes, tables, combined, elements, meta
    )


@contextmanager
def run_logging(folder: str, enabled: bool) -> Iterator[None]:
    """Attach a handler only for the duration of this operation."""
    handler = logging.FileHandler(Path(folder) / "itrax.log", encoding="utf-8") if enabled else None
    if handler is not None:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)
    try:
        yield
    except Exception:
        log.exception("ITRAX operation failed")
        raise
    finally:
        if handler is not None:
            log.removeHandler(handler)
            handler.close()


def process_run(
        prepared: PreparedRun,
        *,
        overwrite: bool = False,
        progress: Callable[[float], None] | None = None,
) -> ProcessingResult:
    """Process a validated run. Existing results require explicit authorization."""
    p = prepared
    o = p.options
    if Path(p.folders.root).exists() and not overwrite:
        raise FileExistsError(p.folders.root)
    prepare_output_folders(p.folders, o.high_resolution, o.ruler_images, o.argon_images)
    with run_logging(p.folders.root, o.error_log):
        tables = p.tables
        save_scan_images(
            tables,
            p.folders,
            o.high_resolution,
            p.notes,
            progress=progress,
            ruler=o.ruler_images,
            argon=o.argon_images,
        )
        combined = p.combined
        write_scan_tables(tables, p.folders, o.file_format)
        measurements = sum(len(table) for _, table, _ in tables)
        info: RunInfo = dict(
            version=__version__,
            data_folder=str(o.input_folder),
            td_file=str(o.true_depth),
            scans=len(p.scans),
            cores=int(combined[GROUP_COLUMN].nunique()),
            measurements=measurements,
            left_out=measurements - len(combined),
            settings=p.settings,
            file_format=o.file_format,
            core_workbooks=o.core_workbooks and o.file_format == "xlsx",
            suspect_factor=o.suspect_factor,
            selected=None,
            operator=o.operator
                     or ", ".join(sorted({s.document["user"] for s in p.scans if s.document.get("user")})),
            measured=[
                (s.document["date"], s.document.get("hour")) for s in p.scans if s.document.get("date")
            ],
            invalid_points=o.invalid_points,
        )
        write_run_info(
            p.folders.root,
            f"Run {datetime.now().isoformat(timespec='seconds')}",
            [f"napari-itrax {__version__}", repr(o), p.settings]
            + [f"{s.label} <- {s.display}" for s in p.scans],
            p.notes,
        )
        readme_pdf.write_readme(p.folders.root, p.label, info)
        return ProcessingResult(p, combined, p.elements, p.meta_columns, info)


def export_selection(result: ProcessingResult, options: ExportOptions) -> list[str]:
    """Export scientific tables, profile plots, and the updated results guide."""
    p = result.prepared
    if not options.selected or any(e not in result.elements or not is_element(e) for e in options.selected):
        raise ValueError("Select at least one measured element; every selection must exist in this run.")
    ratios = parse_ratio_pairs(options.ratios)
    logs = parse_ratio_pairs(options.log_ratios)
    missing = {e for pair in ratios + logs for e in pair if e not in result.elements or not is_element(e)}
    if missing:
        raise ValueError(f"Ratio elements absent from this run: {', '.join(sorted(missing))}")
    with run_logging(p.folders.root, p.options.error_log):
        tables = calculate_selection(
            result.combined,
            result.meta_columns,
            options.selected,
            tube=p.tube,
            add_inc_coh=options.include_inc_coh,
            log_ratios=logs,
            ratios=ratios,
        )
        write_selection_tables(
            tables,
            combined=result.combined,
            core_label=p.label,
            settings=p.settings,
            true_depth=p.true_depth,
            output_folder=p.folders.root,
            file_format=p.options.file_format,
            core_workbooks=p.options.core_workbooks,
        )
        notes = tables.notes
        plots.write_profiles(
            str(Path(p.folders.root) / f"{p.label}_plots"),
            result.combined,
            p.label,
            p.settings,
            {"counts": tables.counts, "ratios": tables.ratios},
            scan_photos(p.scans),
            options.detail_pages,
            notes,
        )
        result.info["selected"] = options.selected
        result.info["ratios"] = ratios
        result.info["log_ratios"] = logs
        write_run_info(p.folders.root, "Export selected elements", [repr(options)], notes)
        readme_pdf.write_readme(p.folders.root, p.label, result.info)
    return notes


def quality_table(result: ProcessingResult) -> pd.DataFrame:
    """Element-selection metrics, retaining the original definitions."""
    data = result.combined
    shown = [
        e
        for e, zero in zip(result.elements, all_zero(data[result.elements]), strict=True)
        if not zero and is_element(e)
    ]
    main = main_rows(data, shown)
    if main.empty:
        main = data
    return pd.DataFrame(
        {
            "Element": shown,
            "Zeros": zeroes_percent(main[shown]),
            "CV": cv_percent(main[shown]),
            "Noise": noise_percent(main, shown),
            "Repeat": repeat_percent(data, shown) or [None] * len(shown),
            "Overlap": overlaps(main, shown, result.prepared.tube),
        }
    )


def create_template(input_folder: Path, destination: Path) -> None:
    true_depth_template(find_scan_folders(str(input_folder))).to_excel(destination, index=False)
