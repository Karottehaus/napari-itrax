from __future__ import annotations

import csv
import os
import re
import shutil
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
from openpyxl.styles import PatternFill
from openpyxl.worksheet.worksheet import Worksheet

from ..models import (
    INTERNAL_COLUMNS,
    SCAN_COLUMN,
    ITRAXpyError,
    OutputFolders,
    ScanResult,
    SelectionTables,
)

INVALID_FILL = PatternFill(fill_type="solid", start_color="D9D9D9", end_color="D9D9D9")

SUSPECT_FILL = PatternFill(fill_type="solid", start_color="FFE0B2", end_color="FFE0B2")


def output_folders(output_path: str, core_label: str) -> OutputFolders:
    root = os.path.join(output_path, core_label + "_Results")
    images = os.path.join(root, core_label + "_IMG")
    return OutputFolders(
        root=root,
        xrf=os.path.join(root, core_label + "_XRF-results"),
        reps=os.path.join(root, core_label + "_XRFreps-results"),
        images_lr=os.path.join(images, "LowRes_IMG"),
        images_hr=os.path.join(images, "HighRes_IMG"),
        images_ruler=os.path.join(images, "image with ruler"),
        images_ar=os.path.join(images, "image with Ar counts"),
    )


def prepare_output_folders(
        folders: OutputFolders, high_res: bool, ruler: bool = True, argon: bool = False
) -> None:
    if os.path.exists(folders.root):
        try:
            shutil.rmtree(folders.root)
        except OSError as e:
            raise ITRAXpyError(
                f"Could not delete the previous results folder\n{folders.root}\n\n"
                f"'{e.filename}' is probably open in another program (e.g. Excel). Close it and try again."
            ) from e
    os.makedirs(folders.xrf)
    os.makedirs(folders.images_lr)
    if ruler:
        os.makedirs(folders.images_ruler)
    if argon:
        os.makedirs(folders.images_ar)
    if high_res:
        os.makedirs(folders.images_hr)


def write_run_info(folder: str, title: str, lines: list[str], notes: Sequence[str] = ()) -> None:
    """Append a section to run_info.txt in the results folder (how the results were produced)."""
    with open(os.path.join(folder, "run_info.txt"), "a", encoding="utf-8") as f:
        f.write(f"=== {title} ===\n")
        for line in lines:
            f.write(f"{line}\n")
        if notes:
            f.write("\nMessages shown to the user:\n")
            for note in notes:
                f.write("  - " + note.replace("\n", "\n    ") + "\n")
        f.write("\n")


def _grey_invalid_rows(worksheet: Worksheet, df: pd.DataFrame, header_row: int) -> None:
    """Shade rows with validity = 0 grey and suspect rows orange (header_row: 1-based Excel row of the names)."""
    for name, flag_value, fill in (("suspect", 1, SUSPECT_FILL), ("validity", 0, INVALID_FILL)):  # grey wins
        if name not in df.columns:
            continue
        for i in np.flatnonzero(pd.to_numeric(df[name], errors="coerce").eq(flag_value).to_numpy()):
            for column in range(1, df.shape[1] + 1):
                worksheet.cell(row=header_row + 1 + int(i), column=column).fill = fill


def write_table(df: pd.DataFrame, path_without_extension: str, file_format: str) -> None:
    df = df.drop(columns=INTERNAL_COLUMNS, errors="ignore")
    if file_format == "xlsx":
        with pd.ExcelWriter(path_without_extension + ".xlsx", engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Sheet1", index=False)
            _grey_invalid_rows(writer.sheets["Sheet1"], df, 1)
    else:
        df.to_csv(path_without_extension + ".csv", index=False)


def write_workbook(tables: dict[str, pd.DataFrame], path_without_extension: str) -> None:
    """Several tables as the tabs of one xlsx workbook ({tab name: table}), invalid and suspect rows shaded."""
    with pd.ExcelWriter(path_without_extension + ".xlsx", engine="openpyxl") as writer:
        for sheet, df in tables.items():
            df = df.drop(columns=INTERNAL_COLUMNS, errors="ignore")
            df.to_excel(writer, sheet_name=sheet, index=False)
            _grey_invalid_rows(writer.sheets[sheet], df, 1)


def write_scan_tables(results: list[ScanResult], folders: OutputFolders, file_format: str) -> None:
    """One table per main scan; one table per core with all of its replicates."""
    replicates: dict[str, list[pd.DataFrame]] = {}
    for scan, table, _ in results:
        if scan.rep_id == 0:
            write_table(table, os.path.join(folders.xrf, scan.label), file_format)
        else:
            replicates.setdefault(scan.label, []).append(table)
    if replicates:
        os.makedirs(folders.reps, exist_ok=True)
        for label, tables in replicates.items():
            write_table(
                pd.concat(tables, ignore_index=True, sort=False),
                os.path.join(folders.reps, label + "-xrfreps"),
                file_format,
            )


def _sheet_name(label: str, suffix: str) -> str:
    label = re.sub(r"[\[\]:*?/\\]", "_", label)
    return label[: 31 - len(suffix)] + suffix


def _write_sheet(
        writer: pd.ExcelWriter, df: pd.DataFrame, sheet: str, core_label: str, settings: str
) -> None:
    df.to_excel(writer, sheet_name=sheet, startrow=2, index=False)
    worksheet = writer.sheets[sheet]
    worksheet.cell(row=1, column=1, value=core_label)
    worksheet.cell(row=2, column=1, value=settings)
    _grey_invalid_rows(worksheet, df, 3)


def _write_csv(df: pd.DataFrame, path: str, core_label: str, settings: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([core_label])
        writer.writerow([settings])
        df.to_csv(f, index=False)


def write_selection_tables(
        tables: SelectionTables,
        *,
        combined: pd.DataFrame,
        core_label: str,
        settings: str,
        true_depth: pd.DataFrame,
        output_folder: str,
        file_format: str,
        core_workbooks: bool = False,
) -> None:
    """Write already calculated tables, preserving the combined and per-scan layouts."""
    slct_table, norm_table, clr_table = tables.selected, tables.normalised, tables.clr
    meta = tables.meta
    per_scatter, per_kcps, other = tables.per_scatter, tables.per_kcps, tables.ratios
    if file_format == "xlsx":
        with pd.ExcelWriter(
                os.path.join(output_folder, core_label + "_slct.xlsx"), engine="openpyxl"
        ) as writer:
            _write_sheet(writer, slct_table, _sheet_name(core_label, "_-slct"), core_label, settings)
        with pd.ExcelWriter(
                os.path.join(output_folder, core_label + "_norm.xlsx"), engine="openpyxl"
        ) as writer:
            _write_sheet(writer, norm_table, _sheet_name(core_label, "-norm"), core_label, settings)
            _write_sheet(writer, clr_table, _sheet_name(core_label, "-clr"), core_label, settings)
            true_depth.to_excel(writer, sheet_name="True-Depth", index=False)
    else:
        for suffix, table in (("slct", slct_table), ("norm", norm_table), ("clr", clr_table)):
            _write_csv(table, os.path.join(output_folder, f"{core_label}_{suffix}.csv"), core_label, settings)
        true_depth.to_csv(os.path.join(output_folder, f"{core_label}_True-Depth.csv"), index=False)

    # (sheet name suffix in _ratios.xlsx, csv file name, table)
    sheets = [
        (suffix, csv_name, pd.concat([meta, table], axis=1))
        for suffix, csv_name, table in (
            ("-el_incoh", "per_incoh", per_scatter),
            ("-el_kcps", "per_kcps", per_kcps),
            ("-ratios", "cohinc_and_ratios", other),
        )
        if not table.empty
    ]

    # the same values split by scan (replicates: by core), like the per-scan tables in the _XRF-results folder:
    # a folder per treatment with one file per scan, or (xlsx, core_workbooks) one workbook per scan with all tabs
    if SCAN_COLUMN in combined.columns:
        replicate = pd.to_numeric(combined["RepID"], errors="coerce").fillna(0).ne(0)
        parts = cast(
            dict[tuple[str, bool], pd.Index],
            combined.groupby([combined[SCAN_COLUMN], replicate], sort=False).groups,
        ).items()
        if core_workbooks and file_format == "xlsx":
            folder = os.path.join(output_folder, f"{core_label}_per-core")
            os.makedirs(folder, exist_ok=True)
            tabs = [("slct", slct_table), ("norm", norm_table), ("clr", clr_table)] + [
                (suffix[1:], table) for suffix, _, table in sheets
            ]
            for (label, is_replicate), index in parts:
                write_workbook(
                    {tab: table.loc[index] for tab, table in tabs},
                    os.path.join(folder, f"{label}-xrfreps" if is_replicate else label),
                )
        else:
            for suffix, table in (("slct", slct_table), ("norm", norm_table), ("clr", clr_table)):
                folder = os.path.join(output_folder, f"{core_label}_{suffix}-results")
                os.makedirs(folder, exist_ok=True)
                for (label, is_replicate), index in parts:
                    name = f"{label}-xrfreps_{suffix}" if is_replicate else f"{label}_{suffix}"
                    write_table(table.loc[index], os.path.join(folder, name), file_format)

    if file_format == "xlsx" and sheets:
        with pd.ExcelWriter(
                os.path.join(output_folder, core_label + "_ratios.xlsx"), engine="openpyxl"
        ) as writer:
            for suffix, _, table in sheets:
                _write_sheet(writer, table, _sheet_name(core_label, suffix), core_label, settings)
    else:
        for _, csv_name, table in sheets:
            _write_csv(
                table,
                os.path.join(output_folder, f"{core_label}_ratios_{csv_name}.csv"),
                core_label,
                settings,
            )
