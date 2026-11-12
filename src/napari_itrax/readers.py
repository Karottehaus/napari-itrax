from __future__ import annotations

import io
import os
import re
from collections.abc import Iterable, Iterator
from typing import cast
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from natsort import natsorted, ns
from openpyxl.utils.exceptions import InvalidFileException

from .models import (
    GROUP_COLUMN,
    SCAN_COLUMN,
    Document,
    ITRAXpyError,
    Scan,
    ScanFolder,
    ScanResult,
)

DROP_COLUMNS = ["sample surface", "E-gain", "E-offset", "F-slope", "F-offset"]

COUNT_COLUMNS = ["validity", "cps", "kcps", "Dt", "MSE"]

POSITION_COLUMNS = ["position (mm)", "X Position (mm)"]

TD_COLUMNS = ["Core", "CoreID", "SectionID", "StartID", "StartPosition"]

MAX_LISTED = 12


def normalise_name(text: object) -> str:
    return str(text).strip().lower()


def has_value(value: object) -> bool:
    return (
            value is not None and not (isinstance(value, float) and np.isnan(value)) and str(value).strip() != ""
    )


def parse_number(text: object) -> float | None:
    try:
        value = float(str(text).strip().replace(",", "."))
    except ValueError:
        return None
    return None if np.isnan(value) else value


def format_number(value: object) -> str:
    value = float(str(value))
    return str(int(value)) if value.is_integer() else f"{value:g}"


def describe_items(items: Iterable[object]) -> str:
    items = list(items)
    shown = "\n".join("  " + str(i) for i in items[:MAX_LISTED])
    return shown + (f"\n  ... and {len(items) - MAX_LISTED} more" if len(items) > MAX_LISTED else "")


def find_results_file(folder: str, files: list[str] | None = None) -> str | None:
    """Return the ITRAX results file directly inside folder, or None.

    Like the original ITRAXpy, the re-evaluated 'Results.txt' wins over the instrument's
    'result.txt'. Q-spec style names such as 'result_<name>.txt' are accepted as a fallback.
    """
    if files is None:
        try:
            files = [e.name for e in os.scandir(folder) if e.is_file()]
        except OSError:
            return None
    by_lower = {f.lower(): f for f in files}
    for name in ("results.txt", "result.txt"):
        if name in by_lower:
            return os.path.join(folder, by_lower[name])
    for f in natsorted(files, alg=ns.IGNORECASE):
        if re.fullmatch(r"results?_.+\.txt", f, re.IGNORECASE):
            return os.path.join(folder, f)
    return None


def read_document(folder: str) -> Document:
    """Read the scan settings the ITRAX writes into document.txt.

    Returns a dict with any of: sample, section, user (the operator's User ID), date ('YYYY-MM-DD') and hour of the
    scan, voltage (kV), current (mA), time (s), tube, start, stop (mm), optical_start, optical_end (mm).
    """
    return read_document_file(os.path.join(folder, "document.txt"))


def read_document_file(path: str) -> Document:
    """read_document for an ITRAX document file of any name (e.g. a renamed copy)."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="latin-1") as f:
            lines = [line.rstrip("\r\n").split("\t") for line in f]
    except OSError:
        return {}

    doc: dict[str, str | float | None] = {}
    for i, fields in enumerate(lines):
        key = fields[0].strip().rstrip(":").lower()
        values = [v.strip() for v in fields[1:]]
        if key == "base path" and i + 1 < len(lines):
            header = [h.strip().rstrip(":").lower() for h in fields]
            for name, value in zip(header, lines[i + 1]):
                if name in ("sample name", "section name", "user id") and value.strip():
                    doc[name.split()[0]] = value.strip()  # sample, section, user (the operator)
        elif key == "year" and i + 1 < len(lines):
            # the start of the scan: Year Month Day Time (the hour) on the next line
            parts = [parse_number(v) for v in lines[i + 1][:4]]
            if len(parts) >= 3 and None not in parts[:3]:
                doc["date"] = (
                    f"{int(cast(float, parts[0])):04d}-{int(cast(float, parts[1])):02d}-{int(cast(float, parts[2])):02d}"
                )
                if len(parts) == 4 and parts[3] is not None:
                    doc["hour"] = int(parts[3])
        elif key == "xrf voltage and current":
            numbers = [n for n in map(parse_number, values) if n is not None]
            if len(numbers) >= 2:
                doc["voltage"], doc["current"] = numbers[:2]
        elif key == "xrf exp. time" and values:
            doc["time"] = parse_number(values[0])
        elif key == "tube" and values and values[0]:
            doc["tube"] = values[0]
        elif key == "start coordinate" and values:
            doc["start"] = parse_number(values[0])
            if len(values) >= 3:
                doc["stop"] = parse_number(values[2])
        elif key == "optical start" and len(values) >= 3:
            # scanner coordinates (mm) of the two ends of optical.tif
            doc["optical_start"], doc["optical_end"] = parse_number(values[0]), parse_number(values[2])
    return cast(Document, {k: v for k, v in doc.items() if v is not None})


def _is_spectra_folder(name: str) -> bool:
    """The ITRAX's 'XRF data' folder: thousands of .spe spectra and never a scan, so it is not searched."""
    return re.fullmatch(r"xrf[\s_\-]*data", name, re.IGNORECASE) is not None


def _walk(
        input_folder: str, unreadable: list[OSError] | None = None
) -> Iterator[tuple[str, list[str], list[str]]]:
    """os.walk in natural order, skipping the spectra folders (slow on network drives). Folders that cannot be
    listed are added to unreadable."""
    for root, dirs, files in os.walk(
            input_folder, onerror=unreadable.append if unreadable is not None else None
    ):
        dirs[:] = natsorted((d for d in dirs if not _is_spectra_folder(d)), alg=ns.IGNORECASE)
        yield root, dirs, files


def find_scan_folders(input_folder: str, notes: list[str] | None = None) -> list[ScanFolder]:
    """Every folder at or below input_folder that holds a results file, in natural order."""
    input_folder = os.path.normpath(input_folder)
    if not os.path.isdir(input_folder):
        raise ITRAXpyError(f"The core scanning data folder does not exist:\n{input_folder}")
    base = os.path.basename(input_folder) or input_folder
    found: list[ScanFolder] = []
    unreadable: list[OSError] = []
    for root, dirs, files in _walk(input_folder, unreadable):
        results = find_results_file(root, files)
        if not results:
            continue
        relative = os.path.relpath(root, input_folder)
        parts = [] if relative == "." else relative.split(os.sep)
        names = [base] + parts
        paths = [os.path.join(input_folder, *parts[:k]) for k in range(len(parts) + 1)]
        found.append(ScanFolder(root, results, names, paths, read_document(root)))
    cannot = [e.filename or str(e) for e in unreadable]
    if cannot and notes is not None:
        notes.append(
            f"{len(cannot)} folder(s) could not be read and were not searched (a path longer than 260 "
            f"characters, or no permission):\n{describe_items(cannot)}"
        )
    if not found:
        raise ITRAXpyError(
            "No ITRAX results files (Results.txt / result.txt) were found in\n"
            f"{input_folder}\nor any of its subfolders."
            + (
                f"\n\n{len(cannot)} folder(s) could not be read (path longer than 260 characters?):\n"
                f"{describe_items(cannot)}"
                if cannot
                else ""
            )
        )
    return found


def load_true_depth(path: str, notes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (table with standard column names, table as read for the output workbook)."""
    if os.path.splitext(str(path))[1].lower() not in (".xlsx", ".xlsm"):
        raise ITRAXpyError(
            f"The True Depth file must be an Excel workbook (.xlsx):\n{path}\n\nOpen it in Excel and "
            "save it as 'Excel Workbook (*.xlsx)', or use Create... to make one."
        )
    try:
        raw = pd.read_excel(path)
    except (OSError, ValueError, BadZipFile, InvalidFileException, ParseError) as e:
        raise ITRAXpyError(f"Could not read the True Depth file:\n{path}\n\n{e}") from e
    raw = raw.dropna(how="all")
    td = raw.copy()
    td.columns = [str(c).strip() for c in td.columns]

    for name in ("CoreID", "SectionID"):
        if name not in td.columns:
            raise ITRAXpyError(
                f"The True Depth file has no '{name}' column.\n\nExpected columns: {' | '.join(TD_COLUMNS)}"
            )
    # older templates: StartID / StartPosition were only ever read by position (4th and 5th column)
    for name, position in (("StartID", 3), ("StartPosition", 4)):
        if name not in td.columns:
            if len(td.columns) <= position or td.columns[position] in TD_COLUMNS + ["Folder"]:
                raise ITRAXpyError(
                    f"The True Depth file has no '{name}' column.\n\n"
                    f"Expected columns: {' | '.join(TD_COLUMNS)}"
                )
            notes.append(f"True Depth file: column '{td.columns[position]}' was used as {name}.")
            td = td.rename(columns={td.columns[position]: name})
    if "Core" not in td.columns:
        td = td.rename(columns={td.columns[0]: "Core"})
    # names as text: a column of numbers with one empty cell is read as 101.0, 102.0
    for name in ("Core", "Folder"):
        if name in td.columns:
            td[name] = td[name].map(lambda v: id_text(v).strip() if has_value(v) else None)
    td = td[td["Core"].map(has_value)].reset_index(drop=True)
    if td.empty:
        raise ITRAXpyError("The True Depth file contains no rows with a 'Core' name.")
    return td, raw


def core_group(scan: Scan) -> str:
    core_id = scan.td_row.get("CoreID")
    return id_text(core_id) if has_value(core_id) else scan.core


def _results_lines(path: str) -> tuple[list[str], str | None]:
    """The lines of a results file, from the line with the column names on (line 3 as Q-Spec writes it).

    The scanner PC writes in the Windows code page (an umlaut in a sample name is one byte), newer tools in
    UTF-8; a copy saved by Excel as 'Unicode Text' is UTF-16. Returns (lines, error message or None).
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        return [], f"Could not read\n{path}\n\n{e}"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16", errors="replace")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("cp1252", errors="replace")
    lines = text.splitlines()
    for i, line in enumerate(lines[:10]):
        if any(c in [part.strip() for part in line.split("\t")] for c in POSITION_COLUMNS):
            if not any(rest.strip() for rest in lines[i + 1:]):
                return [], (
                    f"{path}\ncontains no measurements, only the header lines (e.g. an aborted scan). "
                    "Remove its row from the True Depth file, or re-evaluate the scan."
                )
            return lines[i:], None
    return [], (
        f"No 'position (mm)' column in the first lines of\n{path}\n\nITRAXpy reads the column names "
        "from the line after the two title lines that Q-Spec writes."
    )


def _start_values(scan: Scan) -> tuple[float, float, bool]:
    """(StartID, StartPosition, whether StartPosition came from document.txt) of a scan, or an ITRAXpyError."""
    cell = scan.td_row["StartID"]
    start_id = parse_number(cell)
    if start_id is None:
        raise ITRAXpyError(
            f"StartID for '{scan.core}' in the True Depth file is "
            + (f"not a number ({str(cell).strip()!r})." if has_value(cell) else "empty.")
        )
    cell = scan.td_row["StartPosition"]
    start_position = parse_number(cell)
    if start_position is None and has_value(cell):
        raise ITRAXpyError(
            f"StartPosition for '{scan.core}' in the True Depth file is not a number "
            f"({str(cell).strip()!r}). Enter the scanner position in mm, or leave the cell empty "
            "to use the start coordinate from document.txt."
        )
    from_document = start_position is None
    if from_document:
        start_position = scan.document.get("start")
        if start_position is None:
            raise ITRAXpyError(
                f"StartPosition for '{scan.core}' is empty in the True Depth file and there is no "
                f"start coordinate in document.txt for\n  {scan.display}"
            )
    return start_id, cast(float, start_position), from_document


def check_scans(scans: list[Scan]) -> None:
    """Find the input problems that would stop a run, for all scans at once, before the previous results are
    deleted: StartID and StartPosition, and results files that cannot be read or hold no measurements."""
    problems = []
    for scan in scans:
        try:
            _start_values(scan)
        except ITRAXpyError as e:
            problems.append(str(e))
        problem = _results_lines(scan.results_file)[1]
        if problem:
            problems.append(problem)
    if problems:
        raise ITRAXpyError(
            "Please fix these first (nothing was changed):\n\n"
            + "\n\n".join(problems[:MAX_LISTED])
            + (f"\n\n... and {len(problems) - MAX_LISTED} more" if len(problems) > MAX_LISTED else "")
        )


def _numeric_columns(raw: pd.DataFrame, columns: list[str], path: str) -> None:
    """The columns as numbers: comma decimals are read too; any other text stops the run with a message."""
    bad = []
    for c in columns:
        if raw[c].dtype == object or pd.api.types.is_string_dtype(raw[c]):
            text = raw[c].astype("string").str.strip()
            values = pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce")
            wrong = values.isna() & text.notna() & text.ne("")
            if wrong.any():
                bad.append(
                    f"{c} (e.g. '{text[wrong].iloc[0]}' in measurement {int(np.flatnonzero(wrong)[0]) + 1})"
                )
            raw[c] = values.astype(float)
    if bad:
        raise ITRAXpyError(f"Text instead of numbers in\n{path}\n\n{describe_items(bad)}")


def read_scan(scan: Scan) -> tuple[pd.DataFrame, list[str], bool, bool]:
    """Read one results file and add the settings and depth columns.

    Returns (table, element column names, whether StartPosition came from document.txt,
    whether most element values are missing).
    """
    lines, problem = _results_lines(scan.results_file)
    if problem:
        raise ITRAXpyError(problem)
    try:
        raw = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", index_col=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise ITRAXpyError(f"Could not read\n{scan.results_file}\n\n{e}") from e
    # the trailing tab on every line creates an empty 'Unnamed' column
    raw = raw.loc[:, [not str(c).startswith("Unnamed") for c in raw.columns]]
    raw.columns = [str(c).strip() for c in raw.columns]

    position_column = next(c for c in POSITION_COLUMNS if c in raw.columns)
    anchor = next((c for c in ("MSE", "cps", "kcps") if c in raw.columns), None)
    if anchor is None:
        raise ITRAXpyError(
            f"Could not find where the element columns start (no MSE/cps column) in\n{scan.results_file}"
        )
    exclude = set(COUNT_COLUMNS) | set(DROP_COLUMNS)
    elements = [c for c in raw.columns[cast(int, raw.columns.get_loc(anchor)) + 1:] if c not in exclude]
    _numeric_columns(
        raw, [position_column] + [c for c in COUNT_COLUMNS if c in raw.columns] + elements, scan.results_file
    )
    if raw[position_column].isna().all():
        raise ITRAXpyError(f"No position could be read in\n{scan.results_file}")

    start_id, start_position, from_document = _start_values(scan)
    position = raw[position_column]

    first = raw.columns[0]  # 'filename'
    table = pd.DataFrame(
        {
            first: raw[first],
            "Voltage": _plain(scan.settings.get("voltage")),
            "Amperage": _plain(scan.settings.get("current")),
            "position (mm)": position,
            "CoreID": scan.td_row["CoreID"],
            "SectionID": scan.td_row["SectionID"],
            # rounded as in slct, norm and clr (without, the csv tables show 0.020000000000010232)
            "DepthID": (position - start_position + start_id).round(2),
            "SampleID": (position - start_position).round(2),
            "RepID": scan.rep_id,
        }
    )
    for c in COUNT_COLUMNS:
        if c in raw.columns:
            table[c] = raw[c]
    # a results file that was never re-evaluated stops after the first few values of every line, and those
    # values are not the counts of the elements named above them: no element values at all for such a scan
    mostly_empty = bool(elements) and raw[elements].isna().to_numpy().mean() > 0.5
    values = pd.DataFrame(np.nan, index=raw.index, columns=elements) if mostly_empty else raw[elements]
    table = pd.concat([table, values], axis=1)
    table[GROUP_COLUMN] = core_group(scan)
    table[SCAN_COLUMN] = scan.label
    return table, elements, from_document, mostly_empty


def _plain(value: object) -> int | float | None:
    if value is None:
        return None
    value = float(str(value))
    return int(value) if value.is_integer() else value


def read_scans(scans: list[Scan], notes: list[str]) -> list[ScanResult]:
    """Read and validate measurements without writing any output files."""
    results: list[ScanResult] = []
    start_from_document: list[str] = []
    empty: list[str] = []
    for scan in scans:
        table, elements, from_document, mostly_empty = read_scan(scan)
        results.append((scan, table, elements))
        if from_document:
            start_from_document.append(f"{scan.display} ({format_number(scan.document['start'])} mm)")
        if mostly_empty:
            empty.append(f"{scan.display} ({os.path.basename(scan.results_file)})")
    if start_from_document:
        notes.append(
            "StartPosition was empty in the True Depth file, so the start coordinate from document.txt was used for:\n"
            + describe_items(start_from_document)
        )
    if empty:
        notes.append(
            "These scans contain almost no element values and probably need Q-spec re-evaluation. "
            "Their element values remain empty:\n" + describe_items(empty)
        )
    raw_files = [
        s.display
        for s in scans
        if os.path.basename(s.results_file).lower() == "result.txt"
           and f"{s.display} ({os.path.basename(s.results_file)})" not in empty
    ]
    if raw_files and len(raw_files) < len(scans):
        notes.append(
            "Some scans have only the instrument's result.txt; others use re-evaluated Results.txt "
            "(element columns may differ):\n" + describe_items(raw_files)
        )
    return results


def id_text(value: object) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)
