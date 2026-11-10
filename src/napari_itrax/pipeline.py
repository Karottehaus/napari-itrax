from __future__ import annotations
from collections.abc import Callable, Iterable, Iterator, Sequence
from numpy.typing import NDArray
from openpyxl.worksheet.worksheet import Worksheet
from .types import Document, Settings, Photo, ScanPhoto, ArgonCurve
import csv
import io
import logging
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, cast

import numpy as np
import pandas as pd
from natsort import natsorted, ns
from openpyxl.styles import PatternFill
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("itraxpy")

# ITRAX line-scan images easily exceed PIL's decompression-bomb limit; they are local instrument files.
Image.MAX_IMAGE_PIXELS = None

# columns of the ITRAX results file that are not carried into the outputs
DROP_COLUMNS = ["sample surface", "E-gain", "E-offset", "F-slope", "F-offset"]
# meta columns that sit between the fit parameters and the element columns
COUNT_COLUMNS = ["validity", "cps", "kcps", "Dt", "MSE"]
POSITION_COLUMNS = ["position (mm)", "X Position (mm)"]
TD_COLUMNS = ["Core", "CoreID", "SectionID", "StartID", "StartPosition"]
TUBES = ["Cr", "Mo", "Rh", "Ag"]
# Only element symbols (Fe, Ca, ...) can be selected for norm and clr, as in practice in the original ITRAXpy,
# where every other column name contains a space and crashed the export. Not selectable: the tube's scattered
# radiation ('Rh inc', 'Rh coh'; added to every output unchanged) and the other fitted lines ('Fe a*2', 'Fe a+b',
# 'Fe X1', 'W La', 'W La inel', ...; they stay in the per-scan tables).
ELEMENT_SYMBOLS = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr "
    "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split()
)
SCATTER_RE = re.compile(r"[A-Z][a-z]? (inc|coh)")
TUBE_NAMES = {
    "Cr": "Chromium (Cr)",
    "Mo": "Molybdenum (Mo)",
    "Rh": "Rhodium (Rh)",
    "Ag": "Silver (Ag)",
    "Cu": "Copper (Cu)",
    "W": "Tungsten (W)",
}
IMAGE_EXTENSIONS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp")
GROUP_COLUMN = "_core_group"  # internal: which core a row belongs to (for the zero replacement)
SCAN_COLUMN = "_scan"  # internal: which scan a row comes from (for the per-scan norm/clr files)
INTERNAL_COLUMNS = [GROUP_COLUMN, SCAN_COLUMN]
# rows with validity = 0 are kept but shaded light grey in the xlsx files
INVALID_FILL = PatternFill(fill_type="solid", start_color="D9D9D9", end_color="D9D9D9")
# rows marked 'suspect' (Ar clearly above the core's median) are shaded light orange
SUSPECT_FILL = PatternFill(fill_type="solid", start_color="FFE0B2", end_color="FFE0B2")
DEFAULT_SUSPECT_FACTOR = 1.5
# what happens to measurements with validity = 0 in slct, norm, clr, the ratios and the plots (the per-scan
# XRF-results tables always keep them): kept with empty values (the default), kept with their values, left out
INVALID_EMPTY, INVALID_KEEP, INVALID_LEAVE_OUT = "empty", "keep", "leave out"
INVALID_POINTS = {
    INVALID_EMPTY: "rows kept, values empty",
    INVALID_KEEP: "rows kept with values",
    INVALID_LEAVE_OUT: "rows left out",
}

SECTION_RE = re.compile(r"(?<![a-z])section[\s_\-]*(\d+)(?!\d)", re.IGNORECASE)
# section names of the form <core>_<from>-<to>[_anything], e.g. 'AP2_104-204', 'AP2_14.03-14.9', 'AP1_11-12_takeII'
SECTION_NAME_RE = re.compile(r"(.+?)[_ ]\d+(?:[.,]\d+)?-\d+(?:[.,]\d+)?(?:[_ ].*)?", re.IGNORECASE)
REPLICATE_RE = re.compile(r"(?:\.|[_\- ]rep\s*)(\d+)$", re.IGNORECASE)
MAX_LISTED = 12


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
    section: Optional[int]  # section number, if the core has several sections
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


def _norm(text: object) -> str:
    return str(text).strip().lower()


def _has_value(value: object) -> bool:
    return (
            value is not None and not (isinstance(value, float) and np.isnan(value)) and str(value).strip() != ""
    )


def _to_number(text: object) -> float | None:
    try:
        value = float(str(text).strip().replace(",", "."))
    except ValueError:
        return None
    return None if np.isnan(value) else value


def format_number(value: object) -> str:
    value = float(str(value))
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _listing(items: Iterable[object]) -> str:
    items = list(items)
    shown = "\n".join("  " + str(i) for i in items[:MAX_LISTED])
    return shown + (f"\n  ... and {len(items) - MAX_LISTED} more" if len(items) > MAX_LISTED else "")


# --- Finding the input files ---------------------------------------------------------------


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
            parts = [_to_number(v) for v in lines[i + 1][:4]]
            if len(parts) >= 3 and None not in parts[:3]:
                doc["date"] = (
                    f"{int(cast(float, parts[0])):04d}-{int(cast(float, parts[1])):02d}-{int(cast(float, parts[2])):02d}"
                )
                if len(parts) == 4 and parts[3] is not None:
                    doc["hour"] = int(parts[3])
        elif key == "xrf voltage and current":
            numbers = [n for n in map(_to_number, values) if n is not None]
            if len(numbers) >= 2:
                doc["voltage"], doc["current"] = numbers[:2]
        elif key == "xrf exp. time" and values:
            doc["time"] = _to_number(values[0])
        elif key == "tube" and values and values[0]:
            doc["tube"] = values[0]
        elif key == "start coordinate" and values:
            doc["start"] = _to_number(values[0])
            if len(values) >= 3:
                doc["stop"] = _to_number(values[2])
        elif key == "optical start" and len(values) >= 3:
            # scanner coordinates (mm) of the two ends of optical.tif
            doc["optical_start"], doc["optical_end"] = _to_number(values[0]), _to_number(values[2])
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
    cannot = [getattr(e, "filename", None) or str(e) for e in unreadable]
    if cannot and notes is not None:
        notes.append(
            f"{len(cannot)} folder(s) could not be read and were not searched (a path longer than 260 "
            f"characters, or no permission):\n{_listing(cannot)}"
        )
    if not found:
        raise ITRAXpyError(
            "No ITRAX results files (Results.txt / result.txt) were found in\n"
            f"{input_folder}\nor any of its subfolders."
            + (
                f"\n\n{len(cannot)} folder(s) could not be read (path longer than 260 characters?):\n"
                f"{_listing(cannot)}"
                if cannot
                else ""
            )
        )
    return found


def first_document(input_folder: str) -> Document:
    """document.txt values of the first scan below input_folder (to pre-fill the GUI)."""
    for root, dirs, files in _walk(input_folder):
        if "document.txt" in files:
            return read_document(root)
    return {}


# --- True Depth file -----------------------------------------------------------------------


def load_true_depth(path: str, notes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (table with standard column names, table as read for the output workbook)."""
    if os.path.splitext(str(path))[1].lower() not in (".xlsx", ".xlsm"):
        raise ITRAXpyError(
            f"The True Depth file must be an Excel workbook (.xlsx):\n{path}\n\nOpen it in Excel and "
            "save it as 'Excel Workbook (*.xlsx)', or use Create... to make one."
        )
    try:
        raw = pd.read_excel(path)
    except Exception as e:
        raise ITRAXpyError(f"Could not read the True Depth file:\n{path}\n\n{e}")
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
            td[name] = td[name].map(lambda v: _id_text(v).strip() if _has_value(v) else None)
    td = td[td["Core"].map(_has_value)].reset_index(drop=True)
    if td.empty:
        raise ITRAXpyError("The True Depth file contains no rows with a 'Core' name.")
    return td, raw


def true_depth_template(found: list[ScanFolder]) -> pd.DataFrame:
    """A True Depth table with one row per scan folder, for the user to complete.

    Core is the section name from document.txt (else the folder name), made unique; Folder pins each row
    to its scan folder, so the folders never have to be renamed or rearranged. StartPosition is the
    document.txt start coordinate; StartID (the true depth of that point) is left for the user.
    """
    rows: list[dict[str, object]] = []
    used: Counter[str] = Counter()
    core_ids: dict[str, int] = {}
    sections: Counter[str] = Counter()
    folders = {f.folder.lower() for f in found}
    for f in found:
        m = REPLICATE_RE.search(f.folder)
        if m and f.folder[: m.start()].lower() in folders:
            continue  # a replicate of the scan before it: handled with 'XRF replicates'
        name = f.document.get("section") or f.names[-1]
        used[name.lower()] += 1
        if used[name.lower()] > 1:
            name = f"{name}_{used[name.lower()]}"
        # 'AP2_104-204' or 'AP1_11-12_takeII': section of core 'AP2' / 'AP1' -> same CoreID, next SectionID
        m = SECTION_NAME_RE.fullmatch(name)
        core = m.group(1).lower() if m else name.lower()
        core_ids.setdefault(core, len(core_ids) + 1)
        sections[core] += 1
        rows.append(
            {
                "Core": name,
                "CoreID": core_ids[core],
                "SectionID": sections[core],
                "StartID": None,
                "StartPosition": f.document.get("start"),
                "Folder": "/".join(f.names[1:]) or f.names[0],
            }
        )
    return pd.DataFrame(rows, columns=TD_COLUMNS + ["Folder"])


def _section_number(scan_folder: ScanFolder, core_depth: int | None) -> int | None:
    """Section number from a 'Section N' folder below the core folder, or from document.txt."""
    candidates = scan_folder.names[core_depth + 1:] if core_depth is not None else []
    candidates = candidates + [scan_folder.document.get("section", "")]
    for name in candidates:
        m = SECTION_RE.search(name)
        if m:
            return int(m.group(1))
    return None


def _find_parts(names: list[str], parts: list[str]) -> int | None:
    """Index of the last folder of `parts` inside the folder list `names`, or None."""
    n = len(parts)
    for end in range(len(names) - 1, n - 2, -1):
        if names[end - n + 1: end + 1] == parts:
            return end
    return None


def _describe_row(td: pd.DataFrame, i: int) -> str:
    section = td.at[i, "SectionID"]
    return f"'{str(td.at[i, 'Core'])}'" + (
        f" (SectionID {format_number(section)})" if _to_number(section) is not None else ""
    )


def assign_scans(
        found: list[ScanFolder], td: pd.DataFrame, include_replicates: bool, notes: list[str]
) -> list[Scan]:
    """Match scan folders to True Depth rows and return the scans to process, in True Depth order.

    Names are compared exactly first. Only True Depth rows that match no folder at all are then
    compared ignoring upper/lower case, so a core 'A' does not grab every scan folder called 'a'.
    """
    exact_keys = td["Core"].map(lambda name: str(name).strip())
    lower_keys = exact_keys.str.lower()
    section_ids = pd.to_numeric(td["SectionID"], errors="coerce")
    folder_parts = {}
    input_folder = found[0].paths[0]
    if "Folder" in td.columns:
        for i in td.index:
            value = td.at[i, "Folder"]
            if _has_value(value):
                value = str(value).strip()
                if os.path.isabs(value):  # a path copied from the Explorer
                    relative = os.path.relpath(os.path.normpath(value), os.path.dirname(input_folder))
                    value = relative if not relative.startswith("..") else value
                folder_parts[i] = [p for p in re.split(r"[\\/]+", _norm(value)) if p]

    rows_exact: dict[str, list[int]] = {}
    rows_lower: dict[str, list[int]] = {}
    for i in td.index:
        if i not in folder_parts:
            rows_exact.setdefault(exact_keys[i], []).append(i)
            rows_lower.setdefault(lower_keys[i], []).append(i)
    for rows in rows_lower.values():
        ids = section_ids[rows]
        if ids.duplicated(keep=False).any() or (len(rows) > 1 and ids.isna().any()):
            raise ITRAXpyError(
                f"The True Depth file has {len(rows)} rows for '{str(td.at[rows[0], 'Core'])}' without distinct "
                "SectionIDs.\nEach core section needs its own row and SectionID (or use a 'Folder' column)."
            )

    lowered_names = {f.folder: [_norm(n) for n in f.names] for f in found}

    def pinned_end(names: list[str], parts: list[str]) -> tuple[int | None, int | None]:
        """Where a Folder value matches a scan folder's names: the folder itself, its replicate folder
        ('<folder>.1', '<folder>_rep1', ranked just below it) or a scan folder inside it. Returns (end, rank)."""
        end = _find_parts(names, parts)
        if end is not None:
            return end, 0 if end == len(names) - 1 else -2
        m = REPLICATE_RE.search(names[-1])
        if m and _find_parts(names[:-1] + [names[-1][: m.start()]], parts) == len(names) - 1:
            return len(names) - 1, -1
        return None, None

    exact_folder = {
        i
        for i, parts in folder_parts.items()
        if any(pinned_end(names, parts)[1] == 0 for names in lowered_names.values())
    }

    def candidates_for(
            f: ScanFolder, case_sensitive: bool, allowed: set[int] | None = None
    ) -> tuple[dict[int, tuple[int, int | None, int | None]], bool]:
        """True Depth rows this scan folder could belong to: {row: (rank, core depth, section number)}."""
        prepare = (lambda name: str(name).strip()) if case_sensitive else _norm
        lookup = rows_exact if case_sensitive else rows_lower
        names = [prepare(n) for n in f.names]
        candidates: dict[int, tuple[int, int | None, int | None]] = {}
        if case_sensitive:
            lowered = lowered_names[f.folder]
            for i, parts in folder_parts.items():
                end, kind = pinned_end(lowered, parts)
                if end is None or (i in exact_folder and kind == -2):
                    continue
                if kind == -1 and not include_replicates:
                    continue
                candidates[i] = (10_000 + end, end, _section_number(f, end) if kind != -1 else None)

        name_depth = {}
        for depth, name in enumerate(names):
            for i in lookup.get(name, []):
                name_depth[i] = depth  # deepest folder with this name wins
        section_name = f.document.get("section")
        doc_rows = set(lookup.get(prepare(section_name), [])) if section_name else set()
        matched_a_name = False
        for i in set(name_depth) | doc_rows:
            if allowed is not None and i not in allowed:
                continue
            matched_a_name = True
            core_depth = name_depth.get(i)
            number = _section_number(f, core_depth)
            if number is not None and not np.isnan(section_ids[i]):
                if number != section_ids[i]:
                    continue
            elif len(rows_lower[lower_keys[i]]) > 1:
                continue  # several sections, but this scan's is unknown
            rank = (
                len(names) if i in doc_rows else name_depth[i]
            )  # the scanner's own section name is most specific
            candidates[i] = (rank, core_depth, number)
        return candidates, matched_a_name

    matches: dict[
        int, list[tuple[ScanFolder, int | None, int | None]]
    ] = {}  # row -> list of (scan folder, section number, depth of the core folder)
    section_unknown = set()  # scan folders whose core name matched but whose section did not

    def assign(f: ScanFolder, candidates: dict[int, tuple[int, int | None, int | None]]) -> None:
        best = max(rank for rank, _, _ in candidates.values())
        rows = [i for i, c in candidates.items() if c[0] == best]
        if len(rows) > 1:
            raise ITRAXpyError(
                f"The scan folder\n  {f.display}\nmatches several True Depth rows:\n"
                f"{_listing(_describe_row(td, i) for i in rows)}\n\n"
                + (
                    "Give these rows different 'Folder' values (the folder of each scan)."
                    if folder_parts
                    else "Add a 'Folder' column to the True Depth file to say which row belongs to which folder."
                )
            )
        _, depth, number = candidates[rows[0]]
        matches.setdefault(rows[0], []).append((f, number, depth))

    leftovers = []
    for f in found:
        candidates, matched_a_name = candidates_for(f, case_sensitive=True)
        if candidates:
            assign(f, candidates)
        else:
            leftovers.append(f)
            if matched_a_name:
                section_unknown.add(f.folder)
    allowed = {i for i in td.index if i not in matches and i not in folder_parts}
    unassigned = []
    for f in leftovers:
        candidates, matched_a_name = candidates_for(f, False, allowed) if allowed else ({}, False)
        if candidates:
            assign(f, candidates)
        else:
            unassigned.append(f)
            if matched_a_name:
                section_unknown.add(f.folder)
    no_section = [f.display for f in unassigned if f.folder in section_unknown]
    ignored = [f.display for f in unassigned if f.folder not in section_unknown]

    if not matches and no_section:
        raise ITRAXpyError(
            f"The scan folders belong to cores with several rows in the True Depth file, but their section could "
            f"not be matched to a SectionID. Use 'Section N' folder names, or a 'Folder' column with the folder of "
            f"each row:\n{_listing(no_section)}"
        )
    if not matches:
        if not no_section and not folder_parts and len(found) == len(td):
            notes.append(
                "The 'Core' names in the True Depth file match none of the folder names, so rows were "
                "assigned in folder order (1st row = 1st scan folder). Please check the depths."
            )
            matches = {i: [(f, None, None)] for i, f in zip(td.index, found)}
            ignored = []
        else:
            raise ITRAXpyError(
                f"None of the {len(found)} scan folders matches a "
                + ("'Folder' value or " if folder_parts else "")
                + "'Core' name in the True Depth file.\n\n"
                  f"Scan folders found:\n{_listing(f.display for f in found)}\n\n"
                  f"Core names in the True Depth file:\n{_listing(td['Core'])}"
            )

    scans = []
    for i in td.index:
        if i in matches:
            scans.extend(
                _scans_for_row(
                    td, i, matches[i], include_replicates, int((lower_keys == lower_keys[i]).sum()) > 1, notes
                )
            )
    # the per-scan files and plots are named after the label: rows that still share one are numbered
    rows_of: dict[str, list[int]] = {}
    for scan in scans:
        rows_of.setdefault(scan.label.lower(), []).append(cast(int, scan.td_row.name))
    shared = {label: list(dict.fromkeys(rows)) for label, rows in rows_of.items() if len(set(rows)) > 1}
    if shared:
        renamed = set()
        for scan in scans:
            shared_rows = shared.get(scan.label.lower())
            if shared_rows:
                scan.label = f"{scan.label}_{shared_rows.index(cast(int, scan.td_row.name)) + 1}"
                renamed.add(scan.label)
        notes.append(
            "Several True Depth rows have the same Core name (and SectionID), so their files are "
            f"numbered _1, _2, ...: {', '.join(sorted(renamed))}"
        )
    missing = [_describe_row(td, i) for i in td.index if i not in matches]

    if missing:
        notes.append(
            f"No scan folder was found for {len(missing)} row(s) of the True Depth file:\n{_listing(missing)}"
        )
    if no_section:
        notes.append(
            f"{len(no_section)} scan folder(s) belong to a core with several rows in the True Depth file, "
            "but their section could not be matched to a SectionID (use 'Section N' folder names or a "
            f"'Folder' column). They were not processed:\n{_listing(no_section)}"
        )
    if ignored:
        notes.append(
            f"{len(ignored)} scan folder(s) are not listed in the True Depth file and were not "
            f"processed:\n{_listing(ignored)}"
        )
    return scans


def _scans_for_row(
        td: pd.DataFrame,
        i: int,
        entries: list[tuple[ScanFolder, int | None, int | None]],
        include_replicates: bool,
        several_sections: bool,
        notes: list[str],
) -> list[Scan]:
    core = str(td.at[i, "Core"]).strip()
    names = {os.path.basename(f.folder).lower() for f, _, _ in entries}

    mains: list[tuple[ScanFolder, int | None, int | None]] = []
    replicates: list[tuple[tuple[ScanFolder, int | None, int | None], int | None]] = []
    for entry in entries:
        name = os.path.basename(entry[0].folder).lower()
        m = REPLICATE_RE.search(name)
        if m and name[: m.start()] in names:
            replicates.append((entry, int(m.group(1))))
        else:
            mains.append(entry)
    if not mains:
        mains, replicates = [replicates[0][0]], replicates[1:]
    if len(mains) > 1:
        if len({os.path.dirname(f.folder) for f, _, _ in mains}) > 1:
            raise ITRAXpyError(
                f"{len(mains)} scan folders match the True Depth row {_describe_row(td, i)}:\n"
                f"{_listing(f.display for f, _, _ in mains)}\n\n"
                "Remove the extra copies from the input folder, or add a 'Folder' column to the True Depth "
                "file with the folder to use."
            )
        # several scans side by side in one core folder: like the original ITRAXpy, the first is the
        # main scan and the others are replicates
        replicates = [(entry, None) for entry in mains[1:]] + replicates
        mains = mains[:1]

    rows = [(mains[0], 0)]
    if replicates and not include_replicates:
        notes.append(
            f"'{core}' has {len(replicates)} replicate scan(s); tick 'XRF Replicates' to include them."
        )
    elif replicates:
        used = {n for _, n in replicates if n is not None}
        if len(used) < len([n for _, n in replicates if n is not None]):
            notes.append(f"Replicate folders of '{core}' have duplicate replicate numbers.")
        next_id = 1
        for entry, number in replicates:
            if number is None:
                while next_id in used:
                    next_id += 1
                number = next_id
                used.add(number)
            rows.append((entry, number))

    scans = []
    safe_core = re.sub(r'[\\/:*?"<>|]', "_", core)
    row_section = _to_number(td.at[i, "SectionID"])
    for (f, section_number, depth), rep_id in rows:
        section = section_number
        if section is None and several_sections and row_section is not None:
            section = int(row_section)
        label = safe_core + (f"_Section{section}" if section is not None else "")
        core_dir = f.paths[depth] if depth is not None else f.folder
        scans.append(
            Scan(
                core,
                cast(pd.Series, td.loc[i]),
                section,
                rep_id,
                f.folder,
                f.results_file,
                label,
                core_dir,
                f.document,
                f.display,
            )
        )
    return scans


# --- Settings ------------------------------------------------------------------------------


def is_element(name: str) -> bool:
    """'Fe' yes; 'Fe a*2', 'W La', 'Rh inc', 'Dt' no."""
    return str(name).strip() in ELEMENT_SYMBOLS


def is_scatter(name: str) -> bool:
    """'Rh inc', 'Rh coh': the tube's incoherent / coherent scatter peak."""
    return SCATTER_RE.fullmatch(str(name).strip()) is not None


def tube_symbol(text: str) -> str | None:
    """'Rhodium (Rh)' -> 'Rh', 'Rh' -> 'Rh'."""
    text = (text or "").strip()
    m = re.search(r"\(([A-Za-z]{1,2})\)", text)
    if m:
        return m.group(1)
    return text if re.fullmatch(r"[A-Za-z]{1,2}", text) else None


def tube_label(symbol: str) -> str:
    return TUBE_NAMES.get(symbol, symbol)


def resolve_settings(scans: list[Scan], entered: Settings, notes: list[str]) -> tuple[str, str]:
    """Voltage, current, measuring time and tube per scan: document.txt first, the GUI entries otherwise.

    entered: dict with voltage, current, time, tube (None where the field is empty).
    Returns (settings text for the output files, most common tube).
    """
    quantities = (
        ("tube", "Tube", ""),
        ("time", "Time", " s"),
        ("voltage", "Voltage", " kV"),
        ("current", "Current", " mA"),
    )
    differs: dict[tuple[str, str, str], list[str]] = {}
    for scan in scans:
        for key, label, unit in quantities:
            recorded = cast(dict[str, str | float | None], scan.document).get(key)
            typed = cast(dict[str, str | float | None], entered).get(key)
            if recorded is not None:
                if typed is not None and (
                        _norm(recorded) != _norm(typed)
                        if key == "tube"
                        else abs(float(recorded) - float(typed)) > 1e-9
                ):
                    differs.setdefault((label, unit, str(typed)), []).append(scan.display)
                cast(dict[str, str | float | None], scan.settings)[key] = recorded
            elif typed is not None:
                cast(dict[str, str | float | None], scan.settings)[key] = typed
            else:
                raise ITRAXpyError(
                    f"Please enter the {label.lower()} in box 4 (XRF parameters): there is no "
                    f"document.txt with this value for\n  {scan.display}"
                )

    for (label, unit, typed), where in differs.items():
        shown = typed if unit == "" else format_number(typed)
        notes.append(
            f"{label} in box 4 ({shown}{unit}) differs from document.txt for {len(where)} scan(s); "
            f"the document.txt values were used:\n{_listing(where)}"
        )

    combos = Counter(
        (
            tube_label(s.settings["tube"]),
            format_number(s.settings["time"]),
            format_number(s.settings["voltage"]),
            format_number(s.settings["current"]),
        )
        for s in scans
    )
    if len(combos) == 1:
        tube, time, voltage, current = next(iter(combos))
        text = f"Settings: {tube} as input source, {time} s, {voltage} kV, {current} mA."
    else:
        ordered = combos.most_common()
        parts = [
            f"{tube}, {time} s, {voltage} kV, {current} mA ({n} scan{'s' if n != 1 else ''})"
            for (tube, time, voltage, current), n in ordered
        ]
        text = "Settings: mixed - " + "; ".join(parts) + "."
        others = [
            f"{s.display}: {tube_label(s.settings['tube'])}, {format_number(s.settings['time'])} s, "
            f"{format_number(s.settings['voltage'])} kV, {format_number(s.settings['current'])} mA"
            for s in scans
            if (
                   tube_label(s.settings["tube"]),
                   format_number(s.settings["time"]),
                   format_number(s.settings["voltage"]),
                   format_number(s.settings["current"]),
               )
               != ordered[0][0]
        ]
        notes.append(
            f"The scans were not all measured with the same settings. Most used {parts[0]}; these differ:\n"
            f"{_listing(others)}\nThey are combined in one file; check that this is what you want."
        )
    main_tube = Counter(s.settings["tube"] for s in scans).most_common(1)[0][0]
    return text, main_tube


# --- Reading and combining scans -----------------------------------------------------------


def _group_of(scan: Scan) -> str:
    core_id = scan.td_row.get("CoreID")
    return _id_text(core_id) if _has_value(core_id) else scan.core


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
    start_id = _to_number(cell)
    if start_id is None:
        raise ITRAXpyError(
            f"StartID for '{scan.core}' in the True Depth file is "
            + (f"not a number ({str(cell).strip()!r})." if _has_value(cell) else "empty.")
        )
    cell = scan.td_row["StartPosition"]
    start_position = _to_number(cell)
    if start_position is None and _has_value(cell):
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
        raise ITRAXpyError(f"Text instead of numbers in\n{path}\n\n{_listing(bad)}")


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
    except Exception as e:
        raise ITRAXpyError(f"Could not read\n{scan.results_file}\n\n{e}")
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
    table[GROUP_COLUMN] = _group_of(scan)
    table[SCAN_COLUMN] = scan.label
    return table, elements, from_document, mostly_empty


def _plain(value: object) -> int | float | None:
    if value is None:
        return None
    value = float(str(value))
    return int(value) if value.is_integer() else value


def read_scans(scans: list[Scan], notes: list[str], suspect_factor: float | None = None) -> list[ScanResult]:
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
    if suspect_factor:
        flag_suspect(results, suspect_factor, notes)
    check_depths(results, notes)
    if start_from_document:
        notes.append(
            "StartPosition was empty in the True Depth file, so the start coordinate from document.txt was used for:\n"
            + _listing(start_from_document)
        )
    if empty:
        notes.append(
            "These scans contain almost no element values and probably need Q-spec re-evaluation. "
            "Their element values remain empty:\n" + _listing(empty)
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
            "(element columns may differ):\n" + _listing(raw_files)
        )
    return results


def save_scan_images(
        results: list[ScanResult],
        folders: OutputFolders,
        high_res: bool,
        notes: list[str],
        progress: Callable[[float], None] | None = None,
        ruler: bool = True,
        argon: bool = False,
) -> None:
    """Write photos for an already validated set of measurements."""
    missing_images: list[str] = []
    for n, (scan, table, _) in enumerate(results, start=1):
        if scan.rep_id == 0:
            curve = None
            if argon and "Ar" in table.columns:
                curve = tuple(
                    table[c] if c in table.columns else None
                    for c in ("position (mm)", "Ar", "validity", "suspect")
                )
            elif argon:
                missing_images.append(f"Ar image (no 'Ar' column in the results file): {scan.display}")
            missing_images += save_images(scan, folders, high_res, ruler, curve)
        if progress:
            progress(n / len(results))
    if missing_images:
        notes.append(
            "Some images were not found or could not be read, and were skipped:\n" + _listing(missing_images)
        )


def process_scans(
        scans: list[Scan],
        folders: OutputFolders,
        high_res: bool,
        notes: list[str],
        progress: Callable[[float], None] | None = None,
        ruler: bool = True,
        argon: bool = False,
        suspect_factor: float | None = None,
) -> list[ScanResult]:
    """Read measurements and save photos; convenient for non-interactive callers."""
    results = read_scans(scans, notes, suspect_factor)
    save_scan_images(results, folders, high_res, notes, progress, ruler, argon)
    return results


def flag_suspect(results: list[ScanResult], factor: float, notes: list[str]) -> None:
    """Add a 'suspect' column (1/0): Ar above factor x the median Ar of the same core.

    Ar rises where the beam hits air (gaps, cracks, a poor core surface). The median is taken over the points
    with validity != 0. Nothing is removed; the rows are shaded in the outputs.
    """
    tables = [(scan, table) for scan, table, _ in results if "Ar" in table.columns]
    missing = [scan.display for scan, table, _ in results if "Ar" not in table.columns]
    if missing:
        notes.append(f"No 'Ar' column, so no suspect points could be marked for:\n{_listing(missing)}")
    if not tables:
        return
    parts = []
    for _, t in tables:
        valid = (
            pd.to_numeric(t["validity"], errors="coerce").fillna(1).ne(0) if "validity" in t.columns else True
        )
        parts.append(
            pd.DataFrame(
                {"Ar": pd.to_numeric(t["Ar"], errors="coerce"), "group": t[GROUP_COLUMN], "valid": valid}
            )
        )
    ar_all = pd.concat(parts, ignore_index=True)
    medians = ar_all[ar_all["valid"]].groupby("group")["Ar"].median()
    flagged = total = 0
    for scan, table in tables:
        median = medians.get(_group_of(scan), np.nan)
        suspect = (pd.to_numeric(table["Ar"], errors="coerce") > factor * median).astype(int)
        anchor = "validity" if "validity" in table.columns else "RepID"
        table.insert(cast(int, table.columns.get_loc(anchor)) + 1, "suspect", suspect)
        flagged, total = flagged + int(suspect.sum()), total + len(suspect)
    notes.append(
        f"{flagged} of {total} measurements have Ar above {format_number(factor)} x the median Ar of the "
        "valid points of their "
        "core and are marked as suspect (column 'suspect', shaded light orange; nothing was removed)."
    )


def check_depths(results: list[ScanResult], notes: list[str]) -> None:
    """Warn about StartIDs that look like cm or m instead of mm, and about sections that overlap in depth."""
    sections: dict[str, list[tuple[float, float, float, str]]] = {}
    for scan, table, _ in results:
        if scan.rep_id != 0 or table["DepthID"].isna().all():
            continue
        start_id = cast(float, _to_number(scan.td_row["StartID"]))
        sections.setdefault(table[GROUP_COLUMN].iloc[0], []).append(
            (start_id, table["DepthID"].min(), table["DepthID"].max(), scan.label)
        )
    for core, rows in sections.items():
        if len(rows) < 2:
            continue
        rows.sort()
        # consecutive sections: StartID difference / scanned length in mm is about 1 when StartID is in mm
        ratios = [
            (b[0] - a[0]) / (a[2] - a[1]) for a, b in zip(rows, rows[1:]) if a[2] > a[1] and b[0] > a[0]
        ]
        # most pairs decide, so one odd StartID (e.g. a section named in metres) does not hide the pattern
        unit = None
        if ratios and sum(0.05 <= r <= 0.2 for r in ratios) >= max(1, 2 * len(ratios) / 3):
            unit = "cm"
        elif ratios and sum(0.0005 <= r <= 0.002 for r in ratios) >= max(1, 2 * len(ratios) / 3):
            unit = "m"
        if unit:
            notes.append(
                f"Core {core}: the StartIDs of its sections look like {unit}, but StartID must be in mm "
                "(DepthID = position - StartPosition + StartID, and the image rulers use it). "
                "Please check the True Depth file."
            )
            continue
        by_depth = sorted(rows, key=lambda r: r[1])
        overlaps = [
            f"{a[3]} and {b[3]}: {format_number(round((a[2] - b[1]) / 10, 1))} cm"
            for a, b in zip(by_depth, by_depth[1:])
            if a[2] - b[1] > 5
        ]
        if overlaps:
            notes.append(
                f"Core {core}: these sections overlap in true depth (DepthID), so the same depths appear "
                f"twice. Please check StartID / StartPosition:\n{_listing(overlaps)}"
            )


def combine(
        results: list[ScanResult], invalid_points: str, notes: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Stack main scans, then replicates, into one table.

    results: list of (scan, table, elements). invalid_points: what happens to the measurements with validity = 0
    (INVALID_POINTS). Returns (combined, element columns, meta columns).
    """
    ordered = [r for r in results if r[0].rep_id == 0] + [r for r in results if r[0].rep_id != 0]
    combined = pd.concat([table for _, table, _ in ordered], ignore_index=True, sort=False)
    elements = list(dict.fromkeys(e for _, _, els in ordered for e in els))

    if "validity" in combined.columns:
        invalid = pd.to_numeric(combined["validity"], errors="coerce").fillna(1).eq(0)
        count = f"{int(invalid.sum())} of {len(combined)} measurements have validity = 0"
        if invalid.all() and invalid_points != INVALID_KEEP:
            raise ITRAXpyError(
                f"No valid measurements left to process: all {len(combined)} have validity = 0."
            )
        if invalid.any() and invalid_points == INVALID_LEAVE_OUT:
            notes.append(f"{count} and were left out of slct, norm, clr, the ratios and the plots.")
            combined = combined[~invalid].reset_index(drop=True)
        elif invalid.any() and invalid_points == INVALID_EMPTY:
            notes.append(
                f"{count}. Their rows and depths are kept, with empty values in slct, norm, clr, the ratios "
                "and the plots (light grey rows in the xlsx files; the counts are in the XRF-results "
                "tables)."
            )
            # the counts of every element, scatter peak and fitted line; depth, validity and suspect stay
            for c in elements:
                combined[c] = combined[c].where(~invalid)
        elif invalid.any():
            notes.append(
                f"{count}. They are kept with their values, and shown in light grey in the xlsx files "
                "(see the 'validity' column)."
            )
    if combined.empty:
        raise ITRAXpyError("No valid measurements left to process.")

    meta = [c for c in combined.columns if c not in elements and c not in INTERNAL_COLUMNS]
    return combined, elements, meta


# --- Statistics for the element selection --------------------------------------------------


def zeroes_percent(df: pd.DataFrame) -> list[float]:
    """Share of the measurements with 0 counts, in % (of the measurements with a value)."""
    return (df.eq(0).sum() / df.notna().sum().clip(lower=1) * 100).round().tolist()


def cv_percent(df: pd.DataFrame) -> list[float]:
    """Coefficient of variation in %, capped at 100 (the original 'std' column of the zero report)."""
    mean = df.mean()
    cv = (df.std() / mean * 100).where(mean > 0)
    return cv.clip(upper=100).fillna(100).round().tolist()


def all_zero(df: pd.DataFrame) -> list[bool]:
    return df.fillna(0).eq(0).all().tolist()


# Lines of an element that lie within about 0.15 keV of a line of a neighbour that is usually far bigger in
# sediments, e.g. Sb La 3.60 keV under K Kb 3.59 / Ca Ka 3.69, V Ka 4.95 under Ti Kb 4.93, Y Ka under Rb Kb.
# The spectrum fit cannot fully separate them, so the small element can follow its neighbour.
NEIGHBOURS = {
    "P": ("Zr",),
    "Pd": ("Ar",),
    "Ag": ("Ar",),
    "Cd": ("Ar",),
    "Sn": ("K",),
    "Sb": ("Ca", "K"),
    "Te": ("Ca",),
    "I": ("Ca",),
    "Sc": ("Ca",),
    "Ba": ("Ti",),
    "La": ("Ti",),
    "Ce": ("Ti",),
    "Pr": ("Ti",),
    "V": ("Ti",),
    "Eu": ("Mn",),
    "Tb": ("Fe",),
    "Dy": ("Fe", "Mn"),
    "Co": ("Fe",),
    "Er": ("Fe",),
    "Tm": ("Fe",),
    "Yb": ("Ni",),
    "Hf": ("Cu",),
    "Ta": ("Cu",),
    "As": ("Pb",),
    "Pb": ("As",),
    "Y": ("Rb",),
    "Zr": ("Sr",),
}
TUBE_NEIGHBOURS = {"Cr": ("Mn",)}  # Mn Ka 5.90 keV next to the Kb 5.95 keV of a Cr tube
BIGGER = 3  # a neighbour counts if its median counts are at least this many times higher


def main_rows(combined: pd.DataFrame, elements: list[str] | None = None) -> pd.DataFrame:
    """Valid measurements of the main scans (no replicates). With elements, also without the rows that have none of
    their values (results files that were never re-evaluated: read_scan leaves them empty). A scan evaluated with
    fewer elements than another one keeps its rows."""
    rows = combined[combined["RepID"].eq(0)] if "RepID" in combined.columns else combined
    if "validity" in rows.columns:
        rows = rows[pd.to_numeric(rows["validity"], errors="coerce").fillna(1).ne(0)]
    if elements:
        rows = rows[rows[elements].notna().any(axis=1)]
    return rows


def _noise_share(noise: pd.Series, total: pd.Series) -> list[float | None]:
    return [None if pd.isna(v) else round(min(v, 100)) for v in noise / total * 100]


def noise_percent(combined: pd.DataFrame, elements: list[str]) -> list[float | None]:
    """Share of each element's variation along the core that is noise, in % (up to 100).

    A measured element changes gradually along the core, so neighbouring points are alike; noise is not. Half the
    squared difference between neighbouring points estimates the noise variance (von Neumann), compared with the
    variance along the scan, both summed over the main scans. Layers thinner than the step count as noise too.
    """
    noise, total = pd.Series(0.0, index=elements), pd.Series(0.0, index=elements)
    for _, scan in main_rows(combined).groupby(SCAN_COLUMN, sort=False):
        values = scan[elements].astype(float)
        noise += (values.diff() ** 2).sum() / 2
        total += ((values - values.mean()) ** 2).sum()
    return _noise_share(noise, total.where(total > 0))


def repeat_percent(combined: pd.DataFrame, elements: list[str]) -> list[float | None] | None:
    """The same share from replicate scans: half the squared difference between a replicate and its main scan at
    the same depth, compared with the variance along that main scan. None without replicates; an element
    without pairs gets None."""
    if "RepID" not in combined.columns or not combined["RepID"].ne(0).any():
        return None
    rows = combined.dropna(subset=["DepthID"])
    if "validity" in rows.columns:
        rows = rows[pd.to_numeric(rows["validity"], errors="coerce").fillna(1).ne(0)]
    noise, pairs = pd.Series(0.0, index=elements), pd.Series(0.0, index=elements)
    total, points = pd.Series(0.0, index=elements), pd.Series(0.0, index=elements)
    for _, scan in rows.groupby(SCAN_COLUMN, sort=False):
        main = scan[scan["RepID"].eq(0)].sort_values("DepthID")
        replicates = scan[scan["RepID"].ne(0)]
        step = main["DepthID"].diff().median()
        if replicates.empty or len(main) < 2 or not step > 0:
            continue
        values = main[elements].astype(float)
        total += ((values - values.mean()) ** 2).sum()
        points += (values.notna().sum() - 1).clip(lower=0)
        for _, replicate in replicates.groupby("RepID"):
            matched = pd.merge_asof(
                replicate.sort_values("DepthID")[["DepthID"] + elements].astype(float),
                main[["DepthID"] + elements].astype(float),
                on="DepthID",
                direction="nearest",
                tolerance=step / 2,  # type: ignore[arg-type]  # pandas accepts float tolerance for float merge keys
                suffixes=("", " main"),
            )
            difference = matched[elements] - matched[[f"{e} main" for e in elements]].to_numpy()
            noise += (difference ** 2).sum() / 2
            pairs += difference.notna().sum()
    return _noise_share(noise / pairs.where(pairs > 0), (total / points).where(total > 0))


def _log_correlation(rows: pd.DataFrame, a: str, b: str) -> float:
    """Correlation of ln a and ln b along the cores (each scan centred, so level shifts between scans don't count)."""
    pair = rows.loc[(rows[[a, b]] > 0).all(axis=1), [a, b]].astype(float)
    if len(pair) < 3:
        return float("nan")
    logs = cast(pd.DataFrame, np.log(pair))
    logs = logs - logs.groupby(rows.loc[pair.index, SCAN_COLUMN]).transform("mean")
    return logs[a].corr(logs[b])


def overlaps(combined: pd.DataFrame, elements: list[str], tube: str) -> list[tuple[str, float | None] | None]:
    """Per element: (neighbour, r) when its line lies next to a bigger peak of the data, r being their log
    correlation (near 1: the element mostly measures the neighbour); (tube, None) for the tube's own line and for
    lines next to it, ('air', None) for Ar (from the air between detector and core); otherwise None."""
    rows = main_rows(combined)
    medians = rows[elements].median()
    result: list[tuple[str, float | None] | None] = []
    for element in elements:
        if element == tube or element in TUBE_NEIGHBOURS.get(tube, ()):
            result.append((f"{tube} tube", None))
            continue
        if element == "Ar":
            result.append(("air", None))
            continue
        found = [
            (n, _log_correlation(rows, element, n))
            for n in NEIGHBOURS.get(element, ())
            if n in medians.index and medians[n] >= BIGGER * max(medians[element], 1)
        ]
        found = [(n, r) for n, r in found if pd.notna(r)]
        result.append(max(found, key=lambda f: f[1]) if found else None)
    return result


def _replace_zeroes(df: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.notna(values[col]):
            out[col] = out[col].mask(out[col] == 0, values[col])
    return out


def zeroes_replacement(
        df: pd.DataFrame, groups: pd.Series | None = None, notes: list[str] | None = None
) -> pd.DataFrame:
    """Replace zeroes with half of the smallest positive value of the same column.

    With groups (one label per row, e.g. the CoreID) this is done per core, so a core's results do not
    depend on which other cores are processed with it. A core without any positive value in a column
    falls back to the smallest positive value of all cores.
    """
    df = df.astype(float)
    overall = df[df > 0].min() / 2
    if groups is None:
        return _replace_zeroes(df, overall)

    out = df.copy()
    fallback: dict[str, list[str]] = {}
    for core, index in df.groupby(groups, sort=False).groups.items():
        part = df.loc[index]
        half_minimum = part[part > 0].min() / 2
        for col in half_minimum.index[half_minimum.isna() & part.eq(0).any()]:
            fallback.setdefault(col, []).append(str(core))
        out.loc[index] = _replace_zeroes(part, half_minimum.fillna(overall))
    if notes is not None:
        for col, cores in fallback.items():
            notes.append(
                f"'{col}' has no counts above zero in core(s) {', '.join(cores)}; zeroes there were "
                "replaced using the smallest value of all cores."
            )
    return out


def _scatter(df: pd.DataFrame, tube: str) -> tuple[list[str], pd.Series | None, pd.Series | None]:
    """The scatter peaks of each row's own tube: (tubes with inc and coh columns, main tube first; inc; coh).

    A results file has the columns of its own tube only, so in a run of scans measured with different tubes a row
    takes the first tube whose inc or coh it has.
    """
    pairs = sorted(
        (t for t in TUBES if f"{t} inc" in df.columns and f"{t} coh" in df.columns), key=lambda t: t != tube
    )
    inc = coh = None
    for t in pairs:
        i, c = (pd.to_numeric(df[f"{t} {kind}"], errors="coerce") for kind in ("inc", "coh"))
        if inc is None or coh is None:
            inc, coh = i, c
        else:
            use = inc.isna() & coh.isna() & (i.notna() | c.notna())
            inc, coh = inc.where(~use, i), coh.where(~use, c)
    return pairs, inc, coh


def inc_coh_columns(df: pd.DataFrame, tube: str, add_ratio: bool, notes: list[str]) -> pd.DataFrame:
    """The tube's incoherent/coherent scatter columns (of every tube, if the scans were measured with several),
    optionally with their ratio."""
    pairs, inc, coh = _scatter(df, tube)
    if not pairs:
        if add_ratio:
            notes.append(
                "No scatter columns (e.g. 'Rh inc' / 'Rh coh') were found, so no inc/coh ratio was added."
            )
        return pd.DataFrame(index=df.index)
    if tube and tube not in pairs:
        notes.append(
            f"No '{tube} inc' / '{tube} coh' columns found; the {pairs[0]} scatter columns were used instead."
        )
    if len(pairs) > 1:
        notes.append(
            f"The scans were measured with different tubes ({', '.join(pairs)}): the inc and coh columns of "
            "each tube are kept, and inc/coh and element/(inc+coh) use each scan's own tube."
        )
    columns = df[[f"{t} {kind}" for t in pairs for kind in ("inc", "coh")]].copy()
    if add_ratio:
        assert inc is not None and coh is not None
        columns["inc/coh"] = inc / coh.replace(0, np.nan)
    return columns


# --- Writing -------------------------------------------------------------------------------


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
            )
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


def _id_text(value: object) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _unique_path(folder: str, name: str) -> str:
    stem, ext = os.path.splitext(name)
    path, n = os.path.join(folder, name), 2
    while os.path.exists(path):
        path, n = os.path.join(folder, f"{stem}_{n}{ext}"), n + 1
    return path


def _largest_image(folders: list[str]) -> str | None:
    for folder in dict.fromkeys(folders):
        images = [e for e in os.scandir(folder) if e.is_file() and e.name.lower().endswith(IMAGE_EXTENSIONS)]
        if images:
            return max(images, key=lambda e: e.stat().st_size).path
    return None


def _optical_image(folder: str) -> str | None:
    """optical.tif, or the multi-sensor core logger's optical_<name>.tif."""
    path = os.path.join(folder, "optical.tif")
    if os.path.isfile(path):
        return path
    for name in natsorted(os.listdir(folder), alg=ns.IGNORECASE):
        lower = name.lower()
        if lower.startswith("optical") and lower.endswith((".tif", ".tiff")) and "lowres" not in lower:
            return os.path.join(folder, name)
    return None


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def photo_scale(scan: Scan, width: int) -> tuple[float, float] | None:
    """(true depth in mm at the left edge of the 180-degree rotated optical image, mm per pixel), or None.

    The image spans 'Optical Start' to 'Optical End' of document.txt (scanner mm); depth = DepthID of the XRF data.
    """
    o_start, o_end = scan.document.get("optical_start"), scan.document.get("optical_end")
    start_id = cast(float, _to_number(scan.td_row["StartID"]))
    start_position = _to_number(scan.td_row["StartPosition"])
    if start_position is None:
        start_position = scan.document.get("start")
    if o_start is None or o_end is None or start_id is None or start_position is None or o_end <= o_start:
        return None
    return o_start - start_position + start_id, (o_end - o_start) / width


def settings_text(settings: Settings) -> str:
    """'Rhodium (Rh), 20 s, 30 kV, 50 mA' for one scan's settings (see resolve_settings); '' when not known."""
    if not all(key in settings for key in ("tube", "time", "voltage", "current")):
        return ""
    return (
        f"{tube_label(settings['tube'])}, {format_number(settings['time'])} s, "
        f"{format_number(settings['voltage'])} kV, {format_number(settings['current'])} mA"
    )


def scan_photos(scans: list[Scan]) -> list[ScanPhoto]:
    """The main scans for the profile plots: list of (label, core group, settings text, photo), photo being
    (image path, depth of its top, depth of its bottom) in mm, or None."""
    return [(s.label, _group_of(s), settings_text(s.settings), s.photo) for s in scans if s.rep_id == 0]


def add_depth_ruler(image: Image.Image, scan: Scan, argon: ArgonCurve | None = None) -> Image.Image | None:
    """Copy of the (180 degree rotated) optical image with a true-depth ruler below it (mm ticks, cm numbers).

    The scale comes from document.txt: the image spans 'Optical Start' to 'Optical End' (scanner mm). After
    the rotation, Optical Start is on the left (checked against the XRF counts of 29 real scans). The depth
    is the DepthID of the XRF data: position - StartPosition + StartID, in mm. Returns None without a scale.
    argon: optional (positions in scanner mm, Ar counts[, validity]), drawn as a light blue curve over the photo;
    measurements with validity = 0 are shaded red.
    """
    width, height = image.size
    scale = photo_scale(scan, width)
    if scale is None:
        return None
    depth_left, mm_per_px = scale
    o_start = scan.document["optical_start"]

    def x_of(depth_mm: float) -> float:
        return (depth_mm - depth_left) / mm_per_px

    strip = 100
    out = Image.new("RGB", (width, height + strip), "white")
    out.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(out)
    label_font, title_font = _font(20), _font(16)
    top = height
    draw.line([(0, top), (width, top)], fill="black", width=2)
    first_cm = int(np.ceil(max(depth_left, 0) / 10))
    last_cm = int(np.floor((depth_left + width * mm_per_px) / 10))
    first_mm = int(np.ceil(max(depth_left, 0)))
    last_mm = int(np.floor(depth_left + width * mm_per_px))
    for mm in range(first_mm, last_mm + 1):
        if mm % 10:  # the cm ticks follow below
            x = round(x_of(mm))
            draw.line([(x, top), (x, top + (10 if mm % 5 == 0 else 6))], fill="black", width=1)
    for cm in range(first_cm, last_cm + 1):
        x = round(x_of(cm * 10))
        length = 40 if cm % 10 == 0 else 28 if cm % 5 == 0 else 15
        draw.line(
            [(x, top), (x, top + length)], fill="black", width=3 if cm % 10 == 0 else 2 if cm % 5 == 0 else 1
        )
        if cm % 5 == 0:
            draw.text((x, top + length + 2), str(cm), fill="black", font=label_font, anchor="mt")
    draw.text(
        (6, top + strip - 6), f"true depth (cm)  -  {scan.label}", fill="black", font=title_font, anchor="ls"
    )
    if argon is not None:
        _draw_argon(out, argon, o_start, mm_per_px, height, top + strip - 6, title_font)
    return out


ARGON_COLOUR = "#7FD4FF"
INVALID_SHADE = (230, 40, 40, 70)  # translucent red band over invalid measurements
INVALID_DOT = "#D62828"
SUSPECT_SHADE = (255, 150, 0, 70)
SUSPECT_DOT = "#F08C00"


def _draw_argon(
        out: Image.Image,
        argon: ArgonCurve,
        o_start: float,
        mm_per_px: float,
        height: int,
        text_y: int,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Ar counts as a light blue curve over the photo (higher counts higher up), with its range in the strip.

    argon: (positions, Ar counts[, validity[, suspect]]); missing ones may be None. Measurements with
    validity = 0 get a translucent red band over the photo (their stretch of core, half way to the neighbouring
    measurements) and a red dot on the curve; suspect ones (not invalid) the same in orange.
    """
    assert argon[0] is not None
    point_count = len(argon[0])

    def numbers(values: pd.Series | None, default: float) -> NDArray[np.float64]:
        if values is None:
            return np.full(point_count, default, dtype=np.float64)
        return np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)

    argon = tuple(argon) + (None,) * (4 - len(argon))
    positions, counts = numbers(argon[0], np.nan), numbers(argon[1], np.nan)
    validity, suspect = numbers(argon[2], 1), numbers(argon[3], 0)
    ok = np.isfinite(positions) & np.isfinite(counts)
    positions, counts, validity, suspect = positions[ok], counts[ok], validity[ok], suspect[ok]
    if len(counts) == 0:
        return
    order = np.argsort(positions)
    positions, counts, validity, suspect = positions[order], counts[order], validity[order], suspect[order]
    invalid = validity == 0
    suspect = (suspect == 1) & ~invalid
    width = out.size[0]

    def x_of(position_mm: float) -> float:
        return float((position_mm - o_start) / mm_per_px)

    if invalid.any() or suspect.any():
        # each measurement covers the core half way to its neighbours; the end points mirror their neighbour
        middles = (positions[1:] + positions[:-1]) / 2
        step = np.diff(positions).mean() if len(positions) > 1 else 1.0
        edges = np.concatenate([[positions[0] - step / 2], middles, [positions[-1] + step / 2]])
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        shade = ImageDraw.Draw(overlay)
        for flags, colour in ((invalid, INVALID_SHADE), (suspect, SUSPECT_SHADE)):
            for i in np.flatnonzero(flags).tolist():
                shade.rectangle([x_of(edges[i]), 0, x_of(edges[i + 1]), height], fill=colour)
        photo = out.crop((0, 0, width, height)).convert("RGBA")
        out.paste(Image.alpha_composite(photo, overlay).convert("RGB"), (0, 0))

    draw = ImageDraw.Draw(out)
    low, high = float(np.min(counts)), float(np.max(counts))
    top, bottom = 0.08 * height, 0.92 * height
    scaled = (counts - low) / (high - low) if high > low else np.full(len(counts), 0.5)
    points = [(x_of(p), float(bottom - s * (bottom - top))) for p, s in zip(positions, scaled)]
    if len(points) > 1:
        draw.line(points, fill="#0B2E4F", width=5, joint="curve")  # dark outline, visible on light tape
        draw.line(points, fill=ARGON_COLOUR, width=3, joint="curve")
    for (x, y), bad, odd in zip(points, invalid, suspect):
        if bad or odd:
            draw.ellipse(
                [x - 5, y - 5, x + 5, y + 5], fill=INVALID_DOT if bad else SUSPECT_DOT, outline="white"
            )

    # legend in the ruler strip, right-aligned: [blue line] Ar counts ...  [box] validity = 0  [box] suspect
    right = float(width - 6)
    for flags, text, line_colour, box in (
            (suspect, "suspect", SUSPECT_DOT, (255, 205, 140)),
            (invalid, "validity = 0", INVALID_DOT, (240, 150, 150)),
    ):
        if flags.any():
            text = f"{text} ({int(flags.sum())})"
            draw.text((right, text_y), text, fill=line_colour, font=font, anchor="rs")
            right -= draw.textlength(text, font=font) + 8
            draw.rectangle([right - 20, text_y - 13, right, text_y - 1], fill=box, outline=line_colour)
            right -= 40
    text = f"Ar counts: {format_number(low)} (bottom) - {format_number(high)} (top)"
    draw.text((right, text_y), text, fill="#1F6FB2", font=font, anchor="rs")
    text_left = right - draw.textlength(text, font=font)
    draw.line([(text_left - 40, text_y - 6), (text_left - 10, text_y - 6)], fill=ARGON_COLOUR, width=4)


def save_images(
        scan: Scan, folders: OutputFolders, high_res: bool, ruler: bool = True, argon: ArgonCurve | None = None
) -> list[str]:
    """Save the rotated optical (and optionally high-res) image. Returns descriptions of missing images."""
    # <label>_Core<CoreID>_Section<SectionID>, as the original names them; an empty cell is left out
    base = scan.label + "".join(
        f"_{part}{_id_text(scan.td_row[column])}"
        for part, column in (("Core", "CoreID"), ("Section", "SectionID"))
        if _has_value(scan.td_row[column])
    )
    missing = []

    optical = _optical_image(scan.folder)
    rotated = None
    if optical:
        try:
            with Image.open(optical) as image:
                rotated = image.rotate(180, expand=True)
        except (OSError, SyntaxError, ValueError):
            missing.append(f"optical image (could not be read, damaged file?): {scan.display}")
    if rotated is not None:
        lr_path = _unique_path(folders.images_lr, base + "_LR.tif")
        rotated.save(lr_path)
        scale = photo_scale(scan, rotated.size[0])
        if scale:
            scan.photo = (lr_path, scale[0], scale[0] + rotated.size[0] * scale[1])
        ruler_image = add_depth_ruler(rotated, scan) if ruler else None
        if ruler and ruler_image is None:
            missing.append(f"ruler (no 'Optical Start/End' in document.txt or no StartID): {scan.display}")
        elif ruler_image is not None:
            ruler_image.save(_unique_path(folders.images_ruler, base + "_LR_ruler.tif"))
        if argon is not None:
            argon_image = add_depth_ruler(rotated, scan, argon)
            if argon_image is None:
                missing.append(
                    f"Ar image (no 'Optical Start/End' in document.txt or no StartID): {scan.display}"
                )
            else:
                argon_image.save(_unique_path(folders.images_ar, base + "_LR_Ar.tif"))
    elif not optical:
        missing.append(f"optical image: {scan.display}")

    if high_res:
        search = [scan.folder] if scan.section is not None else [scan.core_dir, scan.folder]
        source = _largest_image(search)
        if source:
            try:
                with Image.open(source) as image:
                    image.rotate(90, expand=True).save(_unique_path(folders.images_hr, base + "_HR.tif"))
            except (OSError, SyntaxError, ValueError):
                missing.append(f"high-res image (could not be read, damaged file?): {scan.display}")
        else:
            missing.append(f"high-res image: {scan.display}")
    return missing


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


DEFAULT_LOG_RATIOS = "Ca/Ti, Si/Ti, Zr/Rb, Fe/Mn"
DEFAULT_RATIOS = "Ca/Ti, Zr/Rb"


def parse_ratio_pairs(text: str) -> list[tuple[str, str]]:
    """The ratio fields: 'Ca/Ti, zr/rb' -> [('Ca', 'Ti'), ('Zr', 'Rb')]."""
    pairs = []
    for part in re.split(r"[,;]+", text or ""):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"([A-Za-z]{1,2})\s*/\s*([A-Za-z]{1,2})", part)
        if not m:
            raise ITRAXpyError(f"The ratio '{part}' is not of the form Ca/Ti (element/element).")
        a, b = m.groups()
        pairs.append((a.capitalize(), b.capitalize()))
    return pairs


def ratio_tables(
        combined: pd.DataFrame,
        selected: list[str],
        tube: str,
        log_ratios: Sequence[tuple[str, str]],
        notes: list[str],
        ratios: Sequence[tuple[str, str]] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ratios often used for ITRAX data. Returns three tables (columns only, same rows as combined):

    - element / (inc + coh): counts divided by the tube's scatter, which follows matrix and density;
    - element / kcps: counts divided by the total count rate (kcps, or cps / 1000);
    - coh/inc (organic matter / water proxy), the simple ratios a/b of the raw counts (empty where b is 0), and
      the log-ratios ln(a/b) with zeroes replaced per core as for clr.
    """
    counts = combined[selected].astype(float)
    pairs, inc, coh = _scatter(combined, tube)
    per_scatter = pd.DataFrame(index=combined.index)
    other = pd.DataFrame(index=combined.index)
    if pairs:
        assert inc is not None and coh is not None
        per_scatter = counts.div((inc + coh).replace(0, np.nan), axis=0)
        per_scatter.columns = [f"{e}/(inc+coh)" for e in selected]
        other["coh/inc"] = coh / inc.replace(0, np.nan)
    else:
        notes.append("No scatter columns (e.g. Rh inc / Rh coh), so no element/(inc+coh) and coh/inc ratios.")

    # kcps, else cps / 1000 - per row, as scans evaluated with different software may have either column
    kcps = None
    if "kcps" in combined.columns:
        kcps = pd.to_numeric(combined["kcps"], errors="coerce")
    if "cps" in combined.columns:
        from_cps = pd.to_numeric(combined["cps"], errors="coerce") / 1000
        kcps = from_cps if kcps is None else kcps.fillna(from_cps)
    if kcps is None:
        notes.append("No cps / kcps column, so no element/kcps ratios.")
    per_kcps = pd.DataFrame(index=combined.index)
    if kcps is not None:
        per_kcps = counts.div(kcps.replace(0, np.nan), axis=0)
        per_kcps.columns = [f"{e}/kcps" for e in selected]

    missing = sorted(
        {
            e
            for pair in list(ratios) + list(log_ratios)
            for e in pair
            if e not in combined.columns or not is_element(e)
        }
    )
    if missing:
        notes.append(f"Ratios with {', '.join(missing)} were left out: not in this data.")
    for a, b in ratios:
        if a not in missing and b not in missing:
            numerator, denominator = (pd.to_numeric(combined[e], errors="coerce") for e in (a, b))
            other[f"{a}/{b}"] = numerator / denominator.replace(0, np.nan)
    usable = [(a, b) for a, b in log_ratios if a not in missing and b not in missing]
    if usable:
        needed = list(dict.fromkeys(e for pair in usable for e in pair))
        groups = combined[GROUP_COLUMN] if GROUP_COLUMN in combined.columns else None
        filled = zeroes_replacement(combined[needed], groups)
        # a zero is left only where an element has no count above zero in any measurement: no log-ratio there
        never = [e for e in needed if not filled[e].gt(0).any()]
        if never:
            notes.append(
                f"{', '.join(never)}: no counts above zero in any measurement, so the log-ratios with "
                f"{'it' if len(never) == 1 else 'them'} are empty."
            )
        filled = filled.where(filled.gt(0))
        for a, b in usable:
            other[f"ln({a}/{b})"] = np.log(filled[a] / filled[b])
    return per_scatter, per_kcps, other


def write_selected_outputs(
        combined: pd.DataFrame,
        meta_columns: list[str],
        selected: list[str],
        tube: str,
        add_inc_coh: bool,
        core_label: str,
        settings: str,
        td_raw: pd.DataFrame,
        output_folder: str,
        file_format: str,
        log_ratios: Sequence[tuple[str, str]] = (),
        photos: list[ScanPhoto] | None = None,
        plots: bool = True,
        ratios: Sequence[tuple[str, str]] = (),
        detail_pages: bool = False,
        core_workbooks: bool = False,
) -> list[str]:
    """Write the selected (slct), normalised (norm) and centred log-ratio (clr) tables, the ratios and the plots.

    core_workbooks: the per-core files as one workbook per scan with a tab per treatment (xlsx only), instead of a
    folder per treatment. Returns a list of notes for the user.
    """
    notes = []
    scatter_selected = [c for c in selected if is_scatter(c)]
    lines_selected = [c for c in selected if not is_element(c) and not is_scatter(c)]
    if scatter_selected:
        notes.append(
            f"{', '.join(scatter_selected)}: scattered tube radiation, not elements. Left out of norm and "
            "clr; the columns are added to the files unchanged."
        )
    if lines_selected:
        notes.append(
            f"{', '.join(lines_selected)}: not element symbols (sum peaks, other fitted lines). Left out "
            "of slct, norm and clr; they stay in the per-scan tables."
        )
    selected = [c for c in selected if is_element(c)]
    if not selected:
        raise ITRAXpyError(
            "Please select at least one element. Scatter peaks (e.g. Rh inc / Rh coh) and other "
            "fitted lines (e.g. Fe a*2) are not part of norm and clr."
        )
    meta = combined[meta_columns].copy()
    for c in ("DepthID", "SampleID"):
        if c in meta.columns:
            meta[c] = meta[c].round(2)

    counts = combined[selected]
    scatter = inc_coh_columns(combined, tube, add_inc_coh, notes)
    scatter = scatter[[c for c in scatter.columns if c not in selected]]

    # (invalid points kept with empty values do not count here)
    valid = (
        pd.to_numeric(combined["validity"], errors="coerce").fillna(1).ne(0)
        if "validity" in combined.columns
        else pd.Series(True, index=combined.index)
    )
    incomplete = [c for c in selected if counts.loc[valid, c].isna().any()]
    if incomplete:
        notes.append(
            "Some selected columns are missing from part of the results files, so those rows have empty "
            f"norm and clr values: {', '.join(incomplete)}"
        )

    groups = combined[GROUP_COLUMN] if GROUP_COLUMN in combined.columns else None
    filled = zeroes_replacement(counts, groups, notes)
    # a row without a value for one of the selected elements gets no norm (like clr): a percentage of fewer
    # elements would not be comparable with the other rows
    normalised = filled.div(filled.sum(axis=1, skipna=False), axis=0) * 100
    logs = cast(pd.DataFrame, np.log(filled))
    clr = logs.sub(logs.mean(axis=1, skipna=False), axis=0)

    slct_table = pd.concat([meta, counts, scatter], axis=1)
    norm_table = pd.concat([meta, normalised, scatter], axis=1)
    clr_table = pd.concat([meta, clr, scatter], axis=1)

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
            td_raw.to_excel(writer, sheet_name="True-Depth", index=False)
    else:
        for suffix, table in (("slct", slct_table), ("norm", norm_table), ("clr", clr_table)):
            _write_csv(table, os.path.join(output_folder, f"{core_label}_{suffix}.csv"), core_label, settings)
        td_raw.to_csv(os.path.join(output_folder, f"{core_label}_True-Depth.csv"), index=False)

    # ratios: element/(inc+coh), element/kcps, coh/inc and log-ratios, with the same meta columns
    per_scatter, per_kcps, other = ratio_tables(combined, selected, tube, log_ratios, notes, ratios)
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

    if plots:
        from . import plots as profile_plots  # matplotlib is only loaded when plots are made

        profile_plots.write_profiles(
            os.path.join(output_folder, f"{core_label}_plots"),
            combined,
            core_label,
            settings,
            {"counts": counts, "ratios": other},
            photos or [],
            detail_pages,
            notes,
        )
    return notes
