from __future__ import annotations

import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass

from ..models import RunInfo

NAME = "_README - what is where.pdf"


@dataclass
class GuideContent:
    """A snapshot of the guide's text and tables, independent of page layout."""

    title: str
    footer: str
    facts: list[tuple[str, str]]
    summary: str
    structure: list[tuple[str, str, int]]
    file_uses: list[tuple[str, str, int]]
    highlights_title: str
    swatches: list[tuple[str, str]]
    highlights_note: str
    columns_title: str
    columns_intro: str
    headers: list[tuple[str, str, int]]
    formulas_intro: str
    formulas: list[tuple[str, str, str | None]]


def _and(items: Iterable[str]) -> str:
    items = list(items)
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else "".join(items)


def _settings(settings: str | None) -> str:
    """The settings line of the run in words: 'Settings: Rhodium (Rh) as input source, 20 s, 30 kV, 50 mA.' -> 'a Rh
    tube at 30 kV and 50 mA, 20 s per measurement'; mixed settings: the most common ones, then the others."""

    def tube(text: str) -> str:
        symbol = re.search(r"\((\w+)\)", text)
        return symbol.group(1) if symbol else text.strip()

    m = re.fullmatch(r"Settings: (.+?) as input source, (\S+) s, (\S+) kV, (\S+) mA\.", settings or "")
    if m:
        name, seconds, kv, ma = m.groups()
        return f"a {tube(name)} tube at {kv} kV and {ma} mA, {seconds} s per measurement"
    parts = re.findall(
        r"([^;]+?), (\S+) s, (\S+) kV, (\S+) mA \((\d+) scans?\)",
        (settings or "").replace("Settings: mixed - ", ""),
    )
    if parts:
        (name, seconds, kv, ma, n), others = parts[0], parts[1:]
        text = f"a {tube(name)} tube at {kv} kV and {ma} mA, {seconds} s per measurement ({n} scan{'s' if n != '1' else ''}"
        for o_name, o_seconds, o_kv, o_ma, o_n in others:
            with_tube = "" if tube(o_name) == tube(name) else f"with a {tube(o_name)} tube "
            text += (
                f"; {o_n} scan{'s' if o_n != '1' else ''} {with_tube}at {o_kv} kV and {o_ma} mA, "
                f"{o_seconds} s"
            )
        return text + ")"
    return settings.replace("Settings: ", "settings: ").rstrip(".") if settings else ""


def measured(scans: list[tuple[str, int | None]]) -> str:
    """When the scans were measured, from their document.txt: [(date, hour)] -> '2026-09-07, 11 h',
    '2026-09-07, 11-16 h' or '2026-09-09 to 2026-09-11'."""
    dates = sorted({date for date, _ in scans})
    if not dates:
        return "not recorded"
    if len(dates) > 1:
        return f"{dates[0]} to {dates[-1]}"
    hours = sorted({hour for _, hour in scans if hour is not None})
    return dates[0] + (
        "" if not hours else f", {hours[0]} h" if len(hours) == 1 else f", {hours[0]}–{hours[-1]} h"
    )


def report(label: str, info: RunInfo) -> str:
    """The first paragraph, for the researcher the cores were measured for: what was measured and which elements
    were selected for further analyses."""

    def count(n: int, word: str) -> str:
        return f"{n:,}".replace(",", " ") + f" {word}{'s' if n != 1 else ''}"

    cores, scans, n = info.get("cores") or 0, info.get("scans") or 0, info.get("measurements")
    text = (
        f"This folder contains the results of the ITRAX XRF core scanning of {label}: {count(cores, 'core')}"
    )
    if scans != cores:
        text += f" in {count(scans, 'scan')}"
    if n:
        text += f", {count(n, 'measurement')}"
        if info.get("left_out"):
            text += f" ({count(info['left_out'], 'invalid measurement')} left out of slct, norm, clr and the ratios)"
    settings = _settings(info.get("settings"))
    text += f", measured with {settings}. " if settings else ". "
    selected = info.get("selected")
    if not selected:
        return text + "The elements for further analyses have not been selected yet."
    text += (
        f"For further analyses, {len(selected)} elements were selected: {_and(selected)}. The normalised "
        f"values (norm) and the centred log-ratios (clr) are calculated from these elements"
    )
    ratios = [f"{a}/{b}" for a, b in info.get("ratios") or []]
    logs = [f"ln({a}/{b})" for a, b in info.get("log_ratios") or []]
    if ratios or logs:
        text += "; added were " + " and ".join(
            part
            for part in (
                f"the ratio{'s' if len(ratios) > 1 else ''} {_and(ratios)}" if ratios else "",
                f"the log-ratio{'s' if len(logs) > 1 else ''} {_and(logs)}" if logs else "",
            )
            if part
        )
    return text + "."


HEADER_TEXTS = {
    "filename": "The spectrum file of the measurement; its folder names the scan.",
    "Voltage": "Tube voltage of the scan (kV).",
    "Amperage": "Tube current of the scan (mA).",
    "position (mm)": "Position of the measurement on the scanner (mm).",
    "CoreID": "The core, from the True Depth file.",
    "SectionID": "The section of the core, from the True Depth file.",
    "DepthID": "True depth of the measurement (mm).",
    "SampleID": "position - StartPosition (mm): from the scanner position where StartID applies (the start of "
                "the scan if StartPosition was empty).",
    "RepID": "0 = main scan; 1, 2 ... = replicate scans.",
    "validity": "1 = good; 0 = the scanner could not fit the spectrum well (grey row).",
    "suspect": "1 = Ar unusually high, often an air gap or crack (orange row).",
    "cps": "Count rate of the measurement (counts per second).",
    "kcps": "Count rate of the measurement (thousand counts per second).",
    "Dt": "Dead time of the detector, as the scanner wrote it.",
    "MSE": "Error of the spectrum fit; high values = a poor fit.",
    "inc/coh": "Incoherent / coherent scatter of the tube.",
}


def _headers(root: str, label: str, info: RunInfo) -> list[str]:
    """The column headers of this run's combined slct table, in order (row 3 of the xlsx, line 3 of the csv); []
    before Write element files."""
    path = os.path.join(root, f"{label}_slct.{'csv' if info.get('file_format') == 'csv' else 'xlsx'}")
    if not os.path.isfile(path):
        return []
    if path.endswith(".csv"):
        import csv

        with open(path, encoding="utf-8", newline="") as f:
            lines = list(csv.reader(f))
        return lines[2] if len(lines) > 2 else []
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True)
    rows = list(book.worksheets[0].iter_rows(min_row=3, max_row=3, values_only=True))
    book.close()
    return [str(h) for h in rows[0] if h is not None] if rows else []


def columns(root: str, label: str, info: RunInfo) -> list[tuple[str, str, int]]:
    """'Column headers' rows: the headers of this run's tables as written, in order, with what each holds."""
    selected = info.get("selected") or []
    headers = _headers(root, label, info)
    factor = info.get("suspect_factor")
    rows, elements_done = [], False
    for h in headers:
        if h in selected:
            if not elements_done:
                rows.append(
                    (
                        ", ".join(selected),
                        "The selected elements: counts (peak areas) in slct, % of the "
                        "selected elements in norm, centred log-ratios in clr.",
                        0,
                    )
                )
                elements_done = True
            continue
        text = HEADER_TEXTS.get(h, "")
        if h == "suspect" and factor:
            text = (
                f"1 = Ar above {factor:g} x the median Ar of the valid points of the core, often an air gap or "
                "crack (orange row)."
            )
        elif h.endswith(" inc"):
            text = "Incoherent (Compton) scatter of the tube; never part of norm or clr."
        elif h.endswith(" coh"):
            text = "Coherent (Rayleigh) scatter of the tube; never part of norm or clr."
        if info.get("file_format") == "csv":
            text = text.replace(" (grey row)", "").replace(" (orange row)", "")
        rows.append((h, text, 0))
    # the ratio tables this run has (el_incoh needs the scatter columns, el_kcps a count rate)
    tabs = set()
    if info.get("file_format") == "csv":
        tabs = {
            tab
            for tab, name in (
                ("el_incoh", "per_incoh"),
                ("el_kcps", "per_kcps"),
                ("ratios", "cohinc_and_ratios"),
            )
            if os.path.isfile(os.path.join(root, f"{label}_ratios_{name}.csv"))
        }
    elif os.path.isfile(os.path.join(root, f"{label}_ratios.xlsx")):
        from openpyxl import load_workbook

        book = load_workbook(os.path.join(root, f"{label}_ratios.xlsx"), read_only=True)
        tabs = {name.rsplit("-", 1)[-1] for name in book.sheetnames}
        book.close()
    if selected:
        ratios = [f"{a}/{b}" for a, b in info.get("ratios") or []]
        logs = [f"ln({a}/{b})" for a, b in info.get("log_ratios") or []]
        if "el_incoh" in tabs:
            rows.append(
                (f"{selected[0]}/(inc+coh) ...", "Ratios, el_incoh: each selected element / (inc + coh).", 0)
            )
        if "el_kcps" in tabs:
            rows.append((f"{selected[0]}/kcps ...", "Ratios, el_kcps: each selected element / kcps.", 0))
        if "ratios" in tabs:
            rows.append(("coh/inc", "Ratios, ratios: coherent / incoherent scatter.", 0))
            if ratios:
                rows.append((", ".join(ratios), "Ratios, ratios: the ratios of the raw counts.", 0))
            if logs:
                rows.append((", ".join(logs), "Ratios, ratios: the log-ratios.", 0))
    return rows


def _ordered(root: str, label: str) -> list[str]:
    """The names in the results folder in the order of the work: Start's tables and images, the element files,
    split by core, plots, the record; others at the end. The guide is always among them."""
    L = label
    order = [
        f"{L}_XRF-results",
        f"{L}_XRFreps-results",
        f"{L}_IMG",
        f"{L}_slct.xlsx",
        f"{L}_slct.csv",
        f"{L}_norm.xlsx",
        f"{L}_norm.csv",
        f"{L}_clr.csv",
        f"{L}_True-Depth.csv",
        f"{L}_ratios.xlsx",
        f"{L}_ratios_per_incoh.csv",
        f"{L}_ratios_per_kcps.csv",
        f"{L}_ratios_cohinc_and_ratios.csv",
        f"{L}_slct-results",
        f"{L}_norm-results",
        f"{L}_clr-results",
        f"{L}_per-core",
        f"{L}_plots",
        "run_info.txt",
        "Log-File",
        NAME,
    ]

    def place(name: str) -> tuple[int, str]:
        key = "Log-File" if name.startswith("Log-File_") else name
        return (order.index(key), "") if key in order else (len(order), name.lower())

    return sorted(set(os.listdir(root)) | {NAME}, key=place)


def tree(root: str, label: str, info: RunInfo) -> list[tuple[str, str, int]]:
    """The results folder as a tree: (drawn line, short note, 0). Folders show their sub-folders and first files."""
    L = label
    notes = {
        f"{L}_XRF-results": "all elements, raw counts, one table per scan",
        f"{L}_XRFreps-results": "replicate scans, one table per scan",
        f"{L}_IMG": "core images",
        "LowRes_IMG": "optical images, core top on the left",
        "HighRes_IMG": "largest image of each core folder",
        "image with ruler": "with a true-depth ruler",
        "image with Ar counts": "with Ar counts (red invalid, orange suspect)",
        f"{L}_slct.xlsx": "selected elements, raw counts, all cores",
        f"{L}_slct.csv": "selected elements, raw counts, all cores",
        f"{L}_norm.xlsx": "tabs norm, clr and True-Depth, all cores",
        f"{L}_norm.csv": "norm: % of the selected elements, all cores",
        f"{L}_clr.csv": "clr: centred log-ratios, all cores",
        f"{L}_True-Depth.csv": "the True Depth file as used",
        f"{L}_ratios.xlsx": "tabs el_incoh, el_kcps and ratios, all cores",
        f"{L}_ratios_per_incoh.csv": "element / (inc + coh), all cores",
        f"{L}_ratios_per_kcps.csv": "element / kcps, all cores",
        f"{L}_ratios_cohinc_and_ratios.csv": "coh/inc, ratios and log-ratios, all cores",
        f"{L}_slct-results": "slct, one file per scan",
        f"{L}_norm-results": "norm, one file per scan",
        f"{L}_clr-results": "clr, one file per scan",
        f"{L}_per-core": "one workbook per scan with all calculations",
        f"{L}_plots": "profiles against depth, per scan",
        "run_info.txt": "how the results were made (inputs, options, messages)",
        NAME: "this overview",
    }

    def file_note(folder: str, name: str) -> str:
        if folder == f"{L}_per-core" and name.endswith(".xlsx"):
            from openpyxl import load_workbook

            book = load_workbook(os.path.join(root, folder, name), read_only=True)
            tabs = book.sheetnames
            book.close()
            return "tabs " + _and(tabs)
        if folder == f"{L}_plots":
            what = (
                "counts of the selected elements"
                if "_counts" in name
                else "ratios and log-ratios"
                if "_ratios" in name
                else ""
            )
            return f"{what}, all sections of the core" if "_whole-core_" in name and what else what
        return ""

    rows = [(f"{L}_Results/", "", 0)]
    entries = _ordered(root, label)
    for i, name in enumerate(entries):
        # no-break spaces: a long name is never wrapped away from its branch
        branch, pad = (
            ("└──\u00a0", "\u00a0" * 4) if i == len(entries) - 1 else ("├──\u00a0", "│\u00a0\u00a0\u00a0")
        )
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            note = (
                "detailed log of the run"
                if name.startswith("Log-File_")
                else notes.get(name, "not written by ITRAXpy")
            )
            rows.append((branch + name, note, 0))
            continue
        rows.append((branch + name + "/", notes.get(name, "not written by ITRAXpy"), 0))
        images = ["LowRes_IMG", "HighRes_IMG", "image with ruler", "image with Ar counts"]
        children = sorted(os.listdir(path), key=lambda c: (images.index(c) if c in images else 9, c.lower()))
        folders = [c for c in children if os.path.isdir(os.path.join(path, c))]
        files = [c for c in children if not os.path.isdir(os.path.join(path, c))]
        items: list[tuple[str, bool | None]] = [(c, c in folders) for c in folders + files[:3]]
        if len(files) > 3:
            items.append((f"… and {len(files) - 3} more", None))
        for j, (child, is_folder) in enumerate(items):
            twig = "└──\u00a0" if j == len(items) - 1 else "├──\u00a0"
            if is_folder:
                rows.append(
                    (
                        pad + twig + child + "/",
                        f"{notes.get(child, 'not written by ITRAXpy')} ({_count(os.path.join(path, child))})",
                        0,
                    )
                )
            else:
                rows.append((pad + twig + child, file_note(name, child) if is_folder is False else "", 0))
    return rows


def _count(folder: str) -> str:
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return f"{len(files)} file{'s' if len(files) != 1 else ''}"


def uses(root: str, label: str, info: RunInfo) -> list[tuple[str, str, int]]:
    """'Which file for what' rows, only for what is in the folder."""
    there = set(os.listdir(root))
    x = "csv" if info.get("file_format") == "csv" else "xlsx"
    L = label
    rows = []

    def add(task: str, name: str, where: str) -> None:
        if name in there:
            rows.append((task, where, 0))

    whole = os.path.isdir(os.path.join(root, f"{L}_plots")) and any(
        "_whole-core_" in name for name in os.listdir(os.path.join(root, f"{L}_plots"))
    )
    add(
        "A first look at the cores",
        f"{L}_plots",
        f"{L}_plots/: counts and ratios against depth, one figure per "
        "scan, named after it"
        + ("; cores of several sections also as a whole (_whole-core_)." if whole else "."),
    )
    add("Raw counts of the selected elements", f"{L}_slct.{x}", f"{L}_slct.{x}: all scans in one table.")
    if x == "xlsx":
        add(
            "Element profiles without the effect of the total counts",
            f"{L}_norm.xlsx",
            f"{L}_norm.xlsx, tab norm: each element in % of the selected elements.",
        )
        add(
            "Statistics: correlations, PCA, clustering",
            f"{L}_norm.xlsx",
            f"{L}_norm.xlsx, tab clr: centred log-ratios, free of the closure effect of percentages.",
        )
        add(
            "Two elements against each other (e.g. Ca/Ti, Zr/Rb)",
            f"{L}_ratios.xlsx",
            f"{L}_ratios.xlsx, tab ratios: a/b and ln(a/b).",
        )
        add(
            "Effects of matrix, density or water content",
            f"{L}_ratios.xlsx",
            f"{L}_ratios.xlsx, tab el_incoh: each element / (inc + coh).",
        )
        add(
            "Effects of a changing count rate",
            f"{L}_ratios.xlsx",
            f"{L}_ratios.xlsx, tab el_kcps: element / kcps.",
        )
        add(
            "Organic matter or water (proxy)",
            f"{L}_ratios.xlsx",
            f"{L}_ratios.xlsx, tab ratios, column coh/inc.",
        )
    else:
        add(
            "Element profiles without the effect of the total counts",
            f"{L}_norm.csv",
            f"{L}_norm.csv: each element in % of the selected elements.",
        )
        add("Statistics: correlations, PCA, clustering", f"{L}_clr.csv", f"{L}_clr.csv: centred log-ratios.")
        add(
            "Two elements against each other (e.g. Ca/Ti, Zr/Rb)",
            f"{L}_ratios_cohinc_and_ratios.csv",
            f"{L}_ratios_cohinc_and_ratios.csv: a/b, ln(a/b) and coh/inc.",
        )
        add(
            "Effects of matrix, density, count rate",
            f"{L}_ratios_per_incoh.csv",
            f"{L}_ratios_per_incoh.csv and {L}_ratios_per_kcps.csv.",
        )
    add(
        "One scan (core section) at a time",
        f"{L}_per-core",
        f"{L}_per-core/<scan>.xlsx: all calculations of one scan as tabs.",
    )
    add(
        "One scan (core section) at a time",
        f"{L}_norm-results",
        f"{L}_slct-results/, {L}_norm-results/, "
        f"{L}_clr-results/: one file per scan and "
        f"calculation; the ratios only in {L}_ratios.",
    )
    add(
        "All measured elements (selected or not)",
        f"{L}_XRF-results",
        f"{L}_XRF-results/: every element the scanner measured, raw counts, one table per scan.",
    )
    add(
        "Core photos with a depth ruler",
        f"{L}_IMG",
        f"{L}_IMG/: optical images, with ruler and Ar counts if ticked.",
    )
    if f"{L}_XRF-results" in there:
        argon = os.path.isdir(os.path.join(root, f"{L}_IMG", "image with Ar counts"))
        rows.append(
            (
                "Which points to trust",
                ("The grey and orange rows" if info.get("suspect_factor") else "The grey rows")
                + " of the tables (see Highlighted rows below)"
                + (" and the images with Ar counts." if argon else "."),
                0,
            )
        )
    add("How these results were made", "run_info.txt", "run_info.txt: version, inputs, options and messages.")
    return rows


def build_guide(root: str, label: str, info: RunInfo) -> GuideContent:
    """Read the results folder and assemble content without creating or changing files."""
    selected = info.get("selected")
    factor = info.get("suspect_factor")
    xlsx = info.get("file_format") != "csv"
    shaded = (
        f"{f'{factor:g}' if factor else 'the chosen factor'} x the median Ar of the valid points of the core"
    )
    # Describe the selected invalid-point policy.
    invalid = {
        "empty": "In slct, norm, clr and the ratios their rows and depths are kept, but the values are empty "
                 '(option "Invalid points"); the counts are in the XRF-results tables.',
        "leave out": 'They were left out of slct, norm, clr and the ratios (option "Invalid points"); the '
                     "XRF-results tables still have them.",
    }.get(
        info.get("invalid_points", ""),
        "Their values are unreliable; leave them out of interpretations (the option "
        '"Invalid points" can empty or leave out such rows).',
    )
    highlights_title = "Highlighted rows in the Excel tables" if xlsx else "Marked rows in the tables"
    swatches: list[tuple[str, str]] = []
    if xlsx:
        swatches = [
            (
                "Grey",
                "validity = 0: the scanner could not fit the spectrum of this measurement well, "
                f"e.g. at a crack, a gap or the end of the core. {invalid}",
            )
        ]
        if factor:
            swatches.append(
                (
                    "Orange",
                    f"suspect = 1: Ar is above {shaded}. Air between the detector and the "
                    "sediment (a gap, crack or uneven surface) lowers the other elements, "
                    "so check these points, e.g. in the images with Ar counts. Nothing "
                    "is removed.",
                )
            )
        highlights_note = (
            "A row that is both is grey. The same rows are marked in the columns validity and suspect, "
            "which you can use to filter."
            if factor
            else "No suspect points were marked (the suspect factor was empty). The column validity marks the "
                 "grey rows, which you can use to filter."
        )
    else:
        highlights_note = (
                "csv files have no colours: the column validity (0 = the spectrum could not be fitted well) "
                + (
                    f"and the column suspect (1 = Ar above {shaded}, e.g. an air gap) mark these rows. "
                    if factor
                    else "marks these rows; no suspect points were marked. "
                )
                + f"validity = 0: {invalid}"
        )

    # the column headers as the tables of this run have them
    headers = columns(root, label, info)
    columns_title = "Column headers in the Excel tables" if xlsx else "Column headers in the tables"
    columns_intro = (
        f"The combined tables ({label}_slct, _norm and _ratios) start with two title rows, the name and "
        f"the scanner settings, so their column headers are in row 3. The per-core files and the tables in "
        f"{label}_XRF-results have them in row 1. norm and clr have the same columns as slct; the ratio "
        f"tables the same first columns, then their ratios."
        if headers
        else "The tables are written when the elements are selected (Write element files); this guide then lists "
             "their column headers."
    )
    n = len(selected) if selected else None
    formulas_intro = (
        r"$x_i$ = the counts of element $i$ in one measurement; $n$ = the number of selected elements"
        f"{f' (here {n})' if n else ''}; the sums run over the selected elements of that measurement."
    )
    formulas: list[tuple[str, str, str | None]] = [
        (
            "Depth",
            "The true depth of a measurement, in mm.",
            r"$\mathrm{DepthID} = \mathrm{position} - \mathrm{StartPosition} + \mathrm{StartID}$",
        ),
        (
            "Zeroes",
            "For norm, clr and the log-ratios only, a 0 is replaced by half the smallest positive count of "
            "that element in the same core (CoreID); a core without any uses the smallest of all cores. With "
            "invalid points empty or left out, only the valid points count. slct keeps the raw counts.",
            r"$x_i = 0 \quad\rightarrow\quad x_i = \dfrac{1}{2}\ \min\,(x_i > 0)\ \ \mathrm{in\ the\ same\ core}$",
        ),
        (
            "norm",
            "Each selected element in % of the sum of the selected elements; every row adds up to 100 %.",
            r"$\mathrm{norm}_i = \dfrac{x_i}{\sum_j\ x_j} \times 100\ \%$",
        ),
        (
            "clr",
            "The log of each element minus the mean log of the selected elements; every row adds up to 0. "
            "The better choice for correlations and PCA.",
            r"$\mathrm{clr}_i = \ln x_i - \dfrac{1}{n}\ \sum_j\ \ln x_j$",
        ),
        (
            "Ratios",
            "Of the raw counts (empty where the divisor is 0), the log-ratios with zeroes replaced as above; "
            "inc and coh are the scatter peaks of the tube.",
            r"$\dfrac{a}{b}, \qquad \ln \dfrac{a}{b}, \qquad \dfrac{x_i}{\mathrm{inc} + \mathrm{coh}}, \qquad "
            r"\dfrac{x_i}{\mathrm{kcps}}, \qquad \dfrac{\mathrm{coh}}{\mathrm{inc}}$",
        ),
        (
            "Missing values",
            "A measurement without a value for one of the selected elements gets no norm and no clr.",
            None,
        ),
    ]
    return GuideContent(
        title=f"ITRAX XRF results: {label}",
        footer=f"ITRAXpy Bern {info.get('version', '')}   ·   {label}_Results",
        facts=[
            ("Measured", measured(info.get("measured") or [])),
            ("Operator", info.get("operator") or "not given"),
            ("Processed", f"{time.strftime('%Y-%m-%d')}, ITRAXpy Bern {info.get('version', '')}"),
        ],
        summary=report(label, info),
        structure=tree(root, label, info),
        file_uses=uses(root, label, info),
        highlights_title=highlights_title,
        swatches=swatches,
        highlights_note=highlights_note,
        columns_title=columns_title,
        columns_intro=columns_intro,
        headers=headers,
        formulas_intro=formulas_intro,
        formulas=formulas,
    )
