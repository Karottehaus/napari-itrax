from __future__ import annotations

import os
import re
from collections import Counter
from typing import cast

import numpy as np
import pandas as pd

from napari_itrax.models import ITRAXpyError, Scan, ScanFolder, Settings
from napari_itrax.readers import TD_COLUMNS, describe_items, format_number, has_value, normalise_name, parse_number

TUBES = ["Cr", "Mo", "Rh", "Ag"]

TUBE_NAMES = {
    "Cr": "Chromium (Cr)",
    "Mo": "Molybdenum (Mo)",
    "Rh": "Rhodium (Rh)",
    "Ag": "Silver (Ag)",
    "Cu": "Copper (Cu)",
    "W": "Tungsten (W)",
}

SECTION_RE = re.compile(r"(?<![a-z])section[\s_\-]*(\d+)(?!\d)", re.IGNORECASE)

SECTION_NAME_RE = re.compile(r"(.+?)[_ ]\d+(?:[.,]\d+)?-\d+(?:[.,]\d+)?(?:[_ ].*)?", re.IGNORECASE)

REPLICATE_RE = re.compile(r"(?:\.|[_\- ]rep\s*)(\d+)$", re.IGNORECASE)


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
        f" (SectionID {format_number(section)})" if parse_number(section) is not None else ""
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
            if has_value(value):
                value = str(value).strip()
                if os.path.isabs(value):  # a path copied from the Explorer
                    relative = os.path.relpath(os.path.normpath(value), os.path.dirname(input_folder))
                    value = relative if not relative.startswith("..") else value
                folder_parts[i] = [p for p in re.split(r"[\\/]+", normalise_name(value)) if p]

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

    lowered_names = {f.folder: [normalise_name(n) for n in f.names] for f in found}

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
        prepare = (lambda name: str(name).strip()) if case_sensitive else normalise_name
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
                f"{describe_items(_describe_row(td, i) for i in rows)}\n\n"
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
            f"each row:\n{describe_items(no_section)}"
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
                  f"Scan folders found:\n{describe_items(f.display for f in found)}\n\n"
                  f"Core names in the True Depth file:\n{describe_items(td['Core'])}"
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
            f"No scan folder was found for {len(missing)} row(s) of the True Depth file:\n{describe_items(missing)}"
        )
    if no_section:
        notes.append(
            f"{len(no_section)} scan folder(s) belong to a core with several rows in the True Depth file, "
            "but their section could not be matched to a SectionID (use 'Section N' folder names or a "
            f"'Folder' column). They were not processed:\n{describe_items(no_section)}"
        )
    if ignored:
        notes.append(
            f"{len(ignored)} scan folder(s) are not listed in the True Depth file and were not "
            f"processed:\n{describe_items(ignored)}"
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
                f"{describe_items(f.display for f, _, _ in mains)}\n\n"
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
    row_section = parse_number(td.at[i, "SectionID"])
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
                        normalise_name(recorded) != normalise_name(typed)
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
            f"the document.txt values were used:\n{describe_items(where)}"
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
            f"{describe_items(others)}\nThey are combined in one file; check that this is what you want."
        )
    main_tube = Counter(s.settings["tube"] for s in scans).most_common(1)[0][0]
    return text, main_tube


def settings_text(settings: Settings) -> str:
    """'Rhodium (Rh), 20 s, 30 kV, 50 mA' for one scan's settings (see resolve_settings); '' when not known."""
    if not all(key in settings for key in ("tube", "time", "voltage", "current")):
        return ""
    return (
        f"{tube_label(settings['tube'])}, {format_number(settings['time'])} s, "
        f"{format_number(settings['voltage'])} kV, {format_number(settings['current'])} mA"
    )
