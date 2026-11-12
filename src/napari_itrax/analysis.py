from __future__ import annotations

import re
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from napari_itrax.models import (
    GROUP_COLUMN,
    INTERNAL_COLUMNS,
    INVALID_EMPTY,
    INVALID_KEEP,
    INVALID_LEAVE_OUT,
    SCAN_COLUMN,
    ITRAXpyError,
    ScanResult,
    SelectionTables,
)
from napari_itrax.readers import core_group, describe_items, format_number, parse_number
from napari_itrax.scans import TUBES

ELEMENT_SYMBOLS = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr "
    "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split()
)

SCATTER_RE = re.compile(r"[A-Z][a-z]? (inc|coh)")


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


def flag_suspect(results: list[ScanResult], factor: float, notes: list[str]) -> None:
    """Add a 'suspect' column (1/0): Ar above factor x the median Ar of the same core.

    Ar rises where the beam hits air (gaps, cracks, a poor core surface). The median is taken over the points
    with validity != 0. Nothing is removed; the rows are shaded in the outputs.
    """
    tables = [(scan, table) for scan, table, _ in results if "Ar" in table.columns]
    missing = [scan.display for scan, table, _ in results if "Ar" not in table.columns]
    if missing:
        notes.append(f"No 'Ar' column, so no suspect points could be marked for:\n{describe_items(missing)}")
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
        median = medians.get(core_group(scan), np.nan)
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
        start_id = cast(float, parse_number(scan.td_row["StartID"]))
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
                f"twice. Please check StartID / StartPosition:\n{describe_items(overlaps)}"
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


def calculate_selection(
        combined: pd.DataFrame,
        meta_columns: list[str],
        selected: list[str],
        *,
        tube: str,
        add_inc_coh: bool = False,
        log_ratios: Sequence[tuple[str, str]] = (),
        ratios: Sequence[tuple[str, str]] = (),
) -> SelectionTables:
    """Calculate selected counts, transformations and ratios without writing files."""
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

    # ratios: element/(inc+coh), element/kcps, coh/inc and log-ratios, with the same meta columns
    per_scatter, per_kcps, other = ratio_tables(combined, selected, tube, log_ratios, notes, ratios)
    return SelectionTables(
        meta=meta,
        counts=counts,
        selected=slct_table,
        normalised=norm_table,
        clr=clr_table,
        per_scatter=per_scatter,
        per_kcps=per_kcps,
        ratios=other,
        notes=notes,
    )
