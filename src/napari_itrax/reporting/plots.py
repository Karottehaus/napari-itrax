from __future__ import annotations

import math
import os
import re
import textwrap

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, MultipleLocator
from numpy.typing import NDArray
from PIL import Image

from ..models import GROUP_COLUMN, SCAN_COLUMN, Photo, ScanPhoto

Image.MAX_IMAGE_PIXELS = None
LINE = "#1F4E79"
INVALID = "#9E9E9E"
SUSPECT = "#FF9800"
PANEL_WIDTH = 1.15  # inches per panel
DPI = 200
CM_PER_INCH = 2.0  # detail figures: 2 cm of core per inch -> 1 mm = 10 pixels at 200 dpi
PAGE_CM = 50  # longest detail figure; longer cores are split into pages of this length
MIN_HEIGHT = 4.0  # inches of plot area at least
PHOTO_MAX_PX = 8000  # the photo keeps its own resolution up to this many pixels
TITLES = {"counts": "raw counts", "ratios": "coh/inc and ratios"}
SEPARATORS = "_- ."
ON_TOP = 0.5  # sections overlapping by more than this part of the shorter one: StartID not set, no whole core
EDGE_CM = 0.5  # a detail page is not started for points less than this far past a round depth


def _with_values(table: pd.DataFrame) -> list[str]:
    """Columns with at least one value other than 0: a results file that was not re-evaluated has only zeroes."""
    return [c for c in table.columns if table[c].fillna(0).ne(0).any()]


def _safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))


def core_name(labels: list[str], group: str) -> str:
    """Name of a core of several sections: what their names share ('AP2' for AP2_104-204, AP2_204-306, ...)."""
    prefix = os.path.commonprefix(list(labels))
    if any(len(label) > len(prefix) and label[len(prefix)] not in SEPARATORS for label in labels):
        prefix = re.sub(r"[^_\- .]*$", "", prefix)  # not in the middle of a word or number
    return prefix.rstrip(SEPARATORS) or f"core {group}"


def _bands(depth: NDArray[np.float64], flags: NDArray[np.bool_]) -> list[tuple[float, float]]:
    """Depth ranges (top, bottom) of consecutive flagged measurements, each reaching half way to its neighbours but
    not further than half the usual step, so a band does not fill the gap to the next section."""
    if not flags.any():
        return []
    steps = np.diff(depth)
    step = np.median(steps[steps > 0]) if (steps > 0).any() else 1.0
    middles = (depth[1:] + depth[:-1]) / 2
    tops = np.maximum(np.concatenate([[-np.inf], middles]), depth - step / 2)
    bottoms = np.minimum(np.concatenate([middles, [np.inf]]), depth + step / 2)
    bands: list[list[float]] = []
    for i in np.flatnonzero(flags):
        if bands and tops[i] <= bands[-1][1] + 1e-9:
            bands[-1][1] = bottoms[i]
        else:
            bands.append([tops[i], bottoms[i]])
    return [(band[0], band[1]) for band in bands]


def _with_gaps(
        depth: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """NaN between sections, so separate sections are not joined by a straight line."""
    if len(depth) < 3:
        return depth, values
    gap = np.flatnonzero(np.diff(depth) > 3 * np.median(np.diff(depth)))
    return np.insert(depth, gap + 1, np.nan), np.insert(values, gap + 1, np.nan)


def _depth_ticks(ax: Axes, top: float, bottom: float) -> None:
    """A number every cm and a tick every mm on detail figures; coarser steps on long overviews.

    The small ticks are drawn once, as one set of lines on the leftmost column: as matplotlib ticks they would
    be built for every panel (hundreds each), which made a 50 cm page take 8 s instead of 1.
    """
    span_cm = bottom - top
    major, minor = (1, 0.1) if span_cm <= 1.2 * PAGE_CM else (10, 1) if span_cm <= 300 else (50, 10)
    ax.yaxis.set_major_locator(MultipleLocator(major))
    marks = np.arange(math.ceil(top / minor), math.floor(bottom / minor) + 1) * minor
    ax.hlines(
        marks, -0.035, 0, transform=ax.get_yaxis_transform(), clip_on=False, color="black", linewidth=0.5
    )


def _profile_figure(
        title: str,
        subtitle: str,
        depth_cm: NDArray[np.float64],
        table: pd.DataFrame,
        flags: tuple[NDArray[np.bool_], NDArray[np.bool_]],
        photos: list[Photo],
        top: float,
        bottom: float,
        height: float,
        parts: NDArray[np.str_] | None = None,
) -> Figure:
    """Figure for the depth range top-bottom (cm) with a plot area height inches tall.

    parts: the section of each row; each section is drawn as its own line, so where sections overlap both are seen.
    """
    columns = _with_values(table)
    panels = len(columns) + (1 if photos else 0)
    width = 0.9 + PANEL_WIDTH * panels
    # title and settings line are wrapped to the figure width (narrow figures, mixed settings), with room made above
    heading = textwrap.wrap(title, max(int((width - 1.0) * 7.5), 12))
    note = textwrap.wrap(
        subtitle + "   Shaded: grey = validity 0, orange = suspect (Ar).", int((width - 1.0) * 15)
    )
    note_y = 0.25 + 0.22 * len(heading)
    head = note_y + 0.15 * len(note) + 0.4
    fig = Figure(figsize=(width, height + head + 0.45))
    # the gap keeps the outer x numbers of neighbouring panels apart (e.g. '1.2' and '3.2')
    axes = np.atleast_1d(fig.subplots(1, panels, sharey=True, gridspec_kw={"wspace": 0.25}))
    fig.subplots_adjust(
        left=0.9 / width,
        right=1 - 0.15 / width,
        top=1 - head / fig.get_figheight(),
        bottom=0.45 / fig.get_figheight(),
    )

    panel_axes = list(axes)
    if photos:
        ax = panel_axes.pop(0)
        for path, photo_top, photo_bottom in photos:
            photo_top, photo_bottom = photo_top / 10, photo_bottom / 10
            show_top, show_bottom = max(photo_top, top), min(photo_bottom, bottom)
            if show_bottom <= show_top:
                continue
            with Image.open(path) as image:
                upright = image.rotate(-90, expand=True)  # shallow end (left) to the top
            pixels_per_cm = upright.size[1] / (photo_bottom - photo_top)  # pixels per cm
            photo_crop = upright.crop(
                (
                    0,
                    int((show_top - photo_top) * pixels_per_cm),
                    upright.size[0],
                    max(int(np.ceil((show_bottom - photo_top) * pixels_per_cm)), 1),
                )
            )
            photo_crop.thumbnail((PHOTO_MAX_PX, PHOTO_MAX_PX))
            ax.imshow(
                np.asarray(photo_crop.convert("RGB")),
                extent=(0, 1, show_bottom, show_top),
                aspect="auto",
                interpolation="bilinear",
            )
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title("core photo", fontsize=9)

    invalid, suspect = flags
    for ax, column in zip(panel_axes, columns):
        for bands, colour, alpha in (
                (_bands(depth_cm, invalid), INVALID, 0.35),
                (_bands(depth_cm, suspect & ~invalid), SUSPECT, 0.25),
        ):
            for band_top, band_bottom in bands:
                if band_bottom >= top and band_top <= bottom:
                    ax.axhspan(band_top, band_bottom, color=colour, alpha=alpha, linewidth=0)
        column_values = table[column].to_numpy(dtype=float)
        for part in [None] if parts is None else pd.unique(parts):
            rows = slice(None) if part is None else parts == part
            y, x = _with_gaps(depth_cm[rows], column_values[rows])
            ax.plot(x, y, color=LINE, linewidth=0.8)
            shown = ~np.isnan(x) & ~np.isnan(y)
            alone = shown & ~np.r_[False, shown[:-1]] & ~np.r_[shown[1:], False]
            if alone.any():  # e.g. a valid point between invalid ones: no line segment to draw
                ax.plot(x[alone], y[alone], "o", color=LINE, markersize=2, linestyle="none")
        ax.set_title(column, fontsize=9)
        ax.margins(x=0.06)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", which="major", color="#E0E0E0", linewidth=0.5)
        if not column.startswith("ln("):
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
            ax.xaxis.get_offset_text().set_fontsize(7)

    axes[0].set_ylim(bottom, top)  # depth increases downwards
    _depth_ticks(axes[0], top, bottom)
    axes[0].set_ylabel("True depth (cm)", fontsize=9, labelpad=8)
    axes[0].tick_params(axis="y", labelsize=8, length=5)
    fig.suptitle(
        "\n".join(heading),
        x=0.9 / width,
        ha="left",
        y=1 - 0.2 / fig.get_figheight(),
        va="top",
        fontsize=12,
        fontweight="bold",
        linespacing=1.1,
    )
    fig.text(
        0.9 / width,
        1 - note_y / fig.get_figheight(),
        "\n".join(note),
        fontsize=8,
        color="#555555",
        va="top",
        linespacing=1.3,
    )
    return fig


def _pages(top: float, bottom: float) -> list[tuple[float, float]]:
    """Depth windows (cm) of PAGE_CM on round depths covering the measurements from top to bottom. A point just
    above a round depth (e.g. at -0.05 mm, a little above StartPosition) or just below one gets no page of its own."""
    first = math.floor((top + EDGE_CM) / PAGE_CM) * PAGE_CM
    last = max(math.floor((bottom - EDGE_CM) / PAGE_CM) * PAGE_CM, first)
    return [(start, start + PAGE_CM) for start in range(first, last + 1, PAGE_CM)]


def _jobs(
        base: str,
        top: float,
        bottom: float,
        depth_cm: NDArray[np.float64],
        values: pd.DataFrame,
        detail_pages: bool,
) -> list[tuple[str, float, float, float, int]]:
    """(path, top, bottom, plot height in inches, dpi) of the figures for one depth range (cm).

    Up to 1.2 x PAGE_CM: one detail figure '<base>.png'. Longer: an overview under the same name; with detail_pages
    it is called '<base>_overview.png' and followed by detail pages '<base>_<top>-<bottom>cm.png' (many and large
    files for long cores, so it is a choice).
    """
    if bottom - top <= 1.2 * PAGE_CM:
        return [(base + ".png", top, bottom, max((bottom - top) / CM_PER_INCH, MIN_HEIGHT), DPI)]
    overview = float(np.clip((bottom - top) / 10, 7, 30))
    if not detail_pages:
        return [(base + ".png", top, bottom, overview, 150)]
    jobs = [(base + "_overview.png", top, bottom, overview, 150)]
    for page_top, page_bottom in _pages(float(np.min(depth_cm)), float(np.max(depth_cm))):
        inside = (depth_cm >= page_top) & (depth_cm < page_bottom)
        if values[inside].notna().any().any():  # not e.g. the gap between two sections
            jobs.append(
                (
                    f"{base}_{page_top:04d}-{page_bottom:04d}cm.png",
                    page_top,
                    page_bottom,
                    PAGE_CM / CM_PER_INCH,
                    DPI,
                )
            )
    return jobs


def write_profiles(
        folder: str,
        combined: pd.DataFrame,
        core_label: str,
        settings: str,
        figures: dict[str, pd.DataFrame],
        scans: list[ScanPhoto],
        detail_pages: bool = False,
        notes: list[str] | None = None,
) -> list[str]:
    """PNG figures in folder, named after the scans like the per-core files: '<scan>_counts.png', '<scan>_ratios.png'.

    A core of several sections (one CoreID) whose sections do not overlap in depth also gets an overview of the whole
    core, '<core name>_whole-core_<name>.png'. figures: {name: table with the values, rows as in combined}.
    scans: list of (scan label, core group, settings text, photo) from scan_photos, photo being
    (image path, top mm, bottom mm) or None. Notes for the user are added to notes. Returns the paths written.
    """
    notes = [] if notes is None else notes
    os.makedirs(folder, exist_ok=True)
    # the figures of an earlier Write element files (other elements, ratios or pages) are replaced, not kept
    stale = []
    for name in os.listdir(folder):
        if name.lower().endswith(".png"):
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                stale.append(name)
    if stale:
        notes.append(
            f"{len(stale)} plot(s) of an earlier Write element files could not be removed from {folder} "
            f"(open in another program?): {', '.join(stale[:5])}{' ...' if len(stale) > 5 else ''}"
        )
    main = pd.to_numeric(combined["RepID"], errors="coerce").fillna(0).eq(0)
    data = combined[main & combined["DepthID"].notna()]
    groups = (
        data[GROUP_COLUMN] if GROUP_COLUMN in data.columns else pd.Series("1", index=data.index)
    ).astype(str)
    labels = data[SCAN_COLUMN].astype(str) if SCAN_COLUMN in data.columns else groups
    info = {str(label): (str(group), text, photo) for label, group, text, photo in scans}
    written = []

    def draw(base: str, title: str, subtitle: str, index: pd.Index, photos: list[Photo], pages: bool) -> bool:
        """The figures of the rows index; False when no column has a value to show."""
        order = data.loc[index, "DepthID"].sort_values(kind="stable").index
        depth_cm = data.loc[order, "DepthID"].to_numpy(dtype=float) / 10
        parts = labels.loc[order].to_numpy()
        validity = pd.to_numeric(data.loc[order, "validity"], errors="coerce") if "validity" in data else None
        invalid = validity.eq(0).to_numpy() if validity is not None else np.zeros(len(order), bool)
        suspect = (
            pd.to_numeric(data.loc[order, "suspect"], errors="coerce").eq(1).to_numpy()
            if "suspect" in data
            else np.zeros(len(order), bool)
        )
        margin = max(0.02 * (float(np.max(depth_cm)) - float(np.min(depth_cm))), 0.3)
        top, bottom = float(np.min(depth_cm)) - margin, float(np.max(depth_cm)) + margin
        drawn = False
        for name, table in figures.items():
            values = table.loc[order]
            if values.empty or not _with_values(values):
                continue
            for path, page_top, page_bottom, height, dpi in _jobs(
                    f"{base}_{name}", top, bottom, depth_cm, values, pages
            ):
                fig = _profile_figure(
                    f"{title}  -  {TITLES.get(name, name)}",
                    subtitle,
                    depth_cm,
                    values,
                    (invalid, suspect),
                    photos,
                    page_top,
                    page_bottom,
                    height,
                    parts,
                )
                fig.savefig(path, dpi=dpi)
                written.append(path)
            drawn = True
        return drawn

    cores: dict[str, list[tuple[str, pd.Index, str, Photo | None]]] = {}
    empty: list[str] = []
    all_invalid: list[str] = []
    for group_label, index in data.groupby(labels, sort=False).groups.items():
        label = str(group_label)
        group, text, photo = info.get(label, (groups[index[0]], "", None))
        cores.setdefault(group, []).append((label, index, text, photo))
        subtitle = f"{core_label}, CoreID {group}.  " + (f"Settings: {text}." if text else settings)
        if not draw(
                os.path.join(folder, _safe(label)), label, subtitle, index, [photo] if photo else [], detail_pages
        ):
            validity = (
                pd.to_numeric(data.loc[index, "validity"], errors="coerce") if "validity" in data else None
            )
            (all_invalid if validity is not None and validity.eq(0).all() else empty).append(label)

    for group, sections in cores.items():
        if len(sections) < 2:
            continue
        name = core_name([label for label, *_ in sections], group)
        ranges = [
            (data.loc[index, "DepthID"].min(), data.loc[index, "DepthID"].max())
            for _, index, _, _ in sections
        ]
        on_top = [
            f"{sections[i][0]} and {sections[j][0]}"
            for i, a in enumerate(ranges)
            for j, b in enumerate(ranges)
            if i < j and min(a[1], b[1]) - max(a[0], b[0]) > ON_TOP * min(a[1] - a[0], b[1] - b[0])
        ]
        if on_top:
            notes.append(
                f"No plot of the whole core {name} (CoreID {group}): sections lie on top of each other in "
                f"true depth ({'; '.join(on_top[:4])}{' ...' if len(on_top) > 4 else ''}). Each section has "
                "its own plots. For the whole core, check StartID in the True Depth file and leave rescans "
                "of the same depths (e.g. _takeII) out of it."
            )
            continue
        texts = list(dict.fromkeys(text for *_, text, _ in sections if text))
        subtitle = f"{core_label}, CoreID {group}, {len(sections)} sections.  " + (
            f"Settings: {'; '.join(texts)}." if texts else settings
        )
        # sections without values (not re-evaluated) are left out of the curves; their photo stays
        draw(
            os.path.join(folder, _safe(name) + "_whole-core"),
            f"{name}  (whole core)",
            subtitle,
            data.index[groups.eq(group) & ~labels.isin(empty + all_invalid)],
            [photo for *_, photo in sections if photo],
            False,
        )
    if empty:
        notes.append(
            f"No plots for {', '.join(empty)}: no element counts other than 0 (the results file was "
            "probably not re-evaluated)."
        )
    if all_invalid:
        notes.append(f"No plots for {', '.join(all_invalid)}: all its measurements have validity = 0.")
    return written
