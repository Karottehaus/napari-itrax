from __future__ import annotations

import os
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
from natsort import natsorted, ns
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from napari_itrax.models import ArgonCurve, OutputFolders, Scan, ScanPhoto, ScanResult
from napari_itrax.readers import core_group, describe_items, format_number, has_value, id_text, parse_number
from napari_itrax.imaging.ruler import load_font
from napari_itrax.scans import settings_text

IMAGE_EXTENSIONS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp")


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
            "Some images were not found or could not be read, and were skipped:\n"
            + describe_items(missing_images)
        )


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


def find_optical_image(folder: str) -> str | None:
    """optical.tif, or the multi-sensor core logger's optical_<name>.tif."""
    path = os.path.join(folder, "optical.tif")
    if os.path.isfile(path):
        return path
    for name in natsorted(os.listdir(folder), alg=ns.IGNORECASE):
        lower = name.lower()
        if lower.startswith("optical") and lower.endswith((".tif", ".tiff")) and "lowres" not in lower:
            return os.path.join(folder, name)
    return None


def photo_scale(scan: Scan, width: int) -> tuple[float, float] | None:
    """(true depth in mm at the left edge of the 180-degree rotated optical image, mm per pixel), or None.

    The image spans 'Optical Start' to 'Optical End' of document.txt (scanner mm); depth = DepthID of the XRF data.
    """
    o_start, o_end = scan.document.get("optical_start"), scan.document.get("optical_end")
    start_id = cast(float, parse_number(scan.td_row["StartID"]))
    start_position = parse_number(scan.td_row["StartPosition"])
    if start_position is None:
        start_position = scan.document.get("start")
    if o_start is None or o_end is None or start_id is None or start_position is None or o_end <= o_start:
        return None
    return o_start - start_position + start_id, (o_end - o_start) / width


def scan_photos(scans: list[Scan]) -> list[ScanPhoto]:
    """The main scans for the profile plots: list of (label, core group, settings text, photo), photo being
    (image path, depth of its top, depth of its bottom) in mm, or None."""
    return [(s.label, core_group(s), settings_text(s.settings), s.photo) for s in scans if s.rep_id == 0]


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
    label_font, title_font = load_font(20), load_font(16)
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
        f"_{part}{id_text(scan.td_row[column])}"
        for part, column in (("Core", "CoreID"), ("Section", "SectionID"))
        if has_value(scan.td_row[column])
    )
    missing = []

    optical = find_optical_image(scan.folder)
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
