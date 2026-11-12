from __future__ import annotations

import math
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from napari_itrax.readers import read_document_file

# Local instrument line scans can exceed Pillow's default image-size limit.
Image.MAX_IMAGE_PIXELS = None
# turns that bring the core top (where it is in the photo) to the left edge
TO_LEFT = {
    "left": None,
    "right": Image.Transpose.ROTATE_180,
    "top": Image.Transpose.ROTATE_90,
    "bottom": Image.Transpose.ROTATE_270,
}


def core_top_left(image: Image.Image, core_top: str) -> Image.Image:
    """The photo turned so that the core top is on the left (core_top: 'left', 'right', 'top' or 'bottom')."""
    turn = TO_LEFT[core_top]
    return image.copy() if turn is None else image.transpose(turn)


def image_dpi(path: str) -> float | None:
    """The resolution stored in the image file, or None (cameras often store a meaningless 72 or 96)."""
    try:
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
    except OSError:
        return None
    return float(dpi[0]) if dpi and dpi[0] and float(dpi[0]) > 0 else None


def optical_range_file(path: str) -> tuple[float, float] | None:
    """(Optical Start, Optical End) in scanner mm from an ITRAX document file (any name), or None."""
    doc = read_document_file(path)
    start, end = doc.get("optical_start"), doc.get("optical_end")
    return (start, end) if start is not None and end is not None and end > start else None


def document_key(stem: str) -> str:
    """A file name without the words 'document' and 'optical', lower case, separators unified: a photo and its
    document file have the same key (optical.tif / document.txt, AP1_optical.tif / AP1_document.txt, AP1.tif /
    document AP1.txt)."""
    key = re.sub(r"document|optical", "", stem.lower())
    return re.sub(r"[\s_.-]+", "_", key).strip("_")


def find_document(photo: str) -> str | None:
    """The ITRAX document file of a photo, in the photo's folder, or None.

    First a .txt file with the same name as the photo (see document_key), then a plain document.txt; only files with
    an 'Optical Start ... Optical End' line count.
    """
    folder = os.path.dirname(photo)
    key = document_key(os.path.splitext(os.path.basename(photo))[0])
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return None
    same = [n for n in names if n.lower().endswith(".txt") and document_key(os.path.splitext(n)[0]) == key]
    for name in same + [n for n in names if n.lower() == "document.txt" and n not in same]:
        if optical_range_file(os.path.join(folder, name)):
            return os.path.join(folder, name)
    return None


def document_scale(path: str, length_px: int, document: str | None = None) -> float | None:
    """mm per pixel from the ITRAX document file of a photo ('Optical Start / Optical End'), or None.

    document: the file to use; by default the one found next to the photo (find_document).
    """
    document = document or find_document(path)
    span = optical_range_file(document) if document else None
    return None if span is None else (span[1] - span[0]) / length_px


def mm_per_pixel(mode: str, value: float | None, length_px: int) -> float | None:
    """Scale from what was typed: mode 'dpi', 'pixel' (mm per pixel) or 'length' (mm along the core)."""
    if value is None or value <= 0:
        return None
    return {"dpi": 25.4 / value, "pixel": value, "length": value / length_px}[mode]


def measured_scale(length_mm: float, start: float, end: float) -> float:
    """Calibrate along the core axis, not the diagonal between markers."""
    if not all(math.isfinite(v) for v in (length_mm, start, end)) or length_mm <= 0 or abs(end - start) < 1:
        raise ValueError(
            "Use a positive finite reference length and markers at least one pixel apart along the core."
        )
    return length_mm / abs(end - start)


def _steps(px_per_mm: float) -> tuple[int, int]:
    """(smallest tick step, numbered step) in mm, so that ticks stay >= 4 and numbers >= ~140 pixels apart."""
    minor = next((s for s in (1, 5, 10, 50, 100) if s * px_per_mm >= 4), 1000)
    label = next((s for s in (10, 20, 50, 100, 200, 500, 1000) if s * px_per_mm >= 140), 5000)
    return minor, max(label, minor)


def draw_rulers(
        image: Image.Image, mm_per_px: float, rulers: list[tuple[str, float]], vertical: bool = False
) -> Image.Image:
    """The photo (core top on the left) with one strip per ruler.

    rulers: list of (title, depth in mm at the core top edge of the photo).
    vertical=False: horizontal, strips below; vertical=True: core top up, strips on the right.
    """
    if vertical:
        image = image.transpose(Image.Transpose.ROTATE_270)  # left edge (core top) to the top
    image = image.convert("RGB")
    length = image.size[1] if vertical else image.size[0]
    across = image.size[0] if vertical else image.size[1]
    u = float(np.clip(across / 500, 1, 6))  # size unit for ticks and text
    strip = round((130 if vertical else 100) * u)
    width, height = image.size
    out = Image.new(
        "RGB",
        (width + strip * len(rulers), height) if vertical else (width, height + strip * len(rulers)),
        "white",
    )
    out.paste(image, (0, 0))
    draw = ImageDraw.Draw(out)
    label_font, title_font = load_font(round(20 * u)), load_font(round(15 * u))
    px_per_mm = 1 / mm_per_px
    minor, label = _steps(px_per_mm)

    for i, (title, start_mm) in enumerate(rulers):
        base = (width if vertical else height) + i * strip  # where the strip starts (x or y)
        line = [(base, 0), (base, height)] if vertical else [(0, base), (width, base)]
        draw.line(line, fill="black", width=max(1, round(2 * u)))
        first = math.ceil(start_mm / minor)
        last = math.floor((start_mm + length * mm_per_px) / minor)
        for k in range(first, last + 1):
            depth = k * minor  # mm
            at = round((depth - start_mm) * px_per_mm)  # pixel along the core
            if depth % label == 0:
                size, thick = 28, 2.0
            elif depth % 10 == 0:
                size, thick = 15, 1.5
            elif depth % 5 == 0:
                size, thick = 10, 1
            else:
                size, thick = 6, 1
            size, thick = round(size * u), max(1, round(thick * u))
            if vertical:
                draw.line([(base, at), (base + size, at)], fill="black", width=thick)
            else:
                draw.line([(at, base), (at, base + size)], fill="black", width=thick)
            if depth % label == 0:
                text = f"{depth / 10:g}"
                # a number at either end is aligned to its tick instead of centred on it, so it is not cut off
                end = round(25 * u)
                if vertical:
                    anchor = "lt" if at < end else "lb" if at > length - end else "lm"
                    draw.text(
                        (base + size + round(4 * u), at), text, fill="black", font=label_font, anchor=anchor
                    )
                else:
                    anchor = "lt" if at < end else "rt" if at > length - end else "mt"
                    draw.text(
                        (at, base + size + round(2 * u)), text, fill="black", font=label_font, anchor=anchor
                    )
        if vertical:
            draw.text(
                (base + round(6 * u), height - round(6 * u)),
                title,
                fill="black",
                font=title_font,
                anchor="ls",
            )
        else:
            draw.text(
                (round(6 * u), base + strip - round(6 * u)), title, fill="black", font=title_font, anchor="ls"
            )
    return out


def add_ruler(
        path: str,
        folder: str,
        mm_per_px: float,
        core_top: str,
        core_depth_cm: float | None = None,
        photo_depth: bool = True,
        vertical: bool = False,
        stem: str | None = None,
) -> str:
    """Save a copy of the photo with its ruler(s) in folder; returns the path written.

    core_depth_cm: core depth at the photo's core top edge (cm), or None for no core-depth ruler.
    stem: the name of the copy before '_ruler' (default: the photo's name).
    """
    rulers = []
    if core_depth_cm is not None:
        rulers.append(("core (cm)" if vertical else "core depth (cm)", core_depth_cm * 10))
    if photo_depth:
        rulers.append(
            ("photo (cm)" if vertical else "photo depth (cm from the core top edge of the photo)", 0.0)
        )
    with Image.open(path) as image:
        turned = core_top_left(image, core_top)
    out = draw_rulers(turned, mm_per_px, rulers, vertical)
    name, ext = os.path.splitext(os.path.basename(path))
    target = os.path.join(folder, f"{stem or name}_ruler{ext}")
    if ext.lower() in (".jpg", ".jpeg"):
        out.save(target, quality=95)
    else:
        out.save(target)
    return target


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()
