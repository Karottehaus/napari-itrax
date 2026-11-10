from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from . import pipeline, ruler


@dataclass(frozen=True)
class PhotoJob:
    path: Path
    mm_per_pixel: float
    core_top: str = "left"
    core_depth_cm: float | None = None
    photo_depth: bool = True
    vertical: bool = False


def measured_scale(length_mm: float, start: float, end: float) -> float:
    """Calibrate along the core axis, not the diagonal between markers."""
    if not all(math.isfinite(v) for v in (length_mm, start, end)) or length_mm <= 0 or abs(end - start) < 1:
        raise ValueError(
            "Use a positive finite reference length and markers at least one pixel apart along the core."
        )
    return length_mm / abs(end - start)


def export_rulers(
        jobs: list[PhotoJob], folder: Path, progress: Callable[[float], None] | None = None
) -> list[Path]:
    """Validate the entire batch before writing any photos."""
    if not jobs:
        raise ValueError("Choose at least one photo.")
    names: set[str] = set()
    stems: list[str] = []
    for job in jobs:
        if not math.isfinite(job.mm_per_pixel) or job.mm_per_pixel <= 0:
            raise ValueError(f"Invalid scale for {job.path.name}.")
        if job.core_top not in ruler.TO_LEFT:
            raise ValueError("Core top must be left, right, top, or bottom.")
        if job.core_depth_cm is not None and not math.isfinite(job.core_depth_cm):
            raise ValueError("Core depth must be finite.")
        if job.core_depth_cm is None and not job.photo_depth:
            raise ValueError("Enable at least one ruler.")
        with Image.open(job.path) as image:
            image.verify()
        stem = job.path.stem
        # Deterministic uniqueness within the batch, independent of existing exports.
        candidate, number = stem, 1
        while candidate.casefold() in names:
            number += 1
            candidate = f"{stem}_{number}"
        names.add(candidate.casefold())
        stems.append(candidate)
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, (job, stem) in enumerate(zip(jobs, stems, strict=True)):
        target = ruler.add_ruler(
            str(job.path),
            str(folder),
            job.mm_per_pixel,
            job.core_top,
            job.core_depth_cm,
            job.photo_depth,
            job.vertical,
            stem,
        )
        written.append(Path(target))
        if progress is not None:
            progress((index + 1) / len(jobs))
    pipeline.write_run_info(str(folder), "Photo rulers", [repr(job) for job in jobs])
    return written
