from __future__ import annotations

import os
import textwrap

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from napari_itrax.models import RunInfo
from napari_itrax.reporting.readme_content import NAME, GuideContent, build_guide

NAVY, ORANGE, INK, MUTED, RULE = "#0B3557", "#F07F23", "#1F2D3A", "#617487", "#D6DFE8"
PAGE_W, PAGE_H, MARGIN = 8.27, 11.69, 0.75  # inches (A4)
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"


class Pages:
    """A few kinds of text placed down A4 pages, with page breaks (positions in inches from the top left)."""

    def __init__(self, path: str, footer: str) -> None:
        self.pdf = PdfPages(path, metadata={"Title": "What is where", "Creator": "ITRAXpy Bern"})
        self.footer, self.number = footer, 0
        self.fig: Figure
        self.new_page()

    def new_page(self) -> None:
        if self.number:
            self.pdf.savefig(self.fig)
        self.number += 1
        self.fig, self.y = Figure(figsize=(PAGE_W, PAGE_H)), MARGIN
        self._text(MARGIN, PAGE_H - 0.45, f"{self.footer}   ·   page {self.number}", 7.5, color=MUTED)

    def close(self) -> None:
        self.pdf.savefig(self.fig)
        self.pdf.close()

    def _text(
            self,
            x: float,
            y: float,
            text: str,
            size: float,
            weight: str = "normal",
            color: str = INK,
            family: str = SANS,
    ) -> None:
        self.fig.text(
            x / PAGE_W,
            1 - y / PAGE_H,
            text,
            fontsize=size,
            fontweight=weight,
            color=color,
            family=family,
            va="top",
            ha="left",
        )

    def _line(self, x0: float, x1: float, y: float, color: str = RULE, width: float = 0.6) -> None:
        self.fig.add_artist(Line2D([x0 / PAGE_W, x1 / PAGE_W], [1 - y / PAGE_H] * 2, color=color, lw=width))

    @staticmethod
    def wrap(text: str, width_in: float, size: float, mono: bool = False, bold: bool = False) -> list[str]:
        """Lines of text that fit width_in inches at this font size (DejaVu: ~0.56 em a character, bold 0.63,
        mono 0.6); file names are not broken at their hyphens."""
        chars = max(10, int(width_in * 72 / (size * (0.6 if mono else 0.63 if bold else 0.56))))
        # a line break in the text (e.g. between a folder and its file) is kept
        return [
            line
            for part in text.split("\n")
            for line in (textwrap.wrap(part, chars, break_on_hyphens=False) or [""])
        ] or [""]

    def formulas(
            self,
            rows: list[tuple[str, str, str | None, float]],
            left_width: float = 1.45,
            size: float = 8.6,
            math_size: float = 11.5,
    ) -> None:
        """Rows of (name, text, formula, formula height in inches): the text, then the formula typeset (matplotlib
        mathtext) below it."""
        line = size * 1.38 / 72
        for name, text, formula, tall in rows:
            right = self.wrap(text, PAGE_W - 2 * MARGIN - left_width - 0.05, size)
            height = len(right) * line + (tall + 0.1 if formula else 0) + 0.08
            self.room(height)
            self._text(MARGIN, self.y, name, size, "bold", NAVY)
            for i, part in enumerate(right):
                self._text(MARGIN + left_width, self.y + i * line, part, size)
            if formula:
                self._text(MARGIN + left_width + 0.15, self.y + len(right) * line + 0.07, formula, math_size)
            self.y += height
            self._line(MARGIN, PAGE_W - MARGIN, self.y - 0.04)
        self.y += 0.08

    def swatches(self, rows: list[tuple[str, str, str]], left_width: float = 1.45, size: float = 8.6) -> None:
        """Rows of (fill colour, name, text): a small box in the colour, the name beside it, the text on the right."""
        from matplotlib.patches import Rectangle

        line = size * 1.38 / 72
        for colour, name, text in rows:
            right = self.wrap(text, PAGE_W - 2 * MARGIN - left_width - 0.05, size)
            height = max(len(right) * line, 0.22) + 0.08
            self.room(height)
            w, h = 0.32, 0.17
            self.fig.add_artist(
                Rectangle(
                    (MARGIN / PAGE_W, 1 - (self.y + h) / PAGE_H),
                    w / PAGE_W,
                    h / PAGE_H,
                    transform=self.fig.transFigure,
                    facecolor=colour,
                    edgecolor="#9AA8B6",
                    linewidth=0.6,
                )
            )
            self._text(MARGIN + w + 0.1, self.y + 0.01, name, size, "bold", NAVY)
            for i, part in enumerate(right):
                self._text(MARGIN + left_width, self.y + i * line, part, size)
            self.y += height
            self._line(MARGIN, PAGE_W - MARGIN, self.y - 0.04)
        self.y += 0.08

    def room(self, height: float) -> None:
        if self.y + height > PAGE_H - MARGIN:
            self.new_page()

    def title(self, text: str, facts: list[tuple[str, str, float]]) -> None:
        """The title, then facts = [(label, value, width in inches)] side by side: a small label above its value."""
        self._text(MARGIN, self.y, text, 19, "bold", NAVY)
        self.y += 0.45
        x, height = MARGIN, 0.0
        for label, value, width in facts:
            self._text(x, self.y, label.upper(), 7.2, "bold", MUTED)
            lines = self.wrap(value, width - 0.15, 9.5)
            for i, line in enumerate(lines):
                self._text(x, self.y + 0.15 + i * 0.18, line, 9.5)
            height = max(height, 0.15 + len(lines) * 0.18)
            x += width
        self.y += height
        self._line(MARGIN, PAGE_W - MARGIN, self.y + 0.08, ORANGE, 2)
        self.y += 0.28

    def heading(self, text: str) -> None:
        self.room(0.75)
        self.y += 0.12
        self._text(MARGIN, self.y, text, 12.5, "bold", NAVY)
        self.y += 0.3

    def paragraph(self, text: str, size: float = 9, color: str = INK) -> None:
        lines = self.wrap(text, PAGE_W - 2 * MARGIN, size)
        self.room(len(lines) * size * 1.4 / 72)
        for line in lines:
            self._text(MARGIN, self.y, line, size, color=color)
            self.y += size * 1.4 / 72
        self.y += 0.07

    def rows(
            self,
            rows: list[tuple[str, str, int]],
            left_width: float = 2.45,
            size: float = 8.6,
            mono_left: bool = False,
            indent: float = 0.0,
            rules: bool = True,
            right_color: str = INK,
    ) -> None:
        """Two columns: a name (bold, or monospace) and its description; rows = [(name, text, level)].
        rules=False: no line under each row and less space (for the folder tree)."""
        line = size * 1.38 / 72
        for name, text, level in rows:
            x = MARGIN + indent + 0.22 * level
            left = self.wrap(name, left_width - 0.22 * level - 0.12, size, mono_left, not mono_left)
            right = self.wrap(text, PAGE_W - MARGIN - (MARGIN + left_width + indent) - 0.05, size)
            height = max(len(left), len(right)) * line + (0.07 if rules else 0.015)
            self.room(height)
            for i, part in enumerate(left):
                self._text(
                    x,
                    self.y + i * line,
                    part,
                    size,
                    "normal" if mono_left else "bold",
                    NAVY if not mono_left else INK,
                    MONO if mono_left else SANS,
                )
            for i, part in enumerate(right):
                self._text(MARGIN + left_width + indent, self.y + i * line, part, size, color=right_color)
            self.y += height
            if rules:
                self._line(MARGIN + indent, PAGE_W - MARGIN, self.y - 0.04)
        self.y += 0.08


def render_guide(path: str, content: GuideContent) -> None:
    """Render an assembled snapshot without reading the result tables or directories."""
    pages = Pages(path, content.footer)
    try:
        widths = (2.05, 1.7, 3.02)
        pages.title(
            content.title,
            [(name, value, width) for (name, value), width in zip(content.facts, widths, strict=True)],
        )
        pages.paragraph(content.summary, 9.5)
        pages.heading("Folder and file structure")
        pages.rows(
            content.structure, left_width=3.7, size=7.5, mono_left=True, rules=False, right_color=MUTED
        )
        pages.heading("Which file for what")
        pages.rows(content.file_uses)
        pages.heading(content.highlights_title)
        if content.swatches:
            colours = {"Grey": "#D9D9D9", "Orange": "#FFE0B2"}
            pages.swatches([(colours[name], name, text) for name, text in content.swatches])
            pages.paragraph(content.highlights_note, 8.6, MUTED)
        else:
            pages.paragraph(content.highlights_note)
        pages.heading(content.columns_title)
        pages.paragraph(content.columns_intro)
        if content.headers:
            pages.rows(content.headers, left_width=2.3, size=8.4, mono_left=True)
        pages.heading("How the numbers are calculated")
        pages.paragraph(content.formulas_intro, 8.6, MUTED)
        heights = {"Depth": 0.2, "Zeroes": 0.42, "norm": 0.58, "clr": 0.42, "Ratios": 0.45}
        pages.formulas(
            [(name, text, formula, heights.get(name, 0)) for name, text, formula in content.formulas]
        )
    finally:
        pages.close()


def write_readme(root: str, label: str, info: RunInfo) -> list[tuple[str, str, int]]:
    """Build and write the results guide, returning its folder-tree rows."""
    content = build_guide(root, label, info)
    render_guide(os.path.join(root, NAME), content)
    return content.structure
