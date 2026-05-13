#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path


PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0
MARGIN_X = 48.0
MARGIN_TOP = 54.0
MARGIN_BOTTOM = 48.0
CONTENT_WIDTH = PAGE_WIDTH - (2 * MARGIN_X)


@dataclass(frozen=True)
class StyledLine:
    text: str
    font: str
    size: float
    leading: float


def _char_capacity(font_size: float, *, monospace: bool = False, bold: bool = False) -> int:
    width_factor = 0.60 if monospace else (0.56 if bold else 0.53)
    capacity = int(CONTENT_WIDTH / max(font_size * width_factor, 1.0))
    return max(capacity, 20)


def _wrap_text(text: str, *, font_size: float, monospace: bool = False, bold: bool = False) -> list[str]:
    text = text.rstrip()
    if not text:
        return [""]
    capacity = _char_capacity(font_size, monospace=monospace, bold=bold)
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= capacity:
            current = candidate
            continue
        if current:
            lines.append(current)
        while len(word) > capacity:
            lines.append(word[:capacity])
            word = word[capacity:]
        current = word
    if current or not lines:
        lines.append(current)
    return lines


def _markdown_to_lines(markdown: str) -> list[StyledLine]:
    styled: list[StyledLine] = []
    in_code = False
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            styled.append(StyledLine("", "Helvetica", 10.0, 8.0))
            continue
        if in_code:
            code_line = line if line else " "
            for wrapped in _wrap_text(code_line, font_size=9.5, monospace=True):
                styled.append(StyledLine(wrapped, "Courier", 9.5, 12.0))
            continue
        if not stripped:
            styled.append(StyledLine("", "Helvetica", 10.0, 8.0))
            continue
        if stripped.startswith("### "):
            text = stripped[4:]
            for wrapped in _wrap_text(text, font_size=12.0, bold=True):
                styled.append(StyledLine(wrapped, "Helvetica-Bold", 12.0, 16.0))
            styled.append(StyledLine("", "Helvetica", 10.0, 6.0))
            continue
        if stripped.startswith("## "):
            text = stripped[3:]
            for wrapped in _wrap_text(text, font_size=14.0, bold=True):
                styled.append(StyledLine(wrapped, "Helvetica-Bold", 14.0, 18.0))
            styled.append(StyledLine("", "Helvetica", 10.0, 6.0))
            continue
        if stripped.startswith("# "):
            text = stripped[2:]
            for wrapped in _wrap_text(text, font_size=18.0, bold=True):
                styled.append(StyledLine(wrapped, "Helvetica-Bold", 18.0, 24.0))
            styled.append(StyledLine("", "Helvetica", 10.0, 10.0))
            continue
        if stripped.startswith("- "):
            bullet = f"• {stripped[2:]}"
            for wrapped in _wrap_text(bullet, font_size=10.5):
                styled.append(StyledLine(wrapped, "Helvetica", 10.5, 14.0))
            continue
        if stripped[:2].isdigit() and stripped[2:4] == ". ":
            for wrapped in _wrap_text(stripped, font_size=10.5):
                styled.append(StyledLine(wrapped, "Helvetica", 10.5, 14.0))
            continue
        if stripped.startswith("```mermaid") or stripped.startswith("```"):
            continue
        for wrapped in _wrap_text(stripped, font_size=10.5):
            styled.append(StyledLine(wrapped, "Helvetica", 10.5, 14.0))
    return styled


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _paginate(lines: list[StyledLine]) -> list[list[StyledLine]]:
    pages: list[list[StyledLine]] = []
    current: list[StyledLine] = []
    y = PAGE_HEIGHT - MARGIN_TOP
    for line in lines:
        needed = line.leading if current else line.leading
        if y - needed < MARGIN_BOTTOM:
            pages.append(current)
            current = []
            y = PAGE_HEIGHT - MARGIN_TOP
        current.append(line)
        y -= line.leading
    if current:
        pages.append(current)
    return pages


def _build_page_stream(lines: list[StyledLine]) -> bytes:
    commands: list[str] = []
    y = PAGE_HEIGHT - MARGIN_TOP
    for line in lines:
        if not line.text:
            y -= line.leading
            continue
        font_ref = {"Helvetica": "F1", "Helvetica-Bold": "F2", "Courier": "F3"}[line.font]
        commands.append("BT")
        commands.append(f"/{font_ref} {line.size:.2f} Tf")
        commands.append(f"1 0 0 1 {MARGIN_X:.2f} {y:.2f} Tm")
        commands.append(f"({_escape_pdf_text(line.text)}) Tj")
        commands.append("ET")
        y -= line.leading
    return "\n".join(commands).encode("utf-8")


def _write_pdf(output_path: Path, pages: list[list[StyledLine]]) -> None:
    objects: list[bytes] = []

    def add_object(content: bytes) -> int:
        objects.append(content)
        return len(objects)

    font1 = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font2 = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    font3 = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    page_object_ids: list[int] = []
    content_object_ids: list[int] = []

    pages_placeholder = add_object(b"")
    for page_lines in pages:
        stream = _build_page_stream(page_lines)
        content_id = add_object(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        content_object_ids.append(content_id)
        page_id = add_object(
            (
                f"<< /Type /Page /Parent {pages_placeholder} 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH:.0f} {PAGE_HEIGHT:.0f}] "
                f"/Resources << /Font << /F1 {font1} 0 R /F2 {font2} 0 R /F3 {font3} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("utf-8")
        )
        page_object_ids.append(page_id)

    kids = " ".join(f"{obj_id} 0 R" for obj_id in page_object_ids)
    objects[pages_placeholder - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_ids)} >>".encode("utf-8")
    catalog = add_object(f"<< /Type /Catalog /Pages {pages_placeholder} 0 R >>".encode("utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for index, obj in enumerate(objects, start=1):
            offsets.append(handle.tell())
            handle.write(f"{index} 0 obj\n".encode("utf-8"))
            handle.write(obj)
            handle.write(b"\nendobj\n")
        xref_start = handle.tell()
        handle.write(f"xref\n0 {len(objects) + 1}\n".encode("utf-8"))
        handle.write(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            handle.write(f"{offset:010d} 00000 n \n".encode("utf-8"))
        handle.write(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\n"
                f"startxref\n{xref_start}\n%%EOF\n"
            ).encode("utf-8")
        )


def render_markdown_to_pdf(input_path: Path, output_path: Path) -> None:
    markdown = input_path.read_text(encoding="utf-8")
    lines = _markdown_to_lines(markdown)
    pages = _paginate(lines)
    _write_pdf(output_path, pages)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a Markdown file to a simple dependency-free PDF.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render_markdown_to_pdf(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
