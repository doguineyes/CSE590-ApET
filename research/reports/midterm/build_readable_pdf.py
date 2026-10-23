"""Typeset the readable Markdown report with ReportLab, without model packages.

PDF-only dependencies: reportlab==4.4.3 and Pillow. For this preparation they
live in /tmp/apet-midterm-pdf-deps; the research/editor environments are unchanged.
Run from the repository root:
  PYTHONPATH=/tmp/apet-midterm-pdf-deps environments/editor/.venv/bin/python \
    research/reports/midterm/build_readable_pdf.py

The small SVG subset renderer keeps the diagram vector-based in the PDF.
It handles only the shapes used by our local diagram, not arbitrary SVG files.
"""

from __future__ import annotations

import html
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "midterm_readable_v2.md"
OUTPUT = ROOT / "output/pdf/midterm_readable_v2.pdf"
COPY = HERE / OUTPUT.name
PAGE_W, PAGE_H = letter
CONTENT_W = PAGE_W - 90
FONT_DIR = Path("/System/Library/Fonts/Supplemental")


def register_fonts():
    files = {
        "ReportSerif": "Times New Roman.ttf",
        "ReportSerif-Bold": "Times New Roman Bold.ttf",
        "ReportSerif-Italic": "Times New Roman Italic.ttf",
        "ReportSerif-BoldItalic": "Times New Roman Bold Italic.ttf",
        "ReportSans": "Arial.ttf",
        "ReportSans-Bold": "Arial Bold.ttf",
    }
    for name, filename in files.items():
        path = FONT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Font missing: {path}; configure FONT_DIR for this machine")
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        "ReportSerif", normal="ReportSerif", bold="ReportSerif-Bold",
        italic="ReportSerif-Italic", boldItalic="ReportSerif-BoldItalic")
    pdfmetrics.registerFontFamily(
        "ReportSans", normal="ReportSans", bold="ReportSans-Bold",
        italic="ReportSans", boldItalic="ReportSans-Bold")


def markup(value):
    value = html.escape(value, quote=False)
    value = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
                   lambda m: f'<link href="{m[2]}" color="#244e70">{m[1]}</link>', value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    return value


class VectorSVG(Flowable):
    """Render this project's simple SVG in PDF coordinates using local fonts."""

    def __init__(self, path, width):
        super().__init__()
        self.svg = ET.parse(path).getroot()
        self.source_w = float(self.svg.get("width"))
        self.source_h = float(self.svg.get("height"))
        self.width = width
        self.height = width * self.source_h / self.source_w
        self.hAlign = "CENTER"
        self.keepWithNext = True
        self.spaceBefore = 3
        self.spaceAfter = 4
        self.css = {}
        for el in self.svg.iter():
            if el.tag.endswith("style"):
                for selector, body in re.findall(r"([.\w-]+)\s*\{([^}]+)\}", el.text or ""):
                    self.css[selector] = dict(
                        (key.strip(), val.strip()) for key, val in
                        (entry.split(":", 1) for entry in body.split(";") if ":" in entry))

    def draw(self):
        c = self.canv
        c.saveState()
        scale = self.width / self.source_w
        c.translate(0, self.height)
        c.scale(scale, -scale)
        for el in self.svg:
            tag = el.tag.rsplit("}", 1)[-1]
            if tag not in ("rect", "line", "polygon", "path", "text"):
                continue
            props = dict(self.css.get(tag, {}))
            props.update(el.attrib)
            for cls in el.get("class", "").split():
                props.update(self.css.get("." + cls, {}))
            fill = props.get("fill", "#172534" if tag == "text" else "none")
            stroke = props.get("stroke", "none")
            c.setFillColor(colors.HexColor(fill) if fill not in ("none", "white") else colors.white)
            c.setStrokeColor(colors.HexColor(stroke) if stroke != "none" else colors.white)
            c.setLineWidth(float(props.get("stroke-width", "1")))
            do_fill, do_stroke = int(fill != "none"), int(stroke != "none")
            if tag == "rect":
                c.roundRect(float(el.get("x", 0)), float(el.get("y", 0)),
                            float(el.get("width")), float(el.get("height")),
                            float(el.get("rx", 0)), fill=do_fill, stroke=do_stroke)
            elif tag == "line":
                c.line(*[float(el.get(k)) for k in ("x1", "y1", "x2", "y2")])
            elif tag == "polygon":
                values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", el.get("points", ""))]
                path = c.beginPath()
                path.moveTo(values[0], values[1])
                for i in range(2, len(values), 2):
                    path.lineTo(values[i], values[i + 1])
                path.close()
                c.drawPath(path, fill=do_fill, stroke=do_stroke)
            elif tag == "path":
                tokens = re.findall(r"[MLHVZ]|-?\d+(?:\.\d+)?", el.get("d", ""))
                path = c.beginPath()
                x = y = 0.0
                i = 0
                while i < len(tokens):
                    command = tokens[i]
                    i += 1
                    if command in ("M", "L"):
                        x, y = float(tokens[i]), float(tokens[i + 1]); i += 2
                        (path.moveTo if command == "M" else path.lineTo)(x, y)
                    elif command == "H":
                        x = float(tokens[i]); i += 1; path.lineTo(x, y)
                    elif command == "V":
                        y = float(tokens[i]); i += 1; path.lineTo(x, y)
                    elif command == "Z":
                        path.close()
                    else:
                        raise ValueError(f"Unsupported diagram path command: {command}")
                c.drawPath(path, fill=do_fill, stroke=do_stroke)
            else:
                x, y = float(el.get("x", 0)), float(el.get("y", 0))
                if "transform" in el.attrib:
                    match = re.fullmatch(r"translate\((-?[\d.]+),\s*(-?[\d.]+)\)", el.get("transform"))
                    if not match:
                        raise ValueError("Unsupported diagram transform")
                    x += float(match[1]); y += float(match[2])
                font = "ReportSans-Bold" if props.get("font-weight") == "700" else "ReportSans"
                size = float(props.get("font-size", "18px").removesuffix("px"))
                c.saveState()
                c.translate(x, y)
                c.scale(1, -1)
                c.setFont(font, size)
                anchor = props.get("text-anchor", "start")
                draw = {"middle": c.drawCentredString, "end": c.drawRightString}.get(anchor, c.drawString)
                draw(0, 0, "".join(el.itertext()))
                c.restoreState()
        c.restoreState()


def styles():
    base = dict(fontName="ReportSerif", fontSize=11, leading=13.7,
                spaceAfter=4, textColor=colors.HexColor("#17212b"),
                allowWidows=0, allowOrphans=0)
    result = {"body": ParagraphStyle("Body", **base)}
    for key, extras in {
        "title": dict(fontName="ReportSerif-Bold", fontSize=17, leading=19.3,
                      spaceAfter=7, alignment=TA_LEFT),
        "meta": dict(fontName="ReportSans", fontSize=9.2, leading=12, spaceAfter=4),
        "h2": dict(fontName="ReportSerif-Bold", fontSize=12, leading=14.5,
                   spaceBefore=8, spaceAfter=4, keepWithNext=True),
        "h3": dict(fontName="ReportSerif-Bold", fontSize=11.2, leading=13.5,
                   spaceBefore=6, spaceAfter=3, keepWithNext=True),
        "caption": dict(fontSize=9.3, leading=11.5, spaceAfter=5),
        "bullet": dict(leftIndent=10, firstLineIndent=0, bulletIndent=0, spaceAfter=3),
        "ref": dict(fontSize=9.1, leading=11.1, leftIndent=15, firstLineIndent=-15,
                    spaceAfter=3),
        "cell": dict(fontSize=9.3, leading=11.2, spaceAfter=0),
        "cellhead": dict(fontName="ReportSerif-Bold", fontSize=9.1, leading=11.1,
                         spaceAfter=0),
    }.items():
        result[key] = ParagraphStyle(key, **(base | extras))
    return result


def make_table(lines, st):
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-+:?", v) for v in r)]
    n = len(rows[0])
    if n == 6:
        widths = [80, 52, 48, 48, 119, 175]
        widths = [w * CONTENT_W / sum(widths) for w in widths]
    elif n == 4 and rows[0][0] == "Cohort":
        widths = [115, 120, 130, 157]
    elif n == 4:
        widths = [95, 85, 183, 159]
    elif n == 2:
        widths = [102, CONTENT_W - 102]
    else:
        widths = [CONTENT_W / n] * n
    cells = [[Paragraph(markup(cell), st["cellhead" if i == 0 else "cell"])
              for cell in row] for i, row in enumerate(rows)]
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf1f4")),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.HexColor("#4b5966")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#9da9b4")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#4b5966")),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#d8dfe5")),
    ]))
    table.spaceAfter = 7
    return table


def story_from_markdown(text, st):
    lines = text.splitlines()
    story = []
    i = 0
    in_references = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line.startswith("# "):
            story.append(Paragraph(markup(line[2:]), st["title"]))
        elif line.startswith("## "):
            if line.startswith("## 5."):
                story.append(PageBreak())
            in_references = line.startswith("## 7.")
            story.append(Paragraph(markup(line[3:]), st["h2"]))
        elif line.startswith("### "):
            story.append(Paragraph(markup(line[4:]), st["h3"]))
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[.*\]\(([^)]+)\)", line)
            story.append(VectorSVG(HERE / match[1], CONTENT_W))
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            story.append(make_table(block, st)); continue
        elif line.startswith("- "):
            story.append(Paragraph(markup(line[2:]), st["bullet"], bulletText="•"))
        elif in_references and re.match(r"\d+\. ", line):
            number, body = line.split(". ", 1)
            story.append(Paragraph(f"[{number}] " + markup(body), st["ref"]))
        else:
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||- |!\[)", lines[i]):
                paragraph.append(lines[i].strip()); i += 1
            value = " ".join(paragraph)
            kind = "caption" if value.startswith(("**Table", "**Figure")) else "body"
            if value.startswith(("**Midterm", "**Peng")):
                kind = "meta"
            p = Paragraph(markup(value), st[kind])
            if value.startswith("**Table"):
                p.keepWithNext = True
            story.append(p); continue
        i += 1
    return story


def page_decoration(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#536170"))
    canvas.setFont("ReportSans", 8)
    if doc.page > 1:
        canvas.drawString(45, PAGE_H - 27, "Peng Chen | Reconstruction-Based Visual Token Compression")
        canvas.setStrokeColor(colors.HexColor("#cbd3db"))
        canvas.setLineWidth(0.4)
        canvas.line(45, PAGE_H - 33, PAGE_W - 45, PAGE_H - 33)
    canvas.drawString(45, 23, "CSE590-01 | Midterm Progress Report")
    canvas.drawRightString(PAGE_W - 45, 23, str(doc.page))
    canvas.restoreState()


def main():
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter, rightMargin=45, leftMargin=45,
        topMargin=43, bottomMargin=40,
        title="Understanding Reconstruction-Based Visual Token Compression: When and Why Does ApET Work?",
        author="Peng Chen", subject="CSE590-01 Midterm Progress Report",
        pageCompression=1)
    document.build(story_from_markdown(SOURCE.read_text(), styles()),
                   onFirstPage=page_decoration, onLaterPages=page_decoration)
    shutil.copyfile(OUTPUT, COPY)
    print(OUTPUT.relative_to(ROOT))
    print(COPY.relative_to(ROOT))


if __name__ == "__main__":
    main()
