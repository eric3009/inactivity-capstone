"""Small Markdown to Word converter used for the report (headings, paragraphs, lists, tables, images, call-outs, page numbers)."""
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

NAVY = RGBColor(0x23, 0x2B, 0x30)


def new_document(landscape=False, margins=2.2):
    d = Document()
    sec = d.sections[0]
    if landscape:
        sec.orientation = WD_ORIENT.LANDSCAPE
        sec.page_width, sec.page_height = sec.page_height, sec.page_width
    else:
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(margins)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    st = d.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.12
    for name, size in (("Heading 1", 18), ("Heading 2", 14), ("Heading 3", 12)):
        h = d.styles[name]
        h.font.name = "Calibri"
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = NAVY
        h.paragraph_format.space_before = Pt(14 if name != "Heading 3" else 10)
        h.paragraph_format.space_after = Pt(6)
    _page_number_footer(sec)
    return d


def _page_number_footer(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x5D, 0x67, 0x68)


def _runs(p, text, size=None, bold=False, color=None):
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`|_[^_]+_)", text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("`"):
            r = p.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt((size or 11) - 1)
            continue
        elif part.startswith("_") and part.endswith("_") and len(part) > 2:
            r = p.add_run(part[1:-1])
            r.italic = True
        else:
            r = p.add_run(part)
            r.bold = bold
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color


def _shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def _table(d, rows, font=9.5, widths=None):
    header, body = rows[0], rows[1:]
    t = d.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        _runs(c.paragraphs[0], h, size=font, bold=True)
        _shade(c, "E3E0D6")
    for row in body:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            _runs(cells[i].paragraphs[0], v, size=font)
            cells[i].paragraphs[0].paragraph_format.space_after = Pt(1)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)
    # keep header row repeating across pages
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)
    d.add_paragraph().paragraph_format.space_after = Pt(2)


def _callout(d, text):
    t = d.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.rows[0].cells[0]
    _shade(c, "EEECE5")
    c.text = ""
    _runs(c.paragraphs[0], text, size=10.5)
    d.add_paragraph().paragraph_format.space_after = Pt(2)


def add_markdown(d, md, base: Path, table_font=9.5, table_widths=None):
    lines = md.splitlines()
    i = 0
    pending_widths = None
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        mw = re.fullmatch(r"\{widths=([\d.,]+)\}", ln.strip())
        if mw:
            pending_widths = [float(x) for x in mw.group(1).split(",")]
            i += 1
            continue
        if ln.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                p = d.add_paragraph()
                r = p.add_run(lines[i])
                r.font.name = "Consolas"
                r.font.size = Pt(9.5)
                p.paragraph_format.space_after = Pt(0)
                i += 1
            i += 1
            d.add_paragraph().paragraph_format.space_after = Pt(2)
            continue
        if ln.strip() == r"\pagebreak":
            d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        elif ln.startswith("### "):
            d.add_heading(ln[4:], level=3)
        elif ln.startswith("## "):
            d.add_heading(ln[3:], level=2)
        elif ln.startswith("# "):
            d.add_heading(ln[2:], level=1)
        elif ln.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)(?:\{width=([\d.]+)\})?", ln)
            cap, path, w = m.group(1), m.group(2), float(m.group(3) or 16)
            d.add_picture(str(base / path), width=Cm(w))
            d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if cap:
                p = d.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                _runs(p, cap, size=9.5, color=RGBColor(0x5D, 0x67, 0x68))
        elif ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    block.append(cells)
                i += 1
            _table(d, block, font=table_font, widths=pending_widths)
            pending_widths = None
            continue
        elif ln.startswith("> "):
            _callout(d, ln[2:])
        elif ln.startswith("- "):
            p = d.add_paragraph(style="List Bullet")
            _runs(p, ln[2:])
            p.paragraph_format.space_after = Pt(2)
        elif re.match(r"\d+\. ", ln):
            p = d.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.9)
            p.paragraph_format.first_line_indent = Cm(-0.6)
            _runs(p, ln)
            p.paragraph_format.space_after = Pt(2)
        else:
            _runs(d.add_paragraph(), ln)
        i += 1
