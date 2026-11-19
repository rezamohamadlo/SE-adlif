"""Export the Persian research report as editable Word text, tables and figures.

Presentation only; reads Markdown and saved images without scientific execution.
Optional export dependencies: python-docx and Markdown in the project .venv.
Run from the repository root with python -m scripts.export_research_report_docx.
"""
import argparse
import json
import re
import xml.etree.ElementTree as ET

import markdown
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION_START
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from project_paths import local_path, portable_path


PERSIAN = re.compile(r'[\u0600-\u06ff]')
LATIN = re.compile(r'([A-Za-z0-9][A-Za-z0-9_.*+/\-=(),:\[\]]*(?:[ \u00a0]+[A-Za-z0-9_.*+/\-=(),:\[\]±%]+)*)')


def prop(parent, name, **attrs):
    element = parent.find(qn('w:' + name))
    if element is None:
        element = OxmlElement('w:' + name)
        parent.append(element)
    for key, value in attrs.items():
        element.set(qn('w:' + key), str(value))
    return element


def direction(paragraph, rtl=True, align=None):
    prop(paragraph._p.get_or_add_pPr(), 'bidi', val='1' if rtl else '0')
    paragraph.alignment = align if align is not None else (
        WD_ALIGN_PARAGRAPH.RIGHT if rtl else WD_ALIGN_PARAGRAPH.LEFT)


def run_text(paragraph, value, bold=False, italic=False, size=None, color=None):
    for chunk in LATIN.split(value):
        if not chunk:
            continue
        rtl = bool(PERSIAN.search(chunk))
        run = paragraph.add_run(chunk)
        run.bold, run.italic = bold, italic
        run.font.name = 'B Nazanin' if rtl else 'Times New Roman'
        if size:
            run.font.size = Pt(size if rtl else max(9, size - 1))
        if color:
            run.font.color.rgb = RGBColor.from_string(color)
        rpr = run._r.get_or_add_rPr()
        prop(rpr, 'rFonts', ascii='Times New Roman', hAnsi='Times New Roman', cs='B Nazanin')
        prop(rpr, 'rtl', val='1' if rtl else '0')
        prop(rpr, 'lang', val='en-US', bidi='fa-IR')
        prop(rpr, 'bCs', val='1' if bold else '0')
        prop(rpr, 'iCs', val='1' if italic else '0')
        if size:
            prop(rpr, 'szCs', val=str(int(size * 2)))


def inline(paragraph, element, bold=False, italic=False, size=None):
    if element.text:
        run_text(paragraph, element.text, bold, italic, size)
    for child in element:
        if child.tag == 'br':
            paragraph.add_run().add_break()
        else:
            # Local source references retain their descriptive label, as in the PDF.
            # Repository paths are not usable links in a document sent to a recipient.
            inline(paragraph, child, bold or child.tag in ('strong', 'b'),
                   italic or child.tag in ('em', 'i'), size)
        if child.tail:
            run_text(paragraph, child.tail, bold, italic, size)


def configure_section(section, landscape=False):
    section.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
    section.page_width, section.page_height = (
        (Cm(29.7), Cm(21)) if landscape else (Cm(21), Cm(29.7)))
    section.top_margin, section.bottom_margin = Cm(1.6), Cm(1.7)
    section.left_margin = section.right_margin = Cm(1.6)
    section.header_distance = section.footer_distance = Cm(.7)


def page_field(paragraph, name):
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), name)
    paragraph._p.append(field)


def setup(document):
    configure_section(document.sections[0])
    for name in ('Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Heading 3', 'Caption'):
        style = document.styles[name]
        style.font.name = 'B Nazanin'
        style.font.size = Pt(14)
        rpr = style.element.get_or_add_rPr()
        prop(rpr, 'rFonts', ascii='Times New Roman', hAnsi='Times New Roman', cs='B Nazanin')
        prop(rpr, 'szCs', val='28')
        prop(rpr, 'lang', val='fa-IR', bidi='fa-IR')
        prop(style.element.get_or_add_pPr(), 'bidi', val='1')
        style.paragraph_format.space_after = Pt(7)
        style.paragraph_format.line_spacing = 1.2
        style.paragraph_format.widow_control = True
    for name, size in [('Title', 24), ('Heading 1', 19), ('Heading 2', 16), ('Heading 3', 15)]:
        style = document.styles[name]
        style.font.size, style.font.bold = Pt(size), True
        style.font.color.rgb = RGBColor.from_string('184F68')
        prop(style.element.get_or_add_rPr(), 'szCs', val=str(size * 2))
        prop(style.element.get_or_add_rPr(), 'bCs', val='1')
        style.paragraph_format.keep_with_next = True
    document.styles['Caption'].font.size = Pt(12)
    prop(document.styles['Caption'].element.get_or_add_rPr(), 'szCs', val='24')
    footer = document.sections[0].footer.paragraphs[0]
    direction(footer, False, WD_ALIGN_PARAGRAPH.CENTER)
    page_field(footer, 'PAGE')
    run_text(footer, ' / ', size=10)
    page_field(footer, 'NUMPAGES')
    document.core_properties.title = 'گزارش پژوهش: ارتباط آمار زمانی ورودی با عملکرد سازوکارهای تطبیقی'
    document.core_properties.subject = 'نتایج و تحلیل شبکه‌های عصبی اسپایکی'
    document.core_properties.author = ''
    document.core_properties.last_modified_by = ''
    document.core_properties.comments = ''


def table(document, element):
    rows = element.findall('.//tr')
    cols = len(rows[0])
    result = document.add_table(rows=0, cols=cols)
    result.style = 'Table Grid'
    result.autofit = False
    prop(result._tbl.tblPr, 'bidiVisual', val='1')
    width = document.sections[-1].page_width - document.sections[-1].left_margin - document.sections[-1].right_margin
    # Keep dataset/model columns compact and give numerical estimates room.
    weights = [1.0] * cols
    if cols >= 4:
        weights[0] = .55
        weights[1] = .95
        if cols == 6:
            weights = [.5, .95, 1.15, 1.55, 1.65, 1.55]
        elif cols == 7:
            weights = [.5, .8, 1.25, 1.25, 1.25, 1.25, 1.5]
        elif cols == 4:
            weights = [.55, .95, 1.65, 2.25]
    widths = [int(width * weight / sum(weights)) for weight in weights]
    for column, cell_width in zip(result.columns, widths):
        column.width = cell_width
    for index, source_row in enumerate(rows):
        row = result.add_row()
        prop(row._tr.get_or_add_trPr(), 'cantSplit')
        if index == 0:
            prop(row._tr.get_or_add_trPr(), 'tblHeader')
        for cell, source_cell, cell_width in zip(row.cells, source_row, widths):
            cell.width = cell_width
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            paragraph = cell.paragraphs[0]
            text = ''.join(source_cell.itertext())
            direction(paragraph, bool(PERSIAN.search(text)), WD_ALIGN_PARAGRAPH.CENTER)
            paragraph.paragraph_format.space_after = Pt(3)
            paragraph.paragraph_format.space_before = Pt(3)
            paragraph.paragraph_format.line_spacing = 1.05
            # Start a table with its header and at least two data rows together.
            paragraph.paragraph_format.keep_with_next = index < min(2, len(rows) - 1)
            inline(paragraph, source_cell, bold=index == 0, size=11.5)
            if index == 0 or index % 2 == 0:
                prop(cell._tc.get_or_add_tcPr(), 'shd', fill='E4EEF3' if index == 0 else 'F5F8FA')
    return result


def export(source, output):
    text = source.read_text(encoding='utf-8')
    text = re.sub(r'^\s*</?div\b[^>]*>\s*$', '', text, flags=re.MULTILINE)
    markup = markdown.markdown(text, extensions=['tables', 'sane_lists'])
    root = ET.fromstring('<body>' + markup + '</body>')
    document = Document()
    setup(document)
    images, table_count = [], 0
    for element in root:
        if element.tag == 'table':
            cols = len(element.find('.//tr'))
            wide = cols >= 5 or (cols == 4 and any(len(''.join(cell.itertext())) > 65 for cell in element.find('.//tr')))
            if wide:
                configure_section(document.add_section(WD_SECTION_START.NEW_PAGE), True)
                previous = document.paragraphs[-2] if len(document.paragraphs) >= 2 else None
                if previous is not None and previous.style.name.startswith('Heading'):
                    # Keep a heading that immediately introduces a wide table with that table.
                    document._body._body.remove(previous._p)
                    document._body._body.insert(-1, previous._p)
            table(document, element)
            table_count += 1
            if wide:
                configure_section(document.add_section(WD_SECTION_START.NEW_PAGE))
        elif element.tag in ('h1', 'h2', 'h3', 'h4'):
            level = int(element.tag[1])
            paragraph = document.add_paragraph(style='Title' if level == 1 else f'Heading {level - 1}')
            direction(paragraph)
            inline(paragraph, element, bold=True)
        elif element.tag in ('ul', 'ol'):
            for index, item in enumerate(element.findall('li'), 1):
                paragraph = document.add_paragraph()
                direction(paragraph)
                paragraph.paragraph_format.right_indent = Cm(.4)
                run_text(paragraph, f'{index}. ' if element.tag == 'ol' else '• ')
                inline(paragraph, item)
        elif element.tag == 'p':
            image_element = element.find('img')
            if image_element is not None:
                path = local_path(source.parent / image_element.attrib['src'], must_exist=True)
                paragraph = document.add_paragraph()
                direction(paragraph, False, WD_ALIGN_PARAGRAPH.CENTER)
                paragraph.paragraph_format.keep_with_next = True
                shape = paragraph.add_run().add_picture(str(path), width=Cm(17.8))
                if shape.height > Cm(21):
                    factor = Cm(21) / shape.height
                    shape.width, shape.height = int(shape.width * factor), Cm(21)
                caption = image_element.attrib.get('alt', '')
                shape._inline.docPr.set('descr', caption)
                paragraph = document.add_paragraph(style='Caption')
                direction(paragraph, True, WD_ALIGN_PARAGRAPH.CENTER)
                run_text(paragraph, caption, size=12)
                images.append(portable_path(path))
            else:
                paragraph = document.add_paragraph()
                direction(paragraph, bool(PERSIAN.search(''.join(element.itertext()))))
                inline(paragraph, element)
        else:
            raise ValueError('Unsupported report block: ' + element.tag)
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)
    print(json.dumps(dict(output=portable_path(output), tables=table_count, figures=len(images),
                          sections=len(document.sections), scientific_execution=False), ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='research-report-fa.md')
    parser.add_argument('--output', default='research-report-fa.docx')
    args = parser.parse_args()
    export(local_path(args.source, must_exist=True), local_path(args.output))


if __name__ == '__main__':
    main()
