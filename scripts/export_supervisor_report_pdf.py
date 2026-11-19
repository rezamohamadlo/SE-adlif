"""Export the Persian supervisor report with embedded B Nazanin and Chromium RTL shaping.

Presentation only: reads the Markdown report and existing figures; no scientific execution.
Optional tools: Markdown, playwright, PyMuPDF==1.26.7 (inside the project .venv).
Fonts and browser are discovered locally, with explicit override flags for another machine.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile

import markdown
from playwright.sync_api import sync_playwright
import pymupdf

from project_paths import ROOT, local_path, portable_path


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_existing(paths, kind):
    for path in paths:
        if path and Path(path).is_file():
            return Path(path)
    raise FileNotFoundError('Provide the local ' + kind + ' path using the corresponding command option.')


def data_uri(path, mime):
    return 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


class PrintHTML(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.output = []
        self.assets = {}
        self.cell = None
        self.tag = None
        self.table_depth = 0

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'a':
            # Keep web links; local repository links must not leak a machine-specific file URI.
            href = values.get('href', '')
            if not re.match(r'^https?://', href):
                tag = 'span'
                values = {'class': 'local-reference'}
            self.tag = tag
        elif tag == 'img':
            path = local_path(self.source.parent / values['src'], must_exist=True)
            if not path.is_relative_to(ROOT):
                raise ValueError('Report image is outside the repository')
            self.assets[portable_path(path)] = checksum(path)
            mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
            values['src'] = data_uri(path, mime)
        elif tag == 'table':
            self.output.append('<div class="table-container">')
            self.table_depth += 1
            values['dir'] = 'rtl'
        elif tag in ('th', 'td'):
            self.cell = []
            self.cell_tag = tag
            return
        rendered = '<' + tag + ''.join(' ' + key + '="' + html.escape(value or '', quote=True) + '"'
                                     for key, value in values.items()) + '>'
        self.append(rendered)

    def handle_endtag(self, tag):
        if tag == 'a':
            tag = self.tag or tag
            self.tag = None
        if tag in ('th', 'td'):
            contents = ''.join(self.cell)
            plain = html.unescape(re.sub('<[^>]+>', '', contents))
            direction = 'rtl' if re.search(r'[\u0600-\u06ff]', plain) else 'ltr'
            self.cell = None
            self.output.append('<' + tag + ' dir="' + direction + '">' + contents + '</' + tag + '>')
        else:
            self.append('</' + tag + '>')
        if tag == 'table':
            self.output.append('</div>')
            self.table_depth -= 1

    def append(self, text):
        (self.output if self.cell is None else self.cell).append(text)

    def handle_data(self, data):
        # Isolate Latin model names so trailing '*' and hyphens retain their logical order in Persian prose.
        chunks = re.split(r'([A-Za-z][A-Za-z0-9_.*+/\-]*(?:[ \u00a0]+[A-Za-z0-9][A-Za-z0-9_.*+/\-]*)*)', data)
        self.append(''.join('<bdi dir="ltr">' + html.escape(chunk) + '</bdi>' if i % 2 else html.escape(chunk)
                            for i, chunk in enumerate(chunks)))

    def handle_entityref(self, name):
        self.append('&' + name + ';')

    def handle_charref(self, name):
        self.append('&#' + name + ';')


CSS = '''
@font-face {font-family:ReportNazanin;src:url(REGULAR);font-weight:400;font-display:block}
@font-face {font-family:ReportNazanin;src:url(BOLD);font-weight:700;font-display:block}
@page {size:A4 portrait;margin:17mm 15mm 19mm}
@page wide {size:A4 landscape;margin:16mm 15mm 19mm}
* {box-sizing:border-box}
html {direction:rtl; background:white}
body {font-family:ReportNazanin,serif;font-size:17px;line-height:1.7;color:#18252e;margin:0;
      text-align:right;font-synthesis:none}
h1,h2,h3 {font-weight:700;line-height:1.55;break-after:avoid;page-break-after:avoid}
h1 {font-size:29px;color:#163b50;border-bottom:2px solid #306882;padding-bottom:12px;margin-top:0}
h2 {font-size:23px;color:#184f68;margin-top:24px;margin-bottom:10px}
h3 {font-size:20px;color:#244b60;margin-top:16px;margin-bottom:8px}
p {margin:0 0 11px;orphans:3;widows:3}
li {margin-bottom:7px;orphans:3;widows:3}
ul,ol {padding-right:24px;padding-left:0}
strong {font-weight:700}
a,.local-reference {color:#245b75;text-decoration:none}
code {font-family:ReportNazanin,serif;direction:ltr;unicode-bidi:isolate;display:inline-block;
      background:#eef3f5;padding:0 3px;font-size:.93em}
table {width:100%;border-collapse:collapse;margin:10px 0 16px;table-layout:auto;font-size:14px;line-height:1.55}
thead {display:table-header-group}
th {background:#e4eef3;font-weight:700;text-align:center;padding:7px 5px;border:1px solid #b5c6d0}
td {border:1px solid #c5d0d7;padding:6px 5px;text-align:center;vertical-align:middle}
tr:nth-child(even) td {background:#f5f8fa}
tr {break-inside:avoid;page-break-inside:avoid}
td[dir=ltr] {unicode-bidi:isolate;white-space:nowrap;font-variant-numeric:tabular-nums}
td[dir=rtl],th {overflow-wrap:normal}
.wide {page:wide;font-size:16px;break-inside:avoid}
.wide table {font-size:14px;line-height:1.45}
.wide td,.wide th {padding:4px 5px}
.table-container {break-inside:auto}
.figure {break-inside:avoid;page-break-inside:avoid;text-align:center;margin:16px 0}
img {max-width:100%;max-height:212mm;object-fit:contain;display:block;margin:0 auto}
.figure-caption {font-size:14px;line-height:1.5;color:#435764;margin-top:7px;text-align:center}
.table-note {font-size:13px;color:#53656d;line-height:1.5}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='supervisor-guideline-report-fa.md')
    parser.add_argument('--output', default='supervisor-guideline-report-fa.pdf')
    parser.add_argument('--font-regular')
    parser.add_argument('--font-bold')
    parser.add_argument('--browser')
    args = parser.parse_args()
    source = local_path(args.source, must_exist=True)
    output = local_path(args.output)
    user_fonts = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData/Local'))) / 'Microsoft/Windows/Fonts'
    system_fonts = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    regular = first_existing([args.font_regular, user_fonts / 'BNazanin.ttf', system_fonts / 'BNazanin.ttf'], 'B Nazanin regular font')
    bold = first_existing([args.font_bold, user_fonts / 'BNaznnBd.ttf', system_fonts / 'BNaznnBd.ttf'], 'B Nazanin bold font')
    browser = first_existing([args.browser, 'C:/Program Files/Google/Chrome/Application/chrome.exe',
                              'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'], 'Chromium browser')
    original_sha = checksum(source)
    text = source.read_text(encoding='utf-8')
    # Markdown inside the author's raw direction wrappers must be parsed normally.
    text = re.sub(r'^\s*</?div\b[^>]*>\s*$', '', text, flags=re.MULTILINE)
    markup = markdown.markdown(text, extensions=['tables', 'sane_lists'])
    converted = PrintHTML(source)
    converted.feed(markup)
    css = CSS.replace('REGULAR', data_uri(regular, 'font/ttf')).replace('BOLD', data_uri(bold, 'font/ttf'))
    title = 'گزارش پژوهش بر اساس راهنمای استاد'
    document = '<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><title>' + title + \
        '</title><style>' + css + '</style></head><body>' + ''.join(converted.output) + '</body></html>'
    with sync_playwright() as driver:
        with driver.chromium.launch(executable_path=str(browser), headless=True) as chromium:
            page = chromium.new_page(viewport={'width': 900, 'height': 1100}, locale='fa-IR')
            page.set_content(document, wait_until='load')
            page.evaluate('''async () => {
                await document.fonts.ready;
                await document.fonts.load('17px ReportNazanin');
                await document.fonts.load('bold 17px ReportNazanin');
                await Promise.all([...document.images].map(im => im.decode()));
                document.querySelectorAll('table').forEach(table => {
                    if (table.rows[0].cells.length >= 7) {
                        const wrapper = table.parentElement;
                        wrapper.classList.add('wide');
                        const previous = wrapper.previousElementSibling;
                        if (previous && /^H[123]$/.test(previous.tagName)) wrapper.prepend(previous);
                    }
                });
                document.querySelectorAll('p > img').forEach(image => {
                    image.parentElement.classList.add('figure');
                    const caption = document.createElement('div');
                    caption.className = 'figure-caption'; caption.textContent = image.alt;
                    image.parentElement.appendChild(caption);
                });
            }''')
            fonts_loaded = page.evaluate("document.fonts.check('17px ReportNazanin') && document.fonts.check('bold 17px ReportNazanin')")
            if not fonts_loaded:
                raise RuntimeError('B Nazanin did not load; PDF export stopped')
            page.emulate_media(media='print')
            # Retain a self-contained HTML companion for reproducible visual inspection.
            html_document = page.content()
            with tempfile.TemporaryDirectory(prefix='persian_report_pdf_') as folder:
                raw_pdf = Path(folder) / 'raw.pdf'
                page.pdf(path=str(raw_pdf), prefer_css_page_size=True, print_background=True,
                         tagged=True, outline=True)
                pdf = pymupdf.open(raw_pdf)
                # Footer is Latin pagination; Persian body and headings use B Nazanin.
                for i, pdf_page in enumerate(pdf):
                    rect = pymupdf.Rect(35, pdf_page.rect.height - 35, pdf_page.rect.width - 35, pdf_page.rect.height - 19)
                    pdf_page.insert_textbox(rect, f'{i+1} / {len(pdf)}', fontsize=9,
                                            fontname='helv', align=pymupdf.TEXT_ALIGN_CENTER, color=(.35, .40, .43))
                pdf.set_metadata(dict(title=title, author='SE-adLIF research project',
                    subject='Persian supervisor-guideline report; RTL, embedded B Nazanin',
                    creator='Local Markdown presentation export'))
                pdf.save(output, garbage=4, deflate=True)
                pdf.close()
    if checksum(source) != original_sha:
        raise RuntimeError('Source report changed during export')
    evidence = ROOT / 'results/experiments/adaptation_stage8_synthesis_v1/persian_pdf_export_v1'
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'print.html').write_text(html_document, encoding='utf-8')
    with pymupdf.open(output) as pdf:
        names = sorted({font[3] for p in pdf for font in p.get_fonts()})
        if not any('BNazanin' in name for name in names):
            raise RuntimeError('B Nazanin embedding could not be confirmed')
        record = dict(exported_utc=datetime.now(timezone.utc).isoformat(), source=portable_path(source),
            source_sha256=original_sha, pdf=portable_path(output), pdf_sha256=checksum(output),
            pages=len(pdf), fonts=names, direction='rtl', fonts_embedded=True,
            font_files={regular.name: checksum(regular), bold.name: checksum(bold)},
            images=converted.assets, exporter=portable_path(Path(__file__)), exporter_sha256=checksum(Path(__file__)),
            packages={name: importlib.metadata.version(name) for name in ['Markdown', 'playwright', 'PyMuPDF']},
            scientific_execution=False)
    (evidence / 'EXPORT.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=portable_path(output), pages=record['pages'], fonts=names, bytes=output.stat().st_size), ensure_ascii=True))


if __name__ == '__main__':
    main()
