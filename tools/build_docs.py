"""Build the offline documentation page from repository Markdown (no dependencies).

Supports the Markdown subset used by these guides: headings, paragraphs, fences,
tables, links, inline code/emphasis, lists, and roadmap checkboxes.
"""
import html
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [('readme', ROOT / 'README.md', 'User guide'),
           ('environment', ROOT / 'docs/ENVIRONMENT.md', 'Environment'),
           ('commands', ROOT / 'docs/COMMANDS.md', 'Command reference'),
           ('datasets', ROOT / 'docs/DATASETS.md', 'Dataset policy'),
           ('checkpoints', ROOT / 'docs/CHECKPOINTS.md', 'Checkpoints & resume'),
           ('roadmap', ROOT / 'docs/CODEBASE_FIX_ROADMAP.md', 'Repair roadmap')]
SOURCE_IDS = {path.resolve(): key for key, path, _ in SOURCES}


def slug(value):
    return re.sub(r'[^a-z0-9_-]', '', value.lower().replace(' ', '-'))


def link_target(target, source):
    path, _, anchor = target.partition('#')
    if not path or not re.match(r'^[a-z]+:', path, re.I):
        resolved = (source.parent / path).resolve() if path else source.resolve()
        if resolved in SOURCE_IDS:
            prefix = SOURCE_IDS[resolved]
            return '#' + prefix + ('-' + slug(anchor) if anchor else '')
        return str(Path('..') / resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else '#'
    return target if target.startswith(('https://', 'http://', 'mailto:')) else '#'


def inline(value, source):
    pattern = re.compile(r'`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|\*\*([^*]+)\*\*')
    output, offset = [], 0
    for match in pattern.finditer(value):
        output.append(html.escape(value[offset:match.start()]))
        if match[1] is not None:
            output.append('<code>' + html.escape(match[1]) + '</code>')
        elif match[2] is not None:
            output.append('<a href="' + html.escape(link_target(match[3], source), quote=True) + '">' + inline(match[2], source) + '</a>')
        else:
            output.append('<strong>' + html.escape(match[4]) + '</strong>')
        offset = match.end()
    output.append(html.escape(value[offset:]))
    return ''.join(output)


def render(source, key, label):
    lines = source.read_text(encoding='utf-8-sig').splitlines()
    body, navigation = [], []
    open_section = False
    index = 0
    identifiers = {}
    def unique_id(base):
        count = identifiers.get(base, 0)
        identifiers[base] = count + 1
        return base + (f'-{count}' if count else '')
    def section(identifier, title, level):
        nonlocal open_section
        if open_section: body.append('</section>')
        body.append(f'<section class="doc-section" id="{identifier}" data-guide="{html.escape(label)}">')
        body.append(f'<p class="eyebrow">{html.escape(label)}</p><h{level}>{inline(title, source)}</h{level}>')
        navigation.append((identifier, title))
        open_section = True
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        heading = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading:
            level, title = len(heading[1]), heading[2]
            if level <= 2:
                section(unique_id(key if level == 1 else key + '-' + slug(title)), title, 1 if level == 1 else 2)
            else:
                body.append(f'<h{level} id="{unique_id(key + "-" + slug(title))}">{inline(title, source)}</h{level}>')
            index += 1
            continue
        if line.startswith('```'):
            language = line[3:].strip() or 'text'
            code = []
            index += 1
            while index < len(lines) and not lines[index].startswith('```'):
                code.append(lines[index])
                index += 1
            body.append('<div class="code-block"><div class="code-toolbar"><span>' + html.escape(language) +
                        '</span><button type="button" class="copy" aria-label="Copy code block">Copy</button></div><pre><code>' +
                        html.escape('\n'.join(code)) + '</code></pre></div>')
            index += 1
            continue
        if line.startswith('|') and index + 1 < len(lines) and re.match(r'^\|[\s:|\-]+\|$', lines[index + 1]):
            def cells(row): return [cell.strip() for cell in row.strip().strip('|').split('|')]
            headers = cells(line)
            body.append('<div class="table-wrap" tabindex="0" aria-label="Scrollable reference table"><table><thead><tr>' +
                        ''.join('<th scope="col">' + inline(cell, source) + '</th>' for cell in headers) + '</tr></thead><tbody>')
            index += 2
            while index < len(lines) and lines[index].startswith('|'):
                body.append('<tr>' + ''.join('<td>' + inline(cell, source) + '</td>' for cell in cells(lines[index])) + '</tr>')
                index += 1
            body.append('</tbody></table></div>')
            continue
        listing = re.match(r'^\s*(-|\d+\.)\s+(.+)', line)
        if listing:
            ordered = listing[1] != '-'
            tag = 'ol' if ordered else 'ul'
            start = f' start="{listing[1][:-1]}"' if ordered else ''
            body.append(f'<{tag}{start}>')
            while index < len(lines):
                item = re.match(r'^\s*(-|\d+\.)\s+(.+)', lines[index])
                if not item or (item[1] != '-') != ordered: break
                content = item[2]
                index += 1
                while index < len(lines) and lines[index].startswith('  ') and lines[index].strip() and not re.match(r'^\s*(-|\d+\.)\s+', lines[index]):
                    content += ' ' + lines[index].strip()
                    index += 1
                check = re.match(r'^\[([ xX])\]\s+(.*)', content)
                if check:
                    done = check[1].lower() == 'x'
                    body.append('<li class="check-item"><span class="check ' + ('done' if done else 'pending') + '">' +
                                ('Done' if done else 'Pending') + '</span>' + inline(check[2], source) + '</li>')
                else:
                    body.append('<li>' + inline(content, source) + '</li>')
            body.append(f'</{tag}>')
            continue
        paragraph = [line.strip()]
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].startswith(('#', '```', '|')) and not re.match(r'^\s*(-|\d+\.)\s+', lines[index]):
            paragraph.append(lines[index].strip())
            index += 1
        body.append('<p>' + inline(' '.join(paragraph), source) + '</p>')
    if open_section: body.append('</section>')
    nav = '<details open><summary>' + html.escape(label) + '</summary>' + ''.join(
        f'<a href="#{identifier}" data-section="{identifier}">{html.escape(title)}</a>' for identifier, title in navigation) + '</details>'
    return '\n'.join(body), nav


def main():
    rendered = [render(path, key, label) for key, path, label in SOURCES]
    content = '\n'.join(body for body, _ in rendered)
    navigation = '\n'.join(nav for _, nav in rendered)
    css = (ROOT / 'docs/assets/guide.css').read_text(encoding='utf-8-sig')
    script = (ROOT / 'docs/assets/guide.js').read_text(encoding='utf-8-sig')
    page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Offline BreaKHis user and developer guide: setup, commands, parameters, checkpoints and resume.">
<title>BreaKHis · Codebase guide</title><style>''' + css + '''</style></head><body>
<a class="skip-link" href="#main">Skip to documentation</a>
<aside class="sidebar"><a class="brand" href="#readme"><span class="brand-mark">B</span><span>BreaKHis<small>CODEBASE GUIDE</small></span></a>
<p class="sidebar-note">Setup, experiments, and implementation</p>
<nav aria-label="Documentation navigation">''' + navigation + '''</nav>
<div class="sidebar-footer">Offline documentation<br>Generated from repository Markdown</div></aside>
<div class="workspace"><header class="topbar"><div class="search-wrap"><label for="search">Find a topic or parameter</label><div class="search-row"><input id="search" type="search" placeholder="Search --resume, CUDA, checkpoints…" autocomplete="off"><button id="clear" type="button">Clear</button></div></div><button id="print" type="button">Print guide</button></header>
<main id="main"><div class="intro"><p class="eyebrow">USER &amp; DEVELOPER DOCUMENTATION</p><h1>Your guide to the codebase.</h1><p>From a working environment to a reproducible binary classification experiment. Every command, argument, saved file, and current limitation in one place.</p><div class="badges"><span>Python 3.13</span><span>Binary workflow</span><span>Stages 1–6 implemented</span></div></div>
<div class="notice"><strong>Current workspace:</strong> use the saved manifest with <code>cancer_binary</code>. The v2 copy was removed; v1-only manifest rebuilding and multiclass training remain pending.</div>
<p id="search-status" role="status" aria-live="polite"></p><p id="no-results" hidden>No matching section. Try a command name, parameter, or error message.</p>
''' + content + '''
<footer class="page-footer">Edit README.md or docs/*.md, then run <code>python tools/build_docs.py</code> to refresh this guide. No network resources are used.</footer></main></div>
<script>''' + script + '''</script></body></html>'''
    output = ROOT / 'docs/index.html'
    output.write_text(page, encoding='utf-8')
    print(f'Built {output} from {len(SOURCES)} Markdown guides.')


if __name__ == '__main__':
    main()
