from typing import Any

import re
import zlib
import math

def add_no_section(text):
    return text

LINE_PATTERN = re.compile(r'\n([^\n])')

def section_line(text):
    text = LINE_PATTERN.sub(r'\n<sectioN>\1', text)
    return text.split('<sectionN>')

DOC_PATTERN = re.compile(r'\n\n([^\n])')

def section_doc(text):
    text = DOC_PATTERN.sub(r'\n\n<sectioN>\1', text)
    return text.split('<sectionN>')

PYTHON_PATTERN = re.compile(r'\n(def|    def|class|async|    async|@|    @|\nif|\ntry|\n#|\n[A-Za-z0-9_]+\s=) ')

def section_python(code):
    text = PYTHON_PATTERN.sub(r'\n<sectioN>\1 ', code)
    return text.split('<sectionN>')

MARKDOWN_PATTERN = re.compile(r'\n(#)')

def section_markdown(text):
    text = MARKDOWN_PATTERN.sub(r'\n<sectioN>\1', text)
    return text.split('<sectionN>')

def find_section_fn(section):
    func = globals().get(f'section_{section}')
    if func is None:
        patterns = [s.replace('section_', '') for s in globals() if s.startswith('section_')]
        raise ValueError(f'section_{section} is not found. Select section from {patterns}')
    return func

