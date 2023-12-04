from typing import List
import re
import ast
from kogitune.file_utils import filelines, zopen
from .words import *
from .replace import *

single_pattern = re.compile(r'\'\'\'[\s\S]*?\'\'\'', re.DOTALL | re.MULTILINE)
double_pattern = re.compile(r'"""[\s\S]*?"""', re.DOTALL | re.MULTILINE)

def replace_doc_string_with_placeholders(text, docs):
    placeholders = []
    code_blocks = double_pattern.findall(text)
    for i, block in enumerate(code_blocks):
        placeholder = f"_DOC{i:02}_"
        docs.append(block[3:-3])
        # print(f'--{len(block)}')
        # print(block)
        placeholders.append((placeholder, block))
        text = text.replace(block, placeholder)
    code_blocks = single_pattern.findall(text)
    for i, block in enumerate(code_blocks, start=len(placeholders)):
        placeholder = f"_DOC{i:02}_"
        docs.append(block[3:-3])
        placeholders.append((placeholder, block))
        text = text.replace(block, placeholder)
    return text, placeholders

def restore_placeholders(text, placeholders):
    for placeholder, block in placeholders:
        text = text.replace(placeholder, block)
    return text

def remove_common_prefix(lines):
    splitted = False
    if isinstance(lines, str):
        lines = lines.split('\n')
        splitted = True

    if len(lines) > 1:
        # 共通の接頭辞を見つける
        prefix = lines[0]
        for line in lines[1:]:
            while not line.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    break

        # 共通の接頭辞を各行から取り除く
        if prefix:
            lines = [line[len(prefix):] for line in lines]
    else:
        lines[0] = lines[0].lstrip().removeprefix('#')
    if splitted:
        return '\n'.join(lines)
    return lines
    
def is_commented_python(text):
    try:
        # 最初のインデントを揃える
        text = remove_common_prefix(text)
        ast.parse(text)
        return True
    except BaseException as e:
        # print('=', e)
        # print(text)
        return False

def remove_comment(text:str, docs: List[str]):
    lines=[]
    multi_indent=None
    multi_comments=[]
    body=False
    for line in text.split('\n'):
        if not body and (line.startswith('#') or line.startswith('<')):
            continue
        body=True
        code, s, comment = line.partition('#')
        if len(multi_comments) > 0 and multi_indent != code:
            doc = '\n'.join(multi_comments)
            if not is_commented_python(doc):
                lines.extend(f"{multi_indent}#{c}" for c in multi_comments)
                docs.append(doc)
                # print('--doc--')
                # print(doc)
            # else:
            #     print('--dropped--')
            #     print(doc)
            multi_comments=[]
            multi_indent=None
        if s == '':
            lines.append(line)
            multi_indent=None
            continue
        strip_code = code.strip()
        if len(strip_code) > 0:
            lines.append(line)
            if ('"' in strip_code and '"' in comment) or ("'" in strip_code and "'" in comment):
                pass
            else:
                docs.append(comment)
            multi_indent=None
            continue
        multi_indent=code
        multi_comments.append(comment)
    return '\n'.join(lines)

def extract_doc_string(code):
    docs=[]
    code, placeholders = replace_doc_string_with_placeholders(code, docs)
    code = remove_comment(code, docs)
    code = restore_placeholders(code, placeholders)
    doc = '\n'.join(docs)
    return doc, code

def doc_fraction(text):
    doc, code = extract_doc_string(text)
    return len(doc)/len(code) if len(code) > 0 else 0


base64_pattern = RE(
    r'(\b[0-9\+/]+[a-z]+[0-9\+/A-Z]+[a-z]+[0-9\+/A-Z]+[0-9a-zA-Z\+/]*={0,2}\b)',
    r'(\b[0-9\+/]+[A-Z]+[0-9\+/a-z]+[A-Z]+[0-9\+/a-z]+[0-9a-zA-Z\+/]*={0,2}\b)',
    r'(\b[0-9a-zA-Z+/]{4,}={1,2}\b)',
)

uuid_pattern = RE(r'\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b')

hash_pattern = RE(
    r'(\b[a-f]+[0-9]+[a-f]+[0-9]+[a-f0-9]{3,}\b)',
    r'(\b[0-9]+[a-f]+[0-9]+[a-f]+[a-f0-9]{3,}\b)',
    r'(\b[A-F]+[0-9]+[A-F]+[0-9]+[A-F0-9]{3,}\b)',
    r'(\b[0-9]+[A-F]+[0-9]+[A-F]+[A-F0-9]{3,}\b)',
)

def replace_uuid(text):
    return replace_pattern(uuid_pattern, text, '<uuid>')

def clean_starcode(text):
    text = replace_url(text, max_allowed_num=1)
    text = replace_uuid(text)
    text = replace_bar(text)
    text = replace_float(text)
    text = replace_longstring(text)
    return text.replace('<EMAIL>', '<email>').replace('<NAME>', '<name>').replace('<PASSWORD>', '<password>')

