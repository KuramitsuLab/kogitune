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


def clean_code(text):
    stared = '<gh_stars>10' in text
    text = replace_url(text, max_allowed_num=0)
    text = replace_bar(text, code=True)
    text = replace_uuid(text)
    text = replace_float(text)
    text = replace_longstring(text)
    code = text

    docs=[]
    code, placeholders = replace_doc_string_with_placeholders(code, docs)
    code = remove_comment(code, docs)
    code = restore_placeholders(code, placeholders)
    doc = '\n'.join(docs)
    score_doc = len(doc)/len(code) if len(code) > 0 else 0
    score_en = score_english(doc, strict=True)
    score_ja = score_japanese(doc, strict=True)

    return {
        'doc_fraction': round(score_doc, 4),
        'score_en': round(score_en, 4),
        'score_ja': round(score_ja, 4),
        'stared': stared,
        'text': code,
        'text_length': len(code),
    }

def describe(filename, N=10000):
    import pandas as pd
    stat={
        'doc_fraction': [],
        'score_en': [],
        'score_ja': [],
        'text_length': [],
    }
    for line in filelines(filename, N=10000):
        data = clean_code(line.replace('<nL>', '\n'))
        for key, keylist in stat.items():
            keylist.append(data[key])
    df = pd.DataFrame(stat)
    print(df.describe(percentiles=[0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 0.9]))

def between(data, key, kwargs):
    minval = kwargs.get(f'min_{key}', None)
    maxval = kwargs.get(f'max_{key}', None)
    val = data[key]
    if (minval and val < minval) or (maxval and val > maxval):
        return False
    return True

def filter_code(filename, output_file, N=10000, **kwargs):
    import json
    describe(filename)
    with zopen(output_file, 'tw') as w:
        for line in filelines(filename, N=N):
            data = clean_code(line.replace('<nL>', '\n'))
            if between(data, 'text_length', kwargs) and between(data, 'doc_fraction', kwargs):
                if between(data, 'score_en', kwargs) or between(data, 'score_ja', kwargs):
                    print(json.dumps(data, ensure_ascii=False), file=w)
