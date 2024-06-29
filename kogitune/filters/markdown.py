import re
import pandas as pd

from .utils_ja import score_japanese, score_english
from .replaces import RE, replace_pattern, replace_datetime, replace_url, replace_email, replace_uuid

code_pattern = RE(
    r'```[\s\S]+?```', 
    r'\$\$[\s\S]+?\$\$',
    flags=re.DOTALL | re.MULTILINE
)

inline_code_pattern = RE(
    r'`[^`]+?`'
#    r'\$[\s\S]+?\$',
)

def alnum_fraction(text):
    total = len(text)
    if total == 0:
        return 0.0, 0.0
    alnum = 0
    digit = 0
    for c in text:
        if c.isalnum():
            alnum += 1
            if c.isdigit():
                digit +=1
    return alnum/total, digit/alnum

def replace_code_blocks_with_placeholders(text, max_code_length=4096):
    placeholders = []
    code_blocks = code_pattern.findall(text)
    for i, block in enumerate(code_blocks, start=len(placeholders)):
        placeholder = f"<{i:002}pH>"
        if len(block) > max_code_length:
            # b=len(block)
            block = split_by_line(block, max_code_length // 2)
            # print('@', b, block)
        placeholders.append((placeholder, block))
        text = text.replace(block, placeholder)

    code_blocks = inline_code_pattern.findall(text)
    for i, block in enumerate(code_blocks, start=len(placeholders)):
        placeholder = f"<{i:002}pH>"
        placeholders.append((placeholder, block))
        text = text.replace(block, placeholder)

    return text, placeholders

### ```tsx filename=app/routes/posts/index.tsx lines=[1-2,4-17,20-21]
### import { json } from "@remix-run/node";

snipet_head_pattern = RE(
    r'(?P<prefix>```[\S]*)[^\n]*\n',
)

long_number_pattern = RE(
    r'(?P<prefix>\.\d{4})\d+',
)

def split_by_line(text, max_length, dots='...'):
    lines = text.split('\n')
    ss=[]
    length_count = 0
    for line in lines:
        length_count += len(line)
        if length_count > max_length:
            ss.append(dots)
            break
        ss.append(line)
    ss.extend(lines[-1:])
    return '\n'.join(ss)

def clean_code_snipet(text, max_length=512):
    text = replace_pattern(snipet_head_pattern, text, r"\g<prefix>\n")
    text = replace_url(text)
    text = replace_email(text)
    text = replace_uuid(text)
    text = replace_pattern(long_number_pattern, text, r"\g<prefix>")
    if len(text) > max_length:
        # b=len(block)
        text = split_by_line(text, max_length // 2)
        # print('@', b, block)
    return text

def restore_code_blocks(text, placeholders):
    for placeholder, block in placeholders:
        if placeholder in text:
            block = clean_code_snipet(block)
            text = text.replace(placeholder, block)
    return text

## markdown

yaml_pattern = RE(r'---\n[\s\S]*?\n---', flags= re.DOTALL | re.MULTILINE)

html_pattern = RE(
    r'\<[A-Za-z/][^>]*\>', 
    r'\<\![^>]*\>',
    r'\{\%[^%]+\%\}',
    r'~~[\s\S]+?~~',
#    r'\[!(?:[^\[\]]|(?R))*\]',  #[!NOTE]
)
table_pattern = RE(r'\n\s*\|[\s\S]*?\|\s*\n')

content_pattern = RE(
    r'\!?\[(?P<content>[^\[\]]*?)\]\([^\(\)]*?\)',  #image,link
    r'\[(?P<content>[^\[\]]*?)\]\[[^\[\]]*?\]',  #short link
#    r'[^\!]\[(?P<content>.*?)\]\[.*?\]',  #short link
    # r'[^\!]\[(?P<content>.*?)\]\(.*?\)',  #link
    r'\[\!(?P<content>.*?)\]',  #[!NOTE]
)

emph_pattern = RE(
    r'\*\*\*(?P<content>.*?)\*\*\*', #emph
    r'___(?P<content>.*?)___', #emph
)

bold_pattern = RE(
    r'\*\*(?P<content>.*?)\*\*', #bold
    r'__(?P<content>.*?)__', #bold
)

italic_pattern = RE(
    r'\*(?P<content>.*?)\*', #italic
    r'_(?P<content>.*?)_', #italic
)

def replace_emph(text):
    text = replace_pattern(emph_pattern, text, r'\g<content>')
    text = replace_pattern(bold_pattern, text, r'\g<content>')
    text = replace_pattern(italic_pattern, text, r'\g<content>')
    return text

def replace_markdown(text):
    text = replace_pattern(yaml_pattern, text, '')
    text = replace_pattern(html_pattern, text, '')    
    text = replace_pattern(table_pattern, text, '\n')
    text = replace_pattern(table_pattern, text, '\n') # 2回必要
    text = replace_pattern(content_pattern, text, r'\g<content>')
    text = replace_url(text)
    text = replace_email(text)
    text = replace_datetime(text)
    return text


def filter_markdown(text, data=None, min_score_ja = 0.0, min_score_en = 0.0, max_section_length=1024, max_code_length=512):
    text, placeholders = replace_code_blocks_with_placeholders(text, max_code_length)
    text = replace_markdown(text, emph=False)
    # text = replace_pattern(yaml_pattern, text, '')
    # text = replace_pattern(html_pattern, text, '')    
    # text = replace_url(text)
    sec='\n#'
    sections = text.split(sec)
    ss=[]
    if data is not None:
        data['score_ja'] = []
        data['score_en'] = []
    for text in sections:
        score_ja = score_japanese(text, strict=True)
        score_en = score_english(text, strict=True)
        if data is not None and len(data['score_ja']) < 10000:
            data['score_ja'].append(score_ja)
            data['score_en'].append(score_en)
        if len(sec) < max_section_length and min_score_ja <= score_ja and min_score_en <= score_en:
            text = replace_markdown(text, emph=True)
            text = restore_code_blocks(text, placeholders)
            ss.append(text)
        # else:
        #     print(f'@drop len={len(sec)}, ja={score_ja}, en={score_en}')
        #     print(text)
    return sec.join(ss)    

def describe(data=None):
    if data is not None:
        df = pd.DataFrame(data)
        print(df.describe(percentiles=[0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 0.9]))

text = '''
*   `NodeJS-Backport-Review` ([V8](https://bugs.chromium.org/p/v8/issues/list?can=1&q=label%3ANodeJS-Backport-Review), [Chromium](https://bugs.chromium.org/p/chromium/issues/list?can=1&q=label%3ANodeJS-Backport-Review)): to be reviewed if this is applicable to abandoned branches in use by Node.js. This list if regularly reviewed by the Node.js team at Google to determine applicability to Node.js.
*   `NodeJS-Backport-Approved` ([V8](https://bugs.chromium.org/p/v8/issues/list?can=1&q=label%3ANodeJS-Backport-Approved), [Chromium](https://bugs.chromium.org/p/chromium/issues/list?can=1&q=label%3ANodeJS-Backport-Approved)): marks bugs that are deemed relevant to Node.js and should be backported.
*   `NodeJS-Backport-Done` ([V8](https://bugs.chromium.org/p/v8/issues/list?can=1&q=label%3ANodeJS-Backport-Done), [Chromium](https://bugs.chromium.org/p/chromium/issues/list?can=1&q=label%3ANodeJS-Backport-Done)): Backport for Node.js has been performed already.
*   `NodeJS-Backport-Rejected🧑‍💻p16H🧑‍update-v8` script<sup>2</sup>. For example, if you want to replace the copy of V8 in Node.js with the branch-head for V8 5.1 branch:
'''

if __name__ == '__main__':
    print(replace_markdown(text))