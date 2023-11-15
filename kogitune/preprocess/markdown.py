import re
import json
import pyzstd
import html


code_pattern = re.compile(r'```[\s\S]+?```', re.DOTALL | re.MULTILINE)
inline_code_pattern = re.compile(r'`[^`]+?`')
math_pattern = re.compile(r'\$\$[\s\S]+?\$\$', re.DOTALL | re.MULTILINE)
inline_math_pattern = re.compile(r'\$[\s\S]+?\$')

def replace_code_blocks_with_placeholders(text):
    placeholders = []
    code_blocks = code_pattern.findall(text)
    for i, block in enumerate(code_blocks, start=len(placeholders)):
        placeholder = f"_p{i:002}H_"
        placeholders.append((placeholder, block))
        text = text.replace(block, placeholder)

    code_blocks = math_pattern.findall(text)
    for i, block in enumerate(code_blocks, start=len(placeholders)):
        placeholder = f"_p{i:002}H_"
        placeholders.append((placeholder, block))
        text = text.replace(block, placeholder)

    code_blocks = inline_code_pattern.findall(text)
    for i, block in enumerate(code_blocks, start=len(placeholders)):
        placeholder = f"_p{i:002}H_"
        placeholders.append((placeholder, block))
        text = text.replace(block, placeholder)

    code_blocks = inline_math_pattern.findall(text)
    for i, block in enumerate(code_blocks, start=len(placeholders)):
        placeholder = f"_p{i:002}H_"
        placeholders.append((placeholder, block))
        text = text.replace(block, placeholder)
    return text, placeholders

def restore_code_blocks(text, placeholders):
    for placeholder, block in placeholders:
        text = text.replace(placeholder, block)
    return text

yaml_pattern = re.compile(r'---\n[\s\S]*?\n---', re.DOTALL | re.MULTILINE)


base64_pattern = re.compile(r'^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$')
uuid_pattern = re.compile(r'\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b')
hash_pattern = re.compile(r'\b[a-fA-F0-9]{7,}\b')
url_pattern = re.compile(r'^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?(\?[^\s]*)?(#[^\s]*)?$')

include_pattern = re.compile(r'\[!INCLUDE \[.*?\]\(.*?\)\]')
image_pattern = re.compile(r'<img src ?= ?".*?" alt ?= ?".*?" width ?= ?".*?"/?>')
image2_pattern = re.compile(r'!\[.*?\]\(.*?\)')
link_pattern = re.compile(r'\[([^\]]*?)\]\([^\)]*?\)')
html_tag = re.compile(r'<.*?>')
# empty_a_pattern = re.compile(r'<a name="[^"]*?"></a>')
# empty_a_pattern2 = re.compile(r'<a id="[^"]*?"></a>')
# html_comment_pattern = re.compile(r'<!--.*?-->')
google_drive_pattern = re.compile(r'https://drive\.google\.com/file/d/[A-Za-z0-9\-]+')
back_slpash_pattern = re.compile(r'\\([_])')

bold_pattern = re.compile(r'\*\*(.*?)\*\*')
bold_pattern2 = re.compile(r'__(.*?)__')
italic_pattern = re.compile(r'\*(.*?)\*')
italic_pattern2 = re.compile(r'_(.*?)_')
strike_pattern = re.compile(r'~~.*?~~')
extra_newlines_pattern = re.compile(r'\n{3,}')



def filter_markdown(text):
    # Remove YAML headder
    text = re.sub(yaml_pattern, '', text)

    # エンコードされたHTMLエンティティの例
    # encoded_string = "This is an example of a non-breaking space: &nbsp; and an ampersand: &amp;"
    # HTMLエンティティをデコード
    text = html.unescape(text)

    text = re.sub(image_pattern, '(Figure)', text)
    text = re.sub(image2_pattern, '(Figure)', text)
    text = re.sub(link_pattern, r'\1', text)
    text = re.sub(include_pattern, '...', text)
    text = re.sub(html_tag, '', text)
    # text = re.sub(empty_a_pattern2, '', text)
    # text = re.sub(html_comment_pattern, '', text)
    text = re.sub(uuid_pattern, '(UUID)', text)
    text = re.sub(hash_pattern, '(hash)', text)
    text = re.sub(google_drive_pattern, '(google drive)', text)

    text = re.sub(back_slpash_pattern, r'\1', text)  #\_
    text = re.sub(bold_pattern, '\\1', text)  # Remove bold
    text = re.sub(italic_pattern, '\\1', text)      # Remove italic
    text = re.sub(strike_pattern, '', text)      # Remove strikethrough
    text = re.sub(extra_newlines_pattern, '\n\n', text)

#    text = restore_code_blocks(text, placeholders)
    return text

def score_markdown(text, sec="\n#"):
    text, placeholders = replace_code_blocks_with_placeholders(text)
    sections = text.split(sec)
    for text in sections:
        text = filter_markdown(text)
        score_en = score_en(text)
