import regex
import random

# 正規表現パターンをコンパイル
pattern = regex.compile(r'\[(?:[^\[\]]+|(?R))*\]')

def find_nested_brackets(s):
    return regex.findall(pattern, s)

# 使用例
def da(s, mapped=None):
    mapped = mapped or {}
    matches = find_nested_brackets(s)
    for match in matches:
        if '|' not in match and '/' not in match:
            continue
        if match in mapped:
            continue
        inner = match[1:-1]
        if '[' in inner:
            inner = da(inner, mapped)
            match = f"[{inner}]"
        if '/' in inner:
            tt = inner.split('/')
            random.shuffle(tt)
            inner = ''.join(tt)
        if '|' in inner:
            tt = inner.split('|')
            random.shuffle(tt)
            inner = tt[0]
        mapped[match]=inner
    for match, inner in mapped.items():
        s = s.replace(match, inner)
    return s

s = "This is [a simple|an] example, [SEP] but [here [is [a|an] nested] [simple|small] example] for you."
print(da(s))