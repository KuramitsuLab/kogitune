import regex
import random

# 正規表現パターンをコンパイル
pattern = regex.compile(r'\[(?:[^\[\]]+|(?R))*\]')

def find_nested_brackets(s):
    return regex.findall(pattern, s)

def contains_da_notion(text):
    matches = find_nested_brackets(s)
    return len(matches) > 0

# 使用例
def da(s, random_seed=None, random_choice=True, mapped=None):
    mapped = mapped or {}
    random_seed = random_seed or random.randint(0, 117) if random_choice else 0
    matches = find_nested_brackets(s)
    for match in matches:
        if '|' not in match and '/' not in match:
            continue
        if match in mapped:
            continue
        inner = match[1:-1]
        if '[' in inner:
            inner = da(inner, random_seed, random_choice, mapped)
            match = f"[{inner}]"
        if '/' in inner:
            tt = inner.split('/')
            if random_choice:
                random.shuffle(tt)
            inner = ''.join(tt)
        if '|' in inner:
            if inner.startswith('@'):
                tt = inner[1:].split('|')
                inner = tt[random_seed % len(tt)]
            else:
                tt = inner.split('|')
                if random_choice:
                    random.shuffle(tt)
                inner = tt[0]
        mapped[match]=inner
    for match, inner in mapped.items():
        s = s.replace(match, inner)
    return s

# s = "This is [a simple|an] example, [SEP] but [here [is [a|an] nested] [simple|small] example] for you."
# s = "This is [@(1)|1.|[1]] example, [SEP] but [here [is [a|an] nested] [@(2)|2.|[2]] example] for you."

# print(da(s))
