from .patterns import *

### コード用

float_pattern = RE(
    r'(?P<prefix>\d*[\.]\d{3})\d{2,}',
)

def replace_float(text, replaced=None):
    """
    text中の少数点数を短くする

    >>> replace_float("大さじ 1.8942041666667 ")
    '大さじ 1.894 '
    """
    text = replace_pattern(float_pattern, text, r'\g<prefix>')
    return text

string_pattern = RE(
    r'(?P<prefix>\"[\w/_\.\-]{8})[^\"\n\s]{16,}(?P<suffix>[\w\._/\-]{8}\")',
    r"(?P<prefix>\'[\w/_\.\-]{8})[^\'\n\s]{16,}(?P<suffix>[\w\._/\-]{8}\')",
)

def replace_longstring(text, replaced=None):
    """
    コード中の長い文字列をトリムする

    >>> replace_longstring("'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'")
    "'aaaaaaaa(...)aaaaaaaa'"

    >>> replace_longstring('"/usr/local/"')
    '"aaaaaaaa(...)aaaaaaaa"'
    """
    text = replace_pattern(string_pattern, text, r'\g<prefix>(...)\g<suffix>')
    return text
