from kogitune.utils_file import zopen, filelines

from .base import (
    TextFilter, 
    ComposeFilter, 
    ChoiceFilter,
#    ExtractFilter, 
)

from .documents import (
    UnicodeNormalization, 
    LineByLineFilter, DuplicatedLine,
#    JSONConvertor, JSONTemplateConvertor,
)

from .replace import (
    ReplacementFilter
)

from .scores import (
    MaxMinFilter, 
    TokenizerCompression, TokenizerEntropy, ZLibCompression
)

from .lang_en import (
    contains_english, 
    EnglishWordCounter,
    WhitespaceCounter,
)

from .lang_ja import (
    contains_japanese, 
    japanese_fraction,
    JapaneseWordCounter, 
    FootnoteFilter,
)

from .generators import (
    load_filter, generate_filter, filter, maxmin, 
    compose, choice, each_line, 
)