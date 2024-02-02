from kogitune.file_utils import zopen, filelines

from .base import (
    TextFilter, 
    ComposeFilter, 
    ChoiceFilter,
    ExtractFilter, 
)

from .documents import (
    UnicodeNormalization, 
    LineByLineFilter, DuplicatedLineFilter,
    JSONConvertor, JSONTemplateConvertor,
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
)

from .lang_ja import (
    contains_japanese, 
    japanese_fraction,
    JapaneseWordCounter, 
    FootnoteFilter,
)

