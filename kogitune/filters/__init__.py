from kogitune.stores.files import zopen, filelines

from .commons import (
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

from .replaces import (
    ReplacementFilter,
    ReplacementFilter as Replacement, 
    replace,
)

from .scores import (
    MaxMinFilter, 
    TokenizerCompression, 
    TokenizerEntropy, 
    ZLibCompression
)

from .scores_en import (
    contains_english, 
    EnglishWordCounter,
    WhitespaceCounter,
)

from .scores_ja import (
    contains_japanese, 
    japanese_fraction,
    JapaneseWordCounter, 
    FootnoteFilter,
)

from .generators import (
    load_filter, generate_filter, maxmin, 
    compose, choice, each_line, 
)