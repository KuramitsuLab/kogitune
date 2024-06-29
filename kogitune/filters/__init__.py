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

from .evals import (
    load_eval_fn,
    MaxMinFilter, 
    maxmin,
    # TokenizerCompression, 
    # TokenizerEntropy, 
    # ZLibCompression
)

# from .utils_en import (
#     contains_english, 
#     # EnglishWordCounter,
#     # WhitespaceCounter,
# )

# from .utils_ja import (
#     contains_japanese, 
#     # japanese_fraction,
#     # JapaneseWordCounter, 
#     # FootnoteFilter,
# )

from .generators import (
    load_filter, generate_filter, 
    #maxmin, 
    compose, choice, each_line, 
)