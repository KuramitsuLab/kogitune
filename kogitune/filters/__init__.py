from .filters import (
    TextFilter, 
    ComposeFilter, compose,
    ChoiceFilter, choice,
#    ExtractFilter, 
)

from .maxmins import (
    load_eval_fn,
    MaxMinFilter, maxmin,
    # TokenizerCompression, 
    # TokenizerEntropy, 
    # ZLibCompression
)

from .languages import (
    LangSetFilter, langset,
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
