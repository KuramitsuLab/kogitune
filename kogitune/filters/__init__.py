from .filters import (
    TextFilter, 
    ComposeFilter, compose,
    ChoiceFilter, choice,
#    ExtractFilter, 
)

from .maxmins import (
    load_eval_fn,
    MaxMinFilter, maxmin,
)

from .languages import (
    LangSetFilter, langset,
)

from .documents import (
    UnicodeNormalization, DuplicatedLineFilter,
    LineByLineFilter, 
)

from .replaces import (
    ReplacementFilter,
    ReplacementFilter as Replacement, 
    replace,
)
