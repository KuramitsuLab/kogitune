from typing import List, Any, Union

from ..adhoc_args import (
    AdhocArguments, adhoc,
    configurable_tokenizer, 
    configurable_tqdm,
    parse_path_arguments
)

from ..utils_file import zopen, basename
from kogitune.filters.scores_ja import contains_japanese
