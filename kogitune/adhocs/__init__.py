from .dicts import (
    aargs_print as print,
    find_simkey, 
    get_key_value,
    copy_dict_keys, 
    parse_path_args, 
    ChainMap,
)

from .main import (
    AdhocArguments, 
    AdhocArguments as Arguments, 
    parse_main_args, 
    load_class,
    launch_subcommand, 
    aargs_from,
    aargs_from as from_kwargs,
#    from_kwargs,
)

from .inspects import (
    check_kwargs, 
    get_parameters, 
    get_version
)

from .formats import (
    format_unit,
    p_buf as p,
)

from .logs import (
    log, 
    save_log, 
    get_log, 
    notice, 
    warn,
    start_time, 
    end_time,
)

# from .arguments import (
#     AdhocArguments, 
#     AdhocArguments as Arguments, 
#     parse_main_args, 
#     from_kwargs,
# )

from .adhoc_tqdm import (
    adhoc_progress_bar as progress_bar,
    adhoc_tqdm as tqdm,
)

def load_tokenizer(tokenizer=None, **kwargs):
    from ..stores.tokenizers import load_tokenizer
    return load_tokenizer(tokenizer, **kwargs)



