from .dicts import parse_path_args
from .main import aargs_from

import os

DEFAULT_TOKENIZER = os.environ.get('DEFAULT_TOKENIZER', 'llm-jp/llm-jp-1.3b-v1.0')

def adhoc_load_tokenizer(tokenizer = None, **kwargs):
    from transformers import AutoTokenizer
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    with aargs_from(**kwargs) as aargs:
        tokenizer = tokenizer or aargs[f'tokenizer_path|tokenizer|model_path|={DEFAULT_TOKENIZER}']
        if isinstance(tokenizer, str):
            tokenizer, local_args = parse_path_args(tokenizer)
            if 'trust_remote_code' not in local_args:
                local_args['trust_remote_code'] = True
            if 'use_fast' not in local_args:
                local_args['use_fast'] = False
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, **local_args)
    return tokenizer
