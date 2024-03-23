from transformers import AutoTokenizer
from .adhocargs import AdhocArguments, parse_path_arguments
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

DEFAULT_TOKENIZER = os.environ.get('DEFAULT_TOKENIZER', 'llm-jp/llm-jp-1.3b-v1.0')

def configurable_tokenizer(tokenizer = None, akey='', **kwargs):
    with AdhocArguments.from_main(**kwargs) as aargs:
        tokenizer = tokenizer or aargs[f'{akey}tokenizer_path|{akey}tokenizer|tokenizer_path|model_path|tokenizer|={DEFAULT_TOKENIZER}']
        if isinstance(tokenizer, str):
            tokenizer, local_args = parse_path_arguments(tokenizer)
            if 'trust_remote_code' not in local_args:
                local_args['trust_remote_code'] = True
            if 'use_fast' not in local_args:
                local_args['use_fast'] = False
            # AutoTokenizer.from_pretrained(tokenizer, legacy=legacy, trust_remote_code=True, use_fast=False)
            return AutoTokenizer.from_pretrained(tokenizer, **local_args)
    return tokenizer
