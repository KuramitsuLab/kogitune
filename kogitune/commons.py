from typing import Optional


DEFAULT_BLOCK_SIZE = 2048

DEFAULT_TOKENIZER = 'llm-jp/llm-jp-1.3b-v1.0'

# def load_tokenizer(tokenizer:str = None, aargs=None):
#     from transformers import AutoTokenizer
#     aargs = AdhocArguments.to_adhoc(aargs)
#     tokenizer = tokenizer or aargs[f'tokenizer_path|tokenizer|={DEFAULT_TOKENIZER}']
#     if isinstance(tokenizer, str):
#         local_args = aargs.get_subargs('tokenizer_*|trust_remote_code', exclude='tokenizer_path')
#         if 'trust_remote_code' not in local_args:
#             local_args['trust_remote_code'] = True
#         if 'use_fast' not in local_args:
#             local_args['use_fast'] = False
#         # AutoTokenizer.from_pretrained(tokenizer, legacy=legacy, trust_remote_code=True, use_fast=False)
#         return AutoTokenizer.from_pretrained(tokenizer, **local_args)
#     return tokenizer

CHUNK_MAGIC = 8
