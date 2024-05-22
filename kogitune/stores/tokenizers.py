from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os
import hashlib
import kogitune.adhocs as adhoc

DEFAULT_TOKENIZER = os.environ.get('DEFAULT_TOKENIZER', 'llm-jp/llm-jp-1.3b-v1.0')
if 'TOKENIZERS_PARALLELISM' not in os.environ:
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

class Tokenizer:
    pass

def tokenizer_hash(tokenizer: Tokenizer):
    ws = [(id, w) for w, id in tokenizer.get_vocab().items()]
    ws.sort()
    allvoc = ''.join(w for _, w in ws)
    #print(len(allvoc), allvoc[:100], allvoc[:100].encode())
    return hashlib.md5(allvoc.encode()).hexdigest()

def tokenizer_id(tokenizer: Tokenizer):
    _, _, name_or_path = tokenizer.name_or_path.rpartition('/')
    name_or_path=name_or_path.lower().replace('_', '-')
    names = [name for name in name_or_path.split('-') if name.isalpha()]
    if len(names) == 0:
        names = name_or_path.split('-')[:1]
    md5 = tokenizer_hash(tokenizer)
    names.append(md5[:4])
    return '-'.join(names)

def is_special_token(w):
    return w.startswith('<') and w.endswith('>') and not w.startswith('<0x')

def tokenizer_special_tokens(tokenizer: Tokenizer):
    ws = [(id, w) for w, id in tokenizer.get_vocab().items()]
    ws.sort()
    return [(w, id) for id, w in ws if is_special_token(w)]

def tokenizer_as_json(tokenizer: Tokenizer):
    return dict(
        name_or_path=tokenizer.name_or_path,
        vocab_size=tokenizer.vocab_size,
        special_tokens = tokenizer_special_tokens(tokenizer),
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.pad_token_id,
        hash=tokenizer_hash(tokenizer), 
    )

def load_tokenizer(tokenizer: Union[Tokenizer, str] = None, **kwargs):
    from transformers import AutoTokenizer
    with adhoc.from_kwargs(**kwargs) as aargs:
        tokenizer = tokenizer or aargs[f'tokenizer_path|tokenizer|model_path|={DEFAULT_TOKENIZER}']
        if isinstance(tokenizer, str):
            tokenizer, local_args = adhoc.parse_path_args(tokenizer)
            if 'trust_remote_code' not in local_args:
                local_args['trust_remote_code'] = True
            if 'use_fast' not in local_args:
                local_args['use_fast'] = False
            adhoc.check_kwargs(local_args, AutoTokenizer.from_pretrained, path=tokenizer)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, **local_args)
    adhoc.print(f'トークンナイザーをロードしたよ',
                tokenizer_as_json(tokenizer),
                verbose='tokenizer', 
                once=f'tokenizer={tokenizer.name_or_path}')
    return tokenizer


"""

DEFAULT_NL = '<nL>'
DEFAULT_SEP = '<seP>'
DEFAULT_OUTPUT_SEP = '<outpuT>'
DEFAULT_ELLIPSIS = '<ellipsiS>'

def find_token_id(tokenizer: AutoTokenizer, *token: str)->int:
    ids = tokenizer.convert_tokens_to_ids(token)
    for id in ids:
        if id != tokenizer.unk_token_id:
            return id
    return tokenizer.unk_token_id

def find_newline_token_id(tokenizer: AutoTokenizer):
    return find_token_id(tokenizer, DEFAULT_NL, "<nL>", "<nl>")

def find_sep_token_id(tokenizer: AutoTokenizer):
    return find_token_id(tokenizer, DEFAULT_SEP, "<seP>", "<sep>")

def find_output_sep_token_id(tokenizer: AutoTokenizer):
    return find_token_id(tokenizer, DEFAULT_OUTPUT_SEP, "<outpuT>", "<output>")

def find_ellipsis_token_id(tokenizer: AutoTokenizer):
    return find_token_id(tokenizer, DEFAULT_ELLIPSIS, "<ellipsiS>", "<ellipsis>", "<masK>", "<mask>", "<extra_id_99>")

_EXTRA_IDS = [f'<extra_id_{i}>' for i in range(100)]

def find_extra_ids(tokenizer: AutoTokenizer):
    return tokenizer.convert_tokens_to_ids(_EXTRA_IDS)


def _upper_repl(matchobj):
    #print(matchobj, matchobj.group(0))
    return '<cZ>' + matchobj.group(0).lower()

def _cap_repl(matchobj):
    #print(matchobj, matchobj.group(0))
    return matchobj.group(0)[4:].upper()

_UpperPattern = re.compile('([A-Z][a-z])')
_CapitalizedPattern = re.compile(r'(\<cZ\>[a-z])')

def pre_encode(s):
    if isinstance(s, str):
        s = _UpperPattern.sub(_upper_repl, s)
        return s.replace('\t', '    ').replace('\n', '<nL>')
    if isinstance(s, tuple):
        return tuple(map(pre_encode, s))
    return s

def post_decode(s):
    if isinstance(s, str):
        return _CapitalizedPattern.sub(_cap_repl,s).replace('<nL>', '\n')
    return s


def adapt_tokenizer(tokenizer: AutoTokenizer):

    orig_tokenize = tokenizer.tokenize
    def papertown_tokenize(text: str, pair: Optional[str] = None, add_special_tokens: bool = False) -> List[str]:
        text=pre_encode(text)
        pair=pre_encode(pair)
        return orig_tokenize(text)
    tokenizer.tokenize = papertown_tokenize

    orig_encode_plus = tokenizer.encode_plus
    def papertown_encode_plus(
        text,
        text_pair = None,
        add_special_tokens: bool = True,
        padding = False,
        truncation = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        text = pre_encode(text)
        text_pair=pre_encode(text_pair)
        return orig_encode_plus(text, 
        text_pair=text_pair,
        add_special_tokens = add_special_tokens,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        stride=stride,
        is_split_into_words=is_split_into_words,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors,
        return_token_type_ids=return_token_type_ids,
        return_attention_mask=return_attention_mask,
        return_overflowing_tokens=return_overflowing_tokens,
        return_special_tokens_mask=return_special_tokens_mask,
        return_offsets_mapping=return_offsets_mapping,
        return_length=return_length,
        verbose=verbose,
        **kwargs,
    )
    tokenizer.encode_plus = papertown_encode_plus

    orig_batch_encode_plus = tokenizer.batch_encode_plus
    def papertown_batch_encode_plus(
        batch_text_or_text_pairs,
        add_special_tokens: bool = True,
        padding = False,
        truncation = None,
        max_length = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        batch_text_or_text_pairs = [pre_encode(x) for x in batch_text_or_text_pairs]        
        return orig_batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )
    tokenizer.batch_encode_plus = papertown_batch_encode_plus

    orig_convert_tokens_to_string = tokenizer.convert_tokens_to_string
    def papertown_convert_tokens_to_string(tokens: List[str]):
        s = orig_convert_tokens_to_string(tokens)
        return post_decode(s)
    tokenizer.convert_tokens_to_string = papertown_convert_tokens_to_string

    orig_decode = tokenizer.decode
    def papertown_decode(token_ids, 
                         skip_special_tokens: bool = False,
                         clean_up_tokenization_spaces: bool = None,
                        **kwargs):
        s = orig_decode(token_ids, 
                        skip_special_tokens=skip_special_tokens,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        **kwargs)
        return post_decode(s)
    tokenizer.decode = papertown_decode

def load_tokenizer(tokenizer_path=DEFAULT_TOKENIZER, adapt=True):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, legacy=False, trust_remote_code=True, use_fast=False)
    #tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if adapt:
        newline_token_id = find_token_id(tokenizer, "<nL>")
        if newline_token_id != tokenizer.unk_token_id:
            adapt_tokenizer(tokenizer)
    return tokenizer
"""

