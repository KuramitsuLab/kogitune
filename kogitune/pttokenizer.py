from typing import List, Union, Optional

from transformers import T5Tokenizer
import re

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


class PapertownTokenizer(T5Tokenizer):

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False) -> List[str]:
        text=pre_encode(text)
        pair=pre_encode(pair)
        return super.tokenizer(text, add_special_tokens=add_special_tokens)

    def encode_plus(self, 
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
        return super().encode_plus(text, 
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

    def batch_encode_plus(self,
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
        return super().batch_encode_plus(
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

    def onvert_tokens_to_string(self, tokens: List[str]):
        s = super().convert_tokens_to_string(tokens)
        return post_decode(s)
    
    def decode(self,
                         token_ids, 
                         skip_special_tokens: bool = False,
                         clean_up_tokenization_spaces: bool = None,
                        **kwargs):
        s = super().decode(token_ids, 
                        skip_special_tokens=skip_special_tokens,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        **kwargs)
        return post_decode(s)
