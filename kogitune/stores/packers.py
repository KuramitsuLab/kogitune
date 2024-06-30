from typing import List, Union, Tuple
from multiprocessing import Pool

import kogitune.adhocs as adhoc

from .tokenizers import *
from ..commons import *
from .files import *
from .store import DatasetStore
from .sections import find_section_fn

EMPTY_TOKENS = []

class Packer(object):
    def __init__(self, tokenizer, aargs):
        self.tokenizer = tokenizer
        self.rec = {}
        aargs.record(
            'block_size|max_length|!2048',
            'trancate_size|trancate|=0',
            'padding_size|padding|=0',
            'overlap_size|overlap|=0',
            field=self, dic=self.rec,
        )
        tokens = self.tokenizer.encode('a\nあ\n')
        self.NL_id = tokens[-2]
        self.EOS_id = tokens[-1]
#        print('DEBUG', tokens, self.overlap_size)
        self.blocks = []
        self.extra_tokens = EMPTY_TOKENS

    def __repr__(self):
        return repr(self.rec)

    def encode(self, rec: dict):
        self.blocks = []
        extra_tokens = rec['extra_tokens']
        for text in rec['input']:
            extra_tokens = self.block_text(text, extra_tokens, rec)
        rec['input'] = None
        rec['extra_tokens'] = extra_tokens
        rec['output'] = self.blocks
        return rec

    def block_text(self, text:str, extra_tokens:List[int], rec:dict) -> List[int]:
        tokens = self.encode_text(text, rec)
        if len(extra_tokens) == 0:
            rec['line_blocks'] = rec.get('line_blocks', 0) + 1
        else:
            tokens = extra_tokens + tokens
        block_size = self.block_size
        start = 0
        while start < len(tokens):
            segmented = tokens[start: start + block_size]
            if len(segmented) == block_size:
                blocked = segmented
            else:
                blocked = self.pad(segmented, rec)
            if len(blocked) == block_size:
                self.blocks.append(blocked)
                rec['blocks'] = rec.get('blocks', 0) + 1
                segmented = EMPTY_TOKENS
            start = self.find_next_start(tokens, start + block_size, rec)
        return self.trancate_tokens(segmented, rec)

    def encode_text(self, text, rec, include_eos=True):
        rec['texts'] = rec.get('texts', 0) + 1
        rec['chars'] = rec.get('chars', 0) + len(text)
        if include_eos:
            tokens = self.tokenizer.encode(text)
            rec['tokens'] = rec.get('tokens', 0) + (len(tokens)-1)
        else:
            tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
            rec['tokens'] = rec.get('tokens', 0) + len(tokens)
        return tokens

    def trancate_tokens(self, extra_tokens, rec):
        if len(extra_tokens) < self.trancate_size:
            rec['trancated'] = rec.get('trancated', 0) + 1
            rec['trancated_tokens'] = rec.get('trancated', 0) + len(extra_tokens)
            return EMPTY_TOKENS
        return extra_tokens

    def pad(self, tokens, rec):
        length = self.block_size - len(tokens)
        if 0 < length <= self.padding_size:
            # 予想しやすいpaddingを作る
            pad_id = self.EOS_id
            rec['padding'] = rec.get('padding', 0) + 1
            rec['padding_tokens'] = rec.get('padding_tokens', 0) + length
            padding = [pad_id] * length
            if length > 2:
                padding = [pad_id] + [self.tokenizer.vocab_size - length] + padding[2:]
            return tokens + padding
        return tokens
    
    def find_next_start(self, tokens:list, end:int, rec):
        if self.overlap_size > 0:
            # オーバーラップが認められるときは行頭を探す
            tokens = tokens[end-self.overlap_size:end]
            try:
                reverse_index = tokens[::-1].index(self.NL_id)
                rec['overlap_tokens'] = rec.get('overlap_tokens', 0) + reverse_index
                rec['line_blocks'] = rec.get('line_blocks', 0) + 1
                return end - 1 - reverse_index + 1 # 改行の次なので
            except ValueError as e:
                pass
        if 0 < end < len(tokens) - self.trancate_size and tokens[end-1] == self.NL_id:
            rec['line_blocks'] = rec.get('line_blocks', 0) + 1
        return end
    

def find_packer(tokenizer, aargs):
    data_type = aargs['datatype|data_type|=text']
    if data_type == 'text':
        return Packer(tokenizer, aargs)
    else: # ファインチューニング用
        # packer = SimpleTextpacker(tokenizer, args)
        raise NotImplementedError(f'datatype={data_type}')

## Store 用

def get_store_path(filenames, tokenizer, aargs):
    store_path = aargs['store_path|store_dir|store']
    if store_path is None:
        filebase = basename(filenames[0])
        tokenizer_name = tokenizer_id(tokenizer)
        store_path=f'{tokenizer_name}/{filebase}'
        return store_path
    else:
        if '/' in store_path:
            filebase = store_path.replace('/', '_')
        else:
            filebase = store_path
            tokenizer_name = tokenizer_id(tokenizer)
            store_path=f'{tokenizer_name}/{filebase}'
            aargs['store_path'] = store_path
            adhoc.print(f'保存先/Saving To.. {store_path}')

def sum_all_numbers(dics: List[dict]):
    keys = [key for key, value in dics[0].items() if isinstance(value, (int, float))]
    d = {}
    for key in keys:
        d[key] = sum(dic[key] for dic in dics)
    return d

def store_files(filenames: List[str], tokenizer=None, **kwargs):
    """
    ファイルからローカルストアを構築する
    :param filenames: ファイル名、もしくはファイル名のリスト
    """
    filenames = list_filenames(filenames)
    with adhoc.from_kwargs(**kwargs) as aargs:
        tokenizer = adhoc.load_tokenizer(tokenizer=tokenizer)
        packer = find_packer(tokenizer, aargs)
        adhoc.notice('ストアは時間がかかる場合があるから確認してね', 
                    packer=packer, 
                     input_files=filenames, 
                     tokenizer=tokenizer.name_or_path)

        store_path = get_store_path(filenames, tokenizer, aargs)
        adhoc.notice('保存先', store_path=store_path)
 
        store = DatasetStore(store_path)

        num_workers = aargs['num_workers|=1']
        N=aargs['head|N|=-1']
        call_args = dict(extra_tokens = EMPTY_TOKENS, input=None, output=None)
        if num_workers == 1:
            for docs in read_multilines(filenames, N=N, bufsize=1024):
                call_args['input'] = docs
                # call_args['output'] = None
                call_result = packer.encode(call_args)
                store.append(call_result['output'])
            record=sum_all_numbers([call_result])
        else:
            pool = Pool(num_workers)
            args_list = [call_args.copy() for _ in range(num_workers)]
            for batch in read_multilines(filenames, N=N, bufsize=1024 * num_workers):
                batch_size = len(batch) // num_workers
                for i in range(num_workers):
                    args_list[i]['input'] = batch[batch_size*i:batch_size*(i+1)]
                    # args_list[i]['output'] = None
                result_list = pool.map(packer, args_list)
                for i in range(num_workers):
                    store.append(result_list[i]['output'])
            pool.close()
            record=sum_all_numbers(result_list)
        record = packer.rec | record
        make_report(record, packer.block_size)
        adhoc.notice('お連れ様!! トークン化完了です', result=record)
        store.save(tokenizer, skip_validation=aargs['skip_validation'])
    return str(os.path.abspath(store_path))

def make_report(rec, block_size):
    ss = []
    texts = rec['texts']
    ss.append(f'文書数//texts {adhoc.format_unit(texts, scale=1000)}')
    chars = rec['chars']
    ss.append(f'文字数//chars {adhoc.format_unit(chars, scale=1000)}')
    if texts != 0:
        ss.append(f'平均文字数(chars/texts) {adhoc.format_unit(round(chars/texts,3), scale=1000)}')
    tokens = rec['tokens']
    ss.append(f'トークン数//tokens {adhoc.format_unit(tokens, scale=1000)}')
    if chars != 0:
        ss.append(f'トークン効率(tokens/chars) {adhoc.format_unit(round(tokens/chars,3), scale=1000)}')
    if texts != 0:
        ss.append(f'平均トークン数(tokens/texts) {adhoc.format_unit(round(tokens/texts,3), scale=1000)}')
    blocks = rec['blocks']
    ss.append(f'ブロック数//blocks {adhoc.format_unit(blocks, scale=1000)}')
    ss.append(f'有効トークン数 {adhoc.format_unit(blocks * block_size, scale=1000)}')
    ss.append(f'トークン有効率 {blocks * block_size*100/tokens:.2f}%')
    trancated_tokens = rec.get('trancated_tokens', 0)
    if trancated_tokens > 0:
        ss.append(f'切り詰め//trancated {adhoc.format_unit(trancated_tokens, scale=1000)}トークン')
        ss.append(f'切り詰め率 {trancated_tokens*100/tokens:.2f}%')
    padding_tokens = rec.get('padding_tokens', 0)
    if padding_tokens > 0:
        ss.append(f'パッディング//padding {adhoc.format_unit(padding_tokens, scale=1000)}トークン')
        ss.append(f'パッディング率 {padding_tokens*100/tokens:.2f}%')
    overlap_tokens = rec.get('overlap_tokens', 0) 
    if overlap_tokens > 0:
        ss.append(f'オーバラップ//overlap_tokens {adhoc.format_unit(overlap_tokens, scale=1000)}トークン')
        ss.append(f'オーバーラップ率 {overlap_tokens*100/tokens:.2f}%')
    line_blocks = rec.get('line_blocks',0)
    ss.append(f'行頭ブロック {adhoc.format_unit(line_blocks, scale=1000)}ブロック')
    ss.append(f'行頭ブロック率 {line_blocks*100/blocks:.2f}%')
    print('\n'.join(ss))

    print(rec)

