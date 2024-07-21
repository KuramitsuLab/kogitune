from typing import List, Union, Tuple
import numpy as np
import pandas as pd
import random
from multiprocessing import Pool

import kogitune.adhocs as adhoc

from .tokenizers import *
from ..commons import *
from .files import *
from .store import DatasetStore

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
    
class NoizePacker(Packer):
    def __init__(self, tokenizer, aargs):
        super().__init__(tokenizer, aargs)
        self.noize_map = self.load_token_noize_prob(aargs['noize_map|noize|!!'])
        self.mask_token_id = aargs['mask_token_id|mask_id']
        if self.mask_token_id is None:
            mask_token = aargs['mask_token|mask']
            if mask_token is not None:
                ids = self.tokenizer.convert_tokens_to_ids([mask_token])
                adhoc.notice('マスクトークン', mask_token, ids)
                self.mask_token_id = ids[0]
        self.random_seed = aargs['random_seed|=42']

    def load_token_noize_prob(self, noize_path):
        noize_ratio = noize_path if isinstance(noize_path, float) else 0.05
        noize_map = np.full(self.tokenizer.vocab_size, noize_ratio)
        if isinstance(noize_path, str):
            df = pd.read_csv(noize_path)
            for w, r in zip(df['token'], df['ratio']):
                ids = self.tokenizer.convert_tokens_to_ids([w])
                noize_map[ids[0]] = r
            adhoc.notice(f'平均ノイズ確率 {noize_map.mean()}', filepath=noize_path)
        noize_map[self.tokenizer.eos_token_id] = 0.0
        return noize_map

    def encode_text(self, text, rec, include_eos=True):
        tokens = super().encode_text(text, rec, include_eos=include_eos)
        random.seed(self.random_seed)
        new_tokens=[tokens[0]]
        if self.mask_token_id is None:
            for t in tokens[1:]:
                if random.random() > self.noize_map[t]:
                    new_tokens.append(t)
            rec['noize_tokens'] = rec.get('noize_tokens', 0) + (len(tokens) - len(new_tokens))
        else:
            masked=0
            for t in tokens[1:]:
                if random.random() > self.noize_map[t]:
                    new_tokens.append(t)
                elif new_tokens[-1] != self.mask_token_id:
                    new_tokens.append(self.mask_token_id)
                    masked+=1
            rec['noize_tokens'] = rec.get('noize_tokens', 0) + (len(tokens) - (len(new_tokens)-masked))
            rec['masked_tokens'] = rec.get('masked_tokens', 0) + masked
        self.random_seed = random.randint(0, 2**31)
        return new_tokens

def find_packer(tokenizer, aargs):
    data_type = aargs['datatype|data_type|=text']
    if data_type == 'text':
        if 'noize_map' in aargs or 'noize' in aargs:
            return NoizePacker(tokenizer, aargs)
        return Packer(tokenizer, aargs)
    else: # ファインチューニング用
        # packer = SimpleTextpacker(tokenizer, args)
        raise NotImplementedError(f'datatype={data_type}')

## Store 用

def get_store_path(filenames, tokenizer, aargs):
    store_path = aargs['store_path|store_dir|store']
    if store_path is None:
        filebase = basename(rename_linenum(filenames[0], N=0, rename=False))
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

def store_files(files: List[str], tokenizer=None, **kwargs):
    """
    ファイルからローカルストアを構築する
    :param files: ファイル名、もしくはファイル名のリスト
    """
    filenames = list_filenames(files)
    with adhoc.from_kwargs(**kwargs) as aargs:
        tokenizer = adhoc.load_tokenizer(tokenizer=tokenizer)
        packer = find_packer(tokenizer, aargs)

        store_path = get_store_path(filenames, tokenizer, aargs)
        aargs.saved(store_path, 'テンソルデータセットへのパス')
        with adhoc.open_log_file(store_path, 'pack_log.txt') as log:
            adhoc.notice('トークン化を始めます', 
                        packer=packer, 
                        input_files=filenames, 
                        tokenizer=tokenizer.name_or_path)
            store = DatasetStore(store_path)
            num_workers = aargs['num_workers|=1']
            N=aargs['head|N|=-1']
            with adhoc.start_timer() as timer:
                call_args = dict(extra_tokens = EMPTY_TOKENS, input=None, output=None)
                if num_workers == 1:
                    for docs in read_multilines(filenames, N=N, bufsize=1024):
                        call_args['input'] = docs
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
                        result_list = pool.map(packer, args_list)
                        for i in range(num_workers):
                            store.append(result_list[i]['output'])
                    pool.close()
                    record=sum_all_numbers(result_list)
                timer.notice()
                record = packer.rec | record
                make_report(record, packer.block_size)
                timer.notice('お連れ様!! トークン化完了', iteration=record['texts'], **record)
            store.save(tokenizer, skip_validation=aargs['skip_validation'])
    return str(os.path.abspath(store_path))

def make_report(rec, block_size):
    u = lambda x: adhoc.format_unit(x)
    ss = []
    texts = rec['texts']
    chars = rec['chars']
    tokens = rec['tokens']
    ss.append(f'文書数//texts {u(texts)} 文字数//chars {u(chars)} トークン数 {u(tokens)}')
    if texts > 0:
        ss.append(f'平均文字数(chars/texts) {u(round(chars/texts,3))} トークン効率(tokens/chars) {u(round(tokens/chars,3))} 平均トークン数(tokens/texts) {u(round(tokens/texts,3))}')
    trancated_tokens = rec.get('trancated_tokens', 0)
    if trancated_tokens > 0:
        ss.append(f'切り詰め {trancated_tokens*100/tokens:.2f}% {u(trancated_tokens)}トークン')
    padding_tokens = rec.get('padding_tokens', 0)
    if padding_tokens > 0:
        ss.append(f'パッディング {padding_tokens*100/tokens:.2f}% {u(padding_tokens)}トークン')
    overlap_tokens = rec.get('overlap_tokens', 0) 
    if overlap_tokens > 0:
        ss.append(f'オーバーラップ {overlap_tokens*100/tokens:.2f}% {u(overlap_tokens)}トークン')
    noize_tokens = rec.get('noize_tokens', 0) 
    if noize_tokens > 0:
        ss.append(f'ノイズ率 {noize_tokens*100/tokens:.2f}% {u(noize_tokens)}トークン')
    masked_tokens = rec.get('masked_tokens', 0) 
    if masked_tokens > 0:
        ss.append(f'マスク率 {masked_tokens*100/tokens:.2f}% {u(masked_tokens)}トークン')
    blocks = rec['blocks']
    ss.append(f'ブロック数 {u(blocks)} トークン数 {u(blocks * block_size)} 有効率 {blocks * block_size*100/tokens:.2f}%')
    line_blocks = rec.get('line_blocks',0)
    ss.append(f'行頭ブロック {line_blocks*100/blocks:.2f}% {u(line_blocks)}ブロック')
    adhoc.print('\n'.join(ss),face='')
    
