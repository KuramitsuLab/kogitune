from typing import Any, List, Union

import os
import random

import json
import shutil
import hashlib

from filelock import FileLock
import numpy as np

import torch
from torch.utils.data import Dataset

from ..adhocargs import AdhocArguments, adhoc_argument_parser
from ..commons import *
from ..file_utils import *
from ..tokenizers import *
from ..splitters import make_local_store

# ChunkedDataset

def get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

def get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1

def url_to_hash(url):
    return hashlib.md5(url.encode()).hexdigest()

def local_cache_dir(cache_dir, url):
    hash = url_to_hash(url)
    if hash in cache_dir:
        return cache_dir
    return safe_join_path(cache_dir, hash)

class _DummyFileLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

def _FileLock(lockfile: str):
    # lockがNoneなら何もしない
    return _DummyFileLock() if lockfile is None else FileLock(lockfile)


def get_token_size(s):
    try:
        if isinstance(s, str):
            if s[-1] == 'M' or s[-1] == 'm':
                return int(1000_000 * float(s[:-1]))
            if s[-1] == 'B' or s[-1] == 'b':
                return int(1000_000_000 * float(s[:-1]))
            if s[-1] == 'T' or s[-1] == 't':
                return int(1000_000_000_000 * float(s[:-1]))
            s = float(s)
        if isinstance(s, float):
            return int(1000_000_000 * s)
        if isinstance(s, int):
            if s < 1000:
                return int(1000_000_000 * s)
    except:
        raise ValueError(f'{s} is not valid token size.')

def random_chunks(n_chunks, max_length):
    return [np.random.randint(32, 8000, max_length) for _ in range(n_chunks)]

class ChunkQueue(object):
    def __init__(self, n_chunks, max_length, n=4):
        self.keyvalues = [('', random_chunks(n_chunks, max_length))] * n
        self.n = n
        self.head = -1

    def get(self, key):
        for label, chunks in self.keyvalues:
            if label == key:
                return chunks

    def set(self, key, chunks):
        self.head += 1
        self.keyvalues[self.head % self.n] = (key, chunks)

    def last_chunks(self):
        return self.keyvalues[self.head % self.n][1]

    
class TokenDataset(Dataset):
    def __init__(self, url: str, max_length: int, prefix: str, config:dict, local_cache_dir:str, local_args:dict):
        self.url = safe_dir(url)
        if '/' in self.url:
            _, _, self.name = self.url.rpartition('/')
        else:
            self.name = self.url
        # self.args = args
        self.max_length = max_length
        self.cache_dir = local_cache_dir
        self.lock_file = None

        self.prefix = prefix
        self.config = config
        
        # 設定
        # self.n_tokens = config.get('n_tokens', 0)
        self.tokenizer_path = config.get("tokenizer_path", DEFAULT_TOKENIZER)
        self.compressed = config.get("compressed", None)
        self.n_chunks = config.get("n_chunks", N_CHUNKS)
        logs = [f'{self.name}:']

        self.chunk_files = list(config['files'].keys())
        self.n_items = config.get("n_items", len(self.chunk_files)*self.n_chunks)
        if len(self.chunk_files) * self.n_chunks != self.n_items:
            self.chunk_files = self.chunk_files[:-1]
            n_items = len(self.chunk_files) * self.n_chunks
            logs.append(f'末尾トリム {self.n_items - n_items}')
            self.n_items = n_items

        start_index = local_args.get('start', 0)
        if start_index != 0:
            if isinstance(start_index, int):
                start_index = start_index / self.n_items
            start_index = int(len(self.chunk_files) * start_index) % len(self.chunk_files)
            self.chunk_files = self.chunk_files[start_index:] + self.chunk_files[:start_index]
            logs.append(f'先頭の位置調整 start={start_index/len(self.chunk_files):.4f}')

        random_seed = local_args.get('random_seed', None)
        if isinstance(random_seed, int):
            random.seed(random_seed)
            random.shuffle(self.chunk_files)
            logs.append(f'ランダム化 seed={random_seed}')

        if get_world_size() > 1:
            rank = get_rank()
            world_size = get_world_size()
            self.chunk_files = [f for i, f in enumerate(self.chunk_files) if i % world_size == rank]
            self.n_items = len(self.chunk_files) * self.n_chunks
            logs.append(f'ランク rank={rank}/{world_size}')

        self.n_subblocks = 1
        chunk_block_size = 0
        if config.get('max_length', 0) == config.get('min_length', -1):
            chunk_block_size = config.get('max_length', 0)
        if chunk_block_size > max_length and chunk_block_size % max_length == 0: 
            self.n_subblocks = chunk_block_size // max_length
            self.n_chunks = self.n_chunks * self.n_subblocks
            self.n_items = self.n_items * self.n_subblocks
            logs.append(f'再分割 subblocks={self.n_subblocks}')

        resize = local_args.get('tokens', None) or local_args.get('token', None)
        if resize:
            tokens = get_token_size(resize)
            n_items = math.ceil(tokens / self.max_length)
            if n_items > self.n_items:
                factor = math.ceil(n_items/self.n_items)
                logs.append(f'データ重複化 resize={n_items/self.n_items:,.2f}')
                self.chunk_files = self.chunk_files * factor
            else:
                logs.append(f'サイズ調整 resize={n_items/self.n_items:,.2f}')
            self.n_items = n_items

        resize = local_args.get('resize', None) or local_args.get('length', None)
        if resize:
            n_items = math.ceil(len(self.chunk_files) * self.n_chunks * resize)
            if resize > 1.0:
                factor = int(math.ceil(resize // 1))
                logs.append(f'データ重複化 resize={n_items/self.n_items:,.2f}')
                self.chunk_files = self.chunk_files * factor
            else:
                logs.append(f'サイズ調整 resize={n_items/self.n_items:,.2f}')
            self.n_items = n_items
        
        logs.append(f'トークン数 {self.n_items * max_length/10**9:.2f}B')
        verbose_print(' '.join(logs))
        self.queue = ChunkQueue(self.n_chunks, self.max_length)
        self.prefetch=1
        self.try_prefetch(0)

    def __repr__(self):
        return str(self.name)

    def __len__(self):
        return self.n_items

    def get_chunks(self, chunk_file):
        chunks = self.queue.get(chunk_file)
        if chunks is None:
            with _FileLock(self.lock_file):
                chunk_file2 = resolve_file(self.url, chunk_file, self.cache_dir, self.compressed)
                chunks = load_chunk_file(chunk_file2, subblocks=self.n_subblocks)
            if chunks is not None:
                self.queue.set(chunk_file, chunks)
            else:
                verbose_print(f'{self.name}: 破損したファイル {chunk_file}')
                chunks = self.queue.last_chunks()
        return chunks

    def __getitem__(self, index):
        chunk_index = index // self.n_chunks
        chunk_file = self.chunk_files[chunk_index]
        chunks = self.get_chunks(chunk_file)
        if self.prefetch > 0 and index % self.n_chunks == 0:
            self.try_prefetch(index+(self.prefetch*self.n_chunks))
        return chunks[index % self.n_chunks]

    def try_prefetch(self, index):
        if self.prefetch > 0:
            chunk_index = index // self.n_chunks
            chunk_file = self.chunk_files[chunk_index % len(self.chunk_files)]
            resolve_file(self.url, chunk_file, self.cache_dir, self.compressed, sync=False)

def make_prefix(data_type, max_length, split):
    prefix = f'{data_type}{max_length}{split}_config.json'
    return prefix

def find_dataset_config(url, datatype, max_length, split, cache_dir):
    while max_length <= 4096:
        prefix = f'{datatype}{max_length}{split}'
        config_file = resolve_file(url, f'{prefix}_config.json', cache_dir, verbose=False)
        if os.path.exists(config_file):
            try:
                with open(config_file) as f:
                    config = json.load(f)
                return prefix, config
            except BaseException as e:
                print(f'{config_file} is broken: {e}')
                return None
        max_length *= 2
    return None

def prepare_dataset(url_list, max_length, cache_dir, args, tokenizer=None):
    datatype = args['data_type|datatype|=text']
    split = args['split|=train']
    global_args = {'datatype': datatype, 'split': split}
    datasets = []
    for url in url_list:
        url, local_args = parse_url_args(url, global_args)
        url = safe_dir(url)
        if url.endswith('.gz') or url.endswith('.zst') or url.endswith('.jsonl') or url.endswith('.txt'):
            # tokenizer = prepare_tokenizer(tokenizer)
            url = make_local_store(url, tokenizer, local_args)
        cache_dir = local_cache_dir(cache_dir, url)
        found = find_dataset_config(url, datatype, max_length, split, cache_dir)
        if found:
            prefix, config = found
            dataset = TokenDataset(url, max_length, prefix, config, cache_dir, local_args)
            # if self.check_tokenizer(url, dataset) == False:
            #     continue
            datasets.append(dataset)
        else:
            verbose_print(f'{url} は、見つかりません。スキップして学習を続けます。')
    return datasets

class Indexer(Dataset):
    def __init__(self, dataset: TokenDataset, random_seed=42):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.random_seed = random_seed
        self.epoch = 0
        self.count = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = self.count % self.dataset_size
        if index == 0:
            self.epoch += 1
            self.on_epoch()
        self.count += 1
        return self.dataset[index]

    def skip(self):
        if self.count % self.dataset_size == 0:
            self.epoch += 1
            self.on_epoch()
        self.count += 1
    
    def on_epoch(self):
        random.seed(self.random_seed)
        random.shuffle(self.dataset.chunk_files)

class TextBlockCollator(object):
    def __init__(self, max_length, args):
        self.max_length = max_length
        self.is_seq2seq = args['datatype|data_type|=text'] == 'seq2seq'

    def __call__(self, data):
        return torch.tensor(data[:self.max_length].astype(np.int64), dtype=torch.long)


class MixierDataset(Dataset):
    def __init__(self, datasets: List[TokenDataset], collator_fn, batch_size=1024, random_seed=42):
        self.datasets = datasets
        self.collator_fn = collator_fn
        lens = [len(d) for d in datasets]
        self.total_length = sum(lens)
        mixers = [f'{d} {round(100*len(d)/self.total_length, 2)}%' for d in datasets]
        qmixers = [round(batch_size * (dlen/self.total_length)) for dlen in lens]
        verbose_print(f'データセットの混成比率: {mixers} batch_size={sum(qmixers)}')
        indexers = []
        for dataset, mix in zip(datasets, qmixers):
            if mix == 0:
                verbose_print(f'{dataset}は小さ過ぎるので無視されます。')
            indexers.extend([Indexer(dataset)] * mix)
        random.seed(random_seed)
        random.shuffle(indexers)
        self.indexers = indexers
        self.count = 0
        # 正確でないかもしれない
        self.batch_size = len(self.indexers)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        item = self.indexers[self.count % self.batch_size][index]
        self.count += 1
        return self.collator_fn(item)

    def skip(self, count):
        for c in range(count):
            self.indexers[c % self.batch_size].skip()
        self.count = count

    def get_trained_count(self):
        tokens = {}
        for ds in self.indexers:
            key = str(ds)
            if key not in tokens:
                tokens[key] = ds.count
        return tokens

# TEAM
# PROJECT
# RUN

class DummyWandb:
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

def load_wandb(args: AdhocArguments):
    try:
        import wandb
        if 'wandb_team' in args:
            wandb.init(
                entity=args['wandb_team'],
                project=args['project'],
                name=args['run_name'],
            )
        else:
            wandb.init(
                project=args['project'],
                name=args['run_name'],
            )
        return wandb
    except ModuleNotFoundError:
        verbose_print('wandb は入れた方がいいよ')
    return DummyWandb()

def get_trained_global_step(path: str):
    state_file = os.path.join(path, 'trainer_state.json')
    if os.path.exists(state_file):
        try:
            with open(state_file) as f:
                data = json.load(f)
                return data['global_step']
        except:
            pass

    if not os.path.isdir(path):
        return 0

    # 指定されたパス内のすべてのファイルとディレクトリのリストを取得
    dirs = [os.path.join(path, item) for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    if len(dirs) == 0:
        return 0
    
    # 最も新しいディレクトリを見つける
    newest = max(dirs, key=lambda dir: os.path.getmtime(dir))
    return get_trained_global_step(newest)

def create_output_path(run_name):
    for i in range(1, 1000):
        output_path = f'output_{run_name}_{i}'
        if not os.path.exists(output_path):
            return output_path
    return f'output_{run_name}'


def check_composer_args(args:None):
    if args is None:
        args = AdhocArguments({})
    elif isinstance(args, dict):
        args = AdhocArguments(args)

    if 'resume_from_checkpoint' in args and not args['overwrite_output_dir|=True']:
        resume_from_checkpoint = safe_dir(str(args['resume_from_checkpoint']))
        if 'output_dir' not in args and os.path.isdir(resume_from_checkpoint):
            args['output_dir'] = os.path.dirname(resume_from_checkpoint)

    if 'project' not in args:
        args['project'] = f'kogitune-sandbox'

    if 'run_name' not in args:
        args['run_name'] = f'run{os.getpid()}'

    if 'output_dir' not in args:
        args['output_dir'] = create_output_path(args['run_name'])
        verbose_print(f'出力先:', args['output_dir'])

    return args

class DatasetComposer():
    def __init__(self, url_list:List[str], max_length:int, 
                 args:dict=None,
                 cache_dir = None, cleanup=False, 
                 collator_fn = None, tokenizer=None):
        self.max_length = max_length
        self.args = check_composer_args(args)
 
        # キャッシュ
        cache_dir = cache_dir or self.args['kg_cache_dir|cache_dir']
        if cache_dir is None:
            self.cache_dir = safe_join_path('.', get_filename_by_pid('cache'))
            self.cleanup = False if get_rank() > 0 else True
        else:
            self.cache_dir = safe_dir(cache_dir)
            self.cleanup = False if get_rank() > 0 else cleanup
        if os.path.isdir(self.cache_dir):
            verbose_print(f'既に存在するキャッシュ {self.cache_dir} を使います。')
            self.cleanup = False
        os.makedirs(self.cache_dir, exist_ok=True)
        # self.lock_file = safe_join_path(self.cache_dir, get_filename_by_pid('cache')) if use_filelock else None

        url_list = parse_url_list(url_list)
        self.tokenizer = tokenizer
        self.datasets = prepare_dataset(url_list, max_length, self.cache_dir, self.args, tokenizer)
        self.train_dataset = None
        if collator_fn:
            self.collator_fn = collator_fn
        else:
            self.collator_fn = TextBlockCollator(max_length, self.args)

    def get_tokenizer(self):
        if not self.tokenizer and len(self.datasets) > 0:
            self.tokenizer = load_tokenizer(self.datasets[0].tokenizer_path)
        return self.tokenizer

    def get_train_dataset(self, batch_size=None, resume=None):
        if not self.train_dataset:
            batch_size = batch_size or self.args['global_batch_size|batch_size|=1024']
            self.train_dataset = MixierDataset(self.datasets, self.collator_fn, batch_size)
            resume_path = resume or self.args['resume_from_checkpoint']
            if resume_path:
                resume_step = get_trained_global_step(resume_path)
                if resume_step == 0:
                    verbose_print(f'チェックポイント {resume_path} が見つかりません')
                if resume_step > 0:
                    verbose_print(f'チェックポイント step={resume_step} から再開します。')
                    self.train_dataset.skip(resume_step * batch_size)
        return self.train_dataset

    def report(self):
        if self.train_dataset:
            global_count = self.train_dataset.count
            global_step = global_count//1024
            total_tokens = global_count * self.max_length
            verbose_print(f'ステップ {global_step:,} イテレーション {global_count:,} トークン数 {format_unit(total_tokens)} {total_tokens:,}')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.n_items = 0
        self.mixed = None
        self.report()
        if self.cleanup and os.path.isdir(self.cache_dir):
            try:
                shutil.rmtree(self.cache_dir)
                verbose_print('Cleaned up', self.cache_dir)
            except:
                pass

    def get_collator(self, model):
        from transformers import DataCollatorForLanguageModeling
        tokenizer = self.get_tokenizer()
        return DataCollatorForLanguageModeling(tokenizer, 
                                               pad_to_multiple_of=8, 
                                               mlm=False)

    def get_train_args(self, device_batch_size=None, **kwargs):
        from transformers import TrainingArguments
        self.args.update(kwargs)
        args = self.args
        global_batch_size = args['global_batch_size|batch_size|=1024']
        device_batch_size = device_batch_size or args['device_batch_size|=16']
        gas = global_batch_size // device_batch_size
        verbose_print(f'batch_size global={global_batch_size} device={device_batch_size} gradient_accumulation_steps={gas}')
        overwrite_output_dir = 'resume_from_checkpoint' not in self.args
        train_args = TrainingArguments(
            output_dir=args['output_dir|=output'],
            overwrite_output_dir=args[f'overwrite_output_dir|={overwrite_output_dir}'],
            per_device_train_batch_size=args[f'per_device_train_batch_size|={device_batch_size}'],
            gradient_accumulation_steps=args[f'gradient_accumulation_steps|={gas}'],
            # per_device_eval_batch_size=64,
            auto_find_batch_size=args['auto_find_batch_size|=True'],  # バッチサイズ自動
            do_eval=args['do_eval|=False'],
            # evaluation_strategy='steps',
            # eval_steps=50,
            optim=args['optim|=adamw_torch_fused'],
            num_train_epochs=args['num_train_epochs|=1'],
            max_steps=args['max_steps|=-1'],
            weight_decay=args['weight_decay|=0.1'],
            lr_scheduler_type=args['lr_scheduler_type|=constant'],
            learning_rate=args['learning_rate|=4e-4'], #Phi-1
            logging_steps=args['logging_steps|=10'],
            dataloader_pin_memory=False,
            save_steps=args['save_steps|=1000'],
            save_total_limit=args['save_total_limit|=2'],
#            save_only_model=args['save_only_model|=False'],
#            neftune_noise_alpha=args['neftune_noise_alpha'],
            torch_compile=args['torch_compile|=False'],
            bf16=args[f'bf16|={is_bf16_available()}'],
            fp16=args[f'fp16|={torch.cuda.is_available()}'],
        )
        return train_args
    
    def train(self, model=None, save_path=None):
        from transformers import Trainer, AutoModelForCausalLM
        from kogitune.trainers.scratch import print_summary
        if model is None:
            model_path = self.args['resume_from_checkpoint|model_path|model']
            if model_path is None:
                self.args.raise_unset_key('model_path')
            model = AutoModelForCausalLM.from_pretrained(model_path)
        resume_from_checkpoint=self.args['resume_from_checkpoint|=False']
        wandb = load_wandb(self.args)
        if 'max_time' in self.args or 'sge_walltime_sec' in self.args:
            max_time=self.args['max_time|sge_walltime_sec']
            verbose_print(f'安全に止めるタイマーをセットしました。{max_time}')
            trainer = Trainer(
                model=model,
                data_collator=self.get_collator(model),
                train_dataset=self.get_train_dataset(resume=resume_from_checkpoint),
                args=self.get_train_args(),
                callbacks=[TimeoutStoppingCallback(max_time=max_time)]
            )
        else:
            trainer = Trainer(
                model=model,
                data_collator=self.get_collator(model),
                train_dataset=self.get_train_dataset(resume=resume_from_checkpoint),
                args=self.get_train_args(),
            )
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        save_path = save_path or self.args['save_path']
        if save_path:
            self.get_tokenizer().save_pretrained(save_path)
            model.save_pretrained(save_path)
        print_summary(result)
        wandb.finish()
        return result


def is_bf16_available():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            if 'A100' in gpu_name or 'H100' in gpu_name:
                return True
    return False

def parse_time_as_second(time:str):
    if isinstance(time, int):
        return time
    hms = map(int, time.split(':'))
    if len(hms) == 3:
        return hms[0] * 3600 + hms[1] * 60 + hms[2]
    return hms[0] * 3600

import transformers

class TimeoutStoppingCallback(transformers.TrainerCallback):

    def __init__(self, max_time: Union[int,str], safety_time=300, safety_margin=1.05):
        self.start_time = time.time()
        self.estimated_end_time = self.start_time + parse_time_as_second(max_time) - 300
        self.save_count = 0
        self.margin = safety_margin
        self.safety_time = safety_time

    def on_save(self, args, state, control, **kwargs):
        current_time = time.time()
        self.save_count += 1
        interval = (current_time - self.start_time) / self.save_count
        remaining = self.estimated_end_time - current_time
        verbose_print(f'残り時間 {format_unit(remaining, scale=60)} 間隔 {format_unit(interval, scale=60)}')
        if interval * self.margin > remaining:
            verbose_print(f'そろそろ時間だから終了するよ！')
            control.should_training_stop = True

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        remaining = self.estimated_end_time - current_time
        if remaining < 300:
            verbose_print(f'残り時間 {format_unit(remaining, scale=60)} が少ないから緊急停止するよ')
            control.should_save = True
            control.should_training_stop = True

