import os
import json

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers, decoders

import sentencepiece as spm

import kogitune.adhocs as adhoc
from kogitune.stores.files import list_filenames

EOS_TOKEN = '</s>'
EOS_ID = 0
UNK_TOKEN = '<unk>'
UNK_ID = 1
MASK_TOKEN = '…'
MASK_ID = 2

VOCAB = [
    (EOS_TOKEN, 0.0), # EOD
    (UNK_TOKEN, 0.0), # UNK
    (MASK_TOKEN, 0.0), # MASK
    ("\n", -0.01), # 改行

]

def extract_vocab_score(tokenizer):
    state = json.loads(bytes(tokenizer.model.__getstate__()).decode('utf-8'))
    return clean_vocab(state['vocab'], remove_meta=False)

def extract_vocab_score_from_spiece(model_path):
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(model_path)
    vocab = [(sp_model.id_to_piece(id), sp_model.get_score(id)) for id in range(sp_model.GetPieceSize())]
    return vocab

def clean_vocab(vocab, remove_meta=True):
    ws = set(piece for piece, _ in vocab)
    new_vocab = []
    new_vocab_set = set()
    for w, score in vocab:
        if remove_meta and w.startswith('▁') and len(w) > 1:
            piece = w[1:]
            if piece not in ws:
                if w not in new_vocab_set:
                    new_vocab_set.add(piece)
                    new_vocab.append((piece, score))
            else:
                if w not in new_vocab_set:
                    new_vocab_set.add(w)
                    new_vocab.append((w, score))
        else:
            if w not in new_vocab_set:
                new_vocab_set.add(w)
                new_vocab.append((w, score))
    return new_vocab

def add_vocab_score(vocab, words):
    ws = set(piece for piece, _ in vocab)
    scores = [[] for _ in range(100)]
    for piece, score in vocab:
        length = len(piece)
        if length < 100:
            scores[length].append(score)
    _scores = []
    for s in scores:
        if len(s) > 0:
            _scores.append(sum(s)/len(s))
        else:
            _scores.append(-20.0)
    scores = _scores
    vocab = vocab[:]
    for w in words:
        if isinstance(w, (tuple, list)):
            w, score = w
        else:
            score = scores[len(w)]
        if w not in ws:
            # print('@', w, score)
            vocab.append((w, score))
    return vocab

def remove_vocab_score(vocab, words):
    ws = set(words)
    new_vocab = []
    for piece, score in vocab:
        if piece not in ws:
            new_vocab.append((piece, score))
    return new_vocab

def unigram_tokenizer(vocab=VOCAB, unk_id=UNK_ID, byte_fallback=True):
    if byte_fallback:
        vocab = add_vocab_score(vocab, [(f'<0x{n:02X}>', -10.0) for n in range(0,256)])  
    tokenizer = Tokenizer(models.Unigram(vocab, unk_id, byte_fallback=byte_fallback))
    # プレトークナイザーの設定
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    # デコーダーの設定
    tokenizer.decoder = decoders.Sequence([
        decoders.ByteFallback(),
        decoders.Metaspace()
    ])
    if byte_fallback:
        tokenizer.add_special_tokens(['<0x{i:02X}>' for i in range(0, 256)])
    return tokenizer

def create_tokenizer(input_files, vocab, save_path, aargs):
    add_list = aargs['add_list']
    if add_list is not None:
        adhoc.notice('語彙追加', add_list)
        vocab = add_vocab_score(vocab, add_list)

    remove_list = aargs['remove_list']
    if remove_list is not None:
        adhoc.notice('語彙削除', add_list)
        vocab = remove_vocab_score(vocab, remove_list)

    tokenizer = unigram_tokenizer(vocab, byte_fallback=True)
    save_tokenizer(tokenizer, save_path=save_path)
    token_fraction(input_files, save_path)
    aargs.saved(save_path, f"Tokenizerの保存パス by save_path='{save_path}'")

def save_tokenizer(tokenizer, 
                   save_path=None, 
                   text='🦊 token化、大丈夫？'):
    # FastTokenizerの作成
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        eos_token=EOS_TOKEN,
        mask_token=MASK_TOKEN,
    )
    print(fast_tokenizer.get_vocab())
    fast_tokenizer.save_pretrained(save_path)
    print(text)
    encoded = fast_tokenizer.encode(text)
    print(len(encoded), encoded)
    decoded = fast_tokenizer.decode(encoded)
    print(decoded)

###

def train_unigram(input_files, aargs):
        tokenizer = unigram_tokenizer(byte_fallback=False)

        # トレーナーの設定
        trainer = trainers.UnigramTrainer(
            vocab_size=aargs['vocab_size|=32000'],
            n_sub_iterations=aargs['n_sub_iterations|=2'],
            max_piece_length=aargs['max_piece_length|=16'],
            unk_token=UNK_TOKEN,
            special_tokens=[EOS_TOKEN, UNK_TOKEN, MASK_TOKEN] 
        )

        with adhoc.start_timer() as timer:
            adhoc.notice('Unigramモデルの訓練', options=aargs)
            # トークナイザーのトレーニング
            tokenizer.train(input_files, trainer)
            timer.notice('お疲れ様')
        # 再度、作る？
        vocab = extract_vocab_score(tokenizer)
        return vocab

def train_unigram_cli(**kwargs):
    with adhoc.aargs_from(**kwargs) as aargs:
        files = list_filenames(aargs['files|!!'])
        save_path = aargs['save_path|=unigram_tokenizer']
        with TemporaryFiles(files, aargs['wakachi|=False'], merge=False) as temp:
            vocab = train_unigram(temp.input_files, aargs)
            create_tokenizer(temp.input_files, vocab, save_path, aargs)

def train_spm(input_files, aargs):
    kwargs = adhoc.extract_kwargs(spm.SentencePieceTrainer.train, 
                                aargs, 
                                excludes=['save_path', 
                                            'files', 
                                            'use_wakachi', 
                                            'test_sentence'])
    kwargs['input'] = input_files[0]
    prefix = kwargs['model_prefix']
    try:
        with adhoc.start_timer() as timer:
            adhoc.notice('SentencePieceモデルの訓練', options=kwargs)
            spm.SentencePieceTrainer.train(**kwargs)
            timer.notice('お疲れ様')
    except OSError as e:
        adhoc.notice('オプションが間違っている可能性が大', 
                    _read_here='https://github.com/google/sentencepiece/blob/master/doc/options.md')
        raise e
    aargs.saved(f'{prefix}.model', f"SentencePieceモデル model_prefix='{prefix}'")
    aargs.saved(f'{prefix}.vocab', 'SentencePiece語彙')
    vocab = extract_vocab_score_from_spiece(f'{prefix}.model')
    return vocab

def train_spm_cli(**kwargs):
    import sentencepiece as spm
    kwargs = dict (
        #input='progtext_all.txt',  #data.txt
        model_prefix='spiece',  #ngmodel
        vocab_size=32000,
        # num_threads=72,
        # train_extremely_large_corpus=True,
        normalization_rule_name='identity',
        user_defined_symbols=['\n'],
        max_sentencepiece_length=8, # 日本語は最大長8
        split_digits=True,
        byte_fallback=True,
        split_by_whitespace=True, # モデル作成時は空白で区切る
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        pad_id=-1, eos_id=0, unk_id=1, bos_id=-1,
        #control_symbols = ['<pad>', '</s>', '<unk>', '…'],
    ) | kwargs
    with adhoc.aargs_from(**kwargs) as aargs:
        files = list_filenames(aargs['files|!!'])
        use_wakachi = aargs['use_wakachi|=False']
        save_path = aargs['save_path|=spm_tokneizer']
        with TemporaryFiles(files, use_wakachi) as temp:
            kwargs = adhoc.extract_kwargs(spm.SentencePieceTrainer.train, 
                                        aargs, 
                                        excludes=['save_path', 'files', 'use_wakachi', 
                                                    'add_list', 'remove_list',
                                                    'test'])
            vocab = train_spm(temp.input_files, aargs)
            create_tokenizer(temp.input_files, vocab, save_path, aargs)

class TemporaryFiles(object):
    def __init__(self, files, use_wakachi, merge=True):
        self.merge = merge
        self.input_files = []
        self.remove_files = []
        if use_wakachi:
            self.convert_sudachi(files)
        else:
            self.convert_nop(files)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for file in self.remove_files:
            os.remove(file)

    def get_output_stream(self, seq: int, file=None):
        if self.merge:
            if file is not None:
                return file
        else:
            if file is not None:
                file.close()
        output_file = f'input_{os.getpid()}_{seq}.txt'
        self.input_files.append(output_file)
        if not os.path.exists(output_file):
            self.remove_files.append(output_file)
        return open(output_file, 'w', encoding='utf-8')

    def convert_nop(self, files):
        fout = None
        for i, file in enumerate(files):
            fout = self.get_output_stream(i, file=fout)
            with open(file, "r", encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line + "\n")
        if fout is not None:
            fout.close()

    def convert_sudachi(self, files):
        try:
            from sudachipy import tokenizer
            from sudachipy import dictionary
        except ModuleNotFoundError as e:
            print(e)
            print('pip install sudachipy sudachidict_core')
            raise e

        # 辞書の準備
        tokenizer_obj = dictionary.Dictionary().create()

        # 分かち書きを行う関数
        def wakachi(text):
            mode = tokenizer.Tokenizer.SplitMode.C  # 分割モードを選択（A, B, Cがある）
            text = text[:12000]
            tokens = tokenizer_obj.tokenize(text, mode)
            return " ".join([t.surface() for t in tokens])

        # ファイルを読み込んで分かち書きし、結果を書き出す
        fout = None
        for i, file in enumerate(files):
            fout = self.get_output_stream(i, file=fout)
            with open(file, "r", encoding='utf-8') as fin:
                for line in fin:
                    wakachi_line = wakachi(line.strip())
                    fout.write(wakachi_line + "\n")
        if fout is not None:
            fout.close()

PERCENTILES = [.1,.25,.33,.5,.67,.75,.8,.9,.95,.99]

def token_fraction(files, save_path):
    from transformers import AutoTokenizer
    import pandas as pd
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    text_lengths = []
    token_lengths = []
    fractions = []
    with open(files[0], "r", encoding='utf-8') as fin:
        for line in fin:
            text_length = len(line)
            if text_length > 40:
                tokens = tokenizer.encode(line)
                token_length = len(tokens)
                text_lengths.append(text_length)
                token_lengths.append(token_length)
                fractions.append(token_length/text_length)
    df = pd.DataFrame({'token': token_lengths, 'text': text_lengths, 'token/text': fractions})
    print(df.describe(percentiles=PERCENTILES))
    df.to_csv(f'{save_path}/token_fraction.csv', index=False)
    with open(f'{save_path}/token_fraction_describe.txt', 'w') as w:
        print(df.describe(percentiles=PERCENTILES), file=w)


def train_bpe_cli(**kwargs):
    with adhoc.aargs_from(**kwargs) as aargs:
        files = list_filenames(aargs['files|!!'])
        save_path = aargs['save_path']

        # トークナイザーの初期化
        tokenizer = Tokenizer(models.BPE())
        if aargs['bytelevel|byte_level|=True']:
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
        else:
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.UnicodeScripts(),
                pre_tokenizers.Whitespace()
            ])
            # デコーダーの設定
            tokenizer.decoder = decoders.WordPiece(prefix="##")

        # トレーナーの設定
        trainer = trainers.BpeTrainer(
            vocab_size=aargs['vocab_size|=32000'],
            min_frequency=aargs['min_frequency|=2'],
            special_tokens=[EOS_TOKEN, UNK_TOKEN, MASK_TOKEN], 
        )
        with adhoc.start_timer() as timer:
            adhoc.notice('BPEモデルの訓練を始めます', options=aargs)
            # トークナイザーのトレーニング
            tokenizer.train(files, trainer)
            timer.notice('お疲れ様す。')

        if save_path:
            save_tokenizer(tokenizer, save_path=save_path)
            token_fraction(files, save_path)
