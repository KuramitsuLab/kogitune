import os
import json

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers, decoders

import sentencepiece as spm

import kogitune.adhocs as adhoc
from kogitune.stores.files import list_filenames

def is_model_bpe(model_path):
    # モデルファイルを読み込む
    with open(model_path, 'rb') as f:
        model_data = f.read()
    # モデルタイプを確認
    if b'type: BPE' in model_data:
        return True
    return False

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
]

def extract_vocab_score_from_spiece(model_path):
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(model_path)
    vocab = [(sp_model.id_to_piece(id), sp_model.get_score(id)) for id in range(sp_model.GetPieceSize())]
    return vocab

def extract_vocab_score(tokenizer):
    state = json.loads(bytes(tokenizer.model.__getstate__()).decode('utf-8'))
    return clean_vocab(state['vocab'], remove_meta=False)

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
    # tokenizer.decoder = decoders.Metaspace()
    if byte_fallback:
        tokenizer.add_special_tokens(['<0x{i:02X}>' for i in range(0, 256)])
    return tokenizer

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

def train_unigram_cli(**kwargs):
    with adhoc.aargs_from(**kwargs) as aargs:
        files = list_filenames(aargs['files|!!'])
        save_path = aargs['save_path']

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
            adhoc.notice('UnigramModelモデルの訓練', options=aargs)
            # トークナイザーのトレーニング
            tokenizer.train(files, trainer)
            timer.notice('お疲れ様')
        
        # 再度、作る？
        vocab = extract_vocab_score(tokenizer)
        tokenizer = unigram_tokenizer(vocab, byte_fallback=True)
        save_tokenizer(tokenizer, save_path=save_path)

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
        save_path = aargs['save_path']
        test=aargs['test|=🦊お疲れ様！']
        with TemporaryFiles(files, use_wakachi) as temp:
            kwargs = adhoc.extract_kwargs(spm.SentencePieceTrainer.train, 
                                        aargs, 
                                        excludes=['save_path', 
                                                    'files', 
                                                    'use_wakachi', 
                                                    'test_sentence'])
            kwargs['input'] = temp.input_file
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
            if save_path:
                vocab = extract_vocab_score_from_spiece(f'{prefix}.model')
                tokenizer = unigram_tokenizer(vocab)
                save_tokenizer(tokenizer, save_path=save_path)

class TemporaryFiles(object):
    def __init__(self, files, use_wakachi):
        self.input_file = f'input_{os.getpid()}.txt'
        self.remove_input_file = not os.path.exists(self.input_file)
        if use_wakachi:
            generate_sudachi_file(files, output_file=self.input_file)
        else:
            generate_input_file(files, output_file=self.input_file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.remove_input_file:
            os.remove(self.input_file)

def generate_sudachi_file(files, output_file='temp'):
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
    with open(output_file, "w", encoding="utf-8") as fout:
        for file in files:
            with open(file, "r", encoding='utf-8') as fin:
                for line in fin:
                    wakachi_line = wakachi(line.strip())
                    fout.write(wakachi_line + "\n")
    return output_file

def generate_input_file(files, output_file):
    with open(output_file, "w", encoding="utf-8") as fout:
        for file in files:
            with open(file, "r", encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line + "\n")
    return output_file


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

"""

def convert_fast_tokenizer(model_path, save_path='new_tokenizer'):
    if is_model_bpe(model_path):
        print('@BPE')
        from tokenizers import SentencePieceBPETokenizer
        # sentencepieceモデルを読み込む
        tokenizer = SentencePieceBPETokenizer.from_file(model_path)
    else:
        print('@Unigram')
        from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(model_path)
        vocab = [(sp_model.id_to_piece(id), sp_model.get_score(id)) for id in range(sp_model.GetPieceSize())]
        tokenizer = Tokenizer(models.Unigram(vocab, 2, byte_fallback=True))
        # # プレトークナイザーの設定
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        # # デコーダーの設定
        tokenizer.decoder = decoders.Metaspace()

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        byte_fallback=True,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="…",
    )
    print(fast_tokenizer.get_vocab())
    fast_tokenizer.save_pretrained(save_path)
    test_tokenizer(fast_tokenizer)


def remove_prefix_space(w, type=1, score=0.0):
    if type == 1:
        if w.startswith('▁') and len(w) > 1:
            return w[1:]
    return None

def replace_spm(model_path, replace_fn):
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
    from sentencepiece import sentencepiece_model_pb2 as pb2
    m = pb2.ModelProto()
    with open(model_path, 'rb') as f:
        m.ParseFromString(f.read())
    voc = set(piece.piece for piece in m.pieces)
    for id, piece in enumerate(m.pieces):
        replaced = replace_fn(piece.piece, type=piece.type, score=piece.score)
        if replaced is not None:
            #print(id, piece.piece, '=>', replaced, replaced in voc)
            if replaced not in voc:
                piece.piece = replaced
    with open(model_path, 'wb') as f:
        f.write(m.SerializeToString())

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
        pad_id=0, eos_id=1, unk_id=2, bos_id=-1,
        #control_symbols = ['<pad>', '</s>', '<unk>', '…'],
    ) | kwargs
    with adhoc.aargs_from(**kwargs) as aargs:
        files = list_filenames(aargs['files|!!'])
        use_wakachi = aargs['use_wakachi|=False']
        save_path = aargs['save_path']
        test_sentence=aargs['test_sentence|=🦊お疲れ様！']
        kwargs = adhoc.extract_kwargs(spm.SentencePieceTrainer.train, 
                                      aargs, 
                                      excludes=['save_path', 
                                                'files', 
                                                'use_wakachi', 
                                                'test_sentence'])
        input_file = f'input_{os.getpid()}.txt'
        remove_input_file = not os.path.exists(input_file)
        if use_wakachi:
            generate_wakachi_file(files, output_file=input_file)
        else:
            generate_input_file(files, output_file=input_file)
        kwargs['input'] = input_file
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
        finally:
            if remove_input_file:
                os.remove(input_file)

        if use_wakachi:
            replace_spm(f'{prefix}.model', remove_prefix_space)
        aargs.saved(f'{prefix}.model', f"SentencePieceモデル model_prefix='{prefix}'")
        aargs.saved(f'{prefix}.vocab', 'SentencePiece語彙')
        if save_path:
            convert_fast_tokenizer(f'{prefix}.model', save_path=save_path)
            aargs.saved(save_path, f"生成されたTokenizerのパス save_path='{save_path}'")

"""




