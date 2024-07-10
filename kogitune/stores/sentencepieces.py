import os
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

def convert_fast_tokenizer(model_path, save_path='new_tokenizer'):
    import sentencepiece as spm
    from transformers import PreTrainedTokenizerFast
    if is_model_bpe(model_path):
        print('@BPE')
        from tokenizers import SentencePieceBPETokenizer
        # sentencepieceモデルを読み込む
        tokenizer = SentencePieceBPETokenizer.from_file(model_path)
        # FastTokenizerに変換
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    else:
        print('@Unigram')
        from transformers import T5Tokenizer
        fast_tokenizer = T5Tokenizer(model_path, extra_ids=0, additional_special_tokens=[])
    print(fast_tokenizer.get_vocab())
    fast_tokenizer.save_pretrained(save_path)

def generate_wakachi_file(files, output_file='temp'):
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

def spm_train_cli(**kwargs):
    import sentencepiece as spm
    kwargs = kwargs | dict (
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
        unk_id=2, pad_id=0, eos_id=1, bos_id=-1,
    )
    with adhoc.aargs_from(**kwargs) as aargs:
        files = list_filenames(aargs['files|!!'])
        use_wakachi = aargs['use_wakachi|=False']
        save_path = aargs['save_path']
        kwargs = adhoc.extract_kwargs(spm.SentencePieceTrainer.train, 
                                      aargs, 
                                      excludes=['save_path', 
                                                'files', 
                                                'use_wakachi', 
                                                'save_path'])
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

def test_tokenizer(tokenizer):
    # 使用例
    text = """\
自然言語処理は面白い技術です。
while True:
    a += 1 #😄
"""
    encoded = tokenizer.encode(text)
    print(len(encoded), encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)

def train_bpe_cli(**kwargs):
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    with adhoc.aargs_from(**kwargs) as aargs:
        files = list_filenames(aargs['files|!!'])
        save_path = aargs['save_path']

        # トークナイザーの初期化
        tokenizer = Tokenizer(models.BPE())

        if aargs['bytelevel|byte_level|=True']:
            # プレトークナイザーの設定（文字単位）
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

            # デコーダーの設定
            tokenizer.decoder = decoders.ByteLevel()
        else:
            # プレトークナイザーを文字レベルに設定
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.UnicodeScripts(),
                pre_tokenizers.Whitespace()
            ])

            # デコーダーの設定
            tokenizer.decoder = decoders.WordPiece(prefix="##")

            # # プレトークナイザーを文字レベルに設定
            # tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            #     pre_tokenizers.UnicodeScripts(),
            #     pre_tokenizers.Metaspace(replacement=" ")#, add_prefix_space=False)
            # ])

            # # デコーダーの設定
            # tokenizer.decoder = decoders.Metaspace(replacement=" ")#, add_prefix_space=False)

        # トレーナーの設定
        trainer = trainers.BpeTrainer(
            vocab_size=aargs['vocab_size|=32000'],
            min_frequency=aargs['min_frequency|=2'],
            special_tokens=["[PAD]", 
                            "[UNK]", 
                            "[CLS]", 
                            "[SEP]", 
                            "[MASK]"]
        )

        with adhoc.start_timer() as timer:
            adhoc.notice('BPEモデルの訓練を始めます', options=aargs)
            # トークナイザーのトレーニング
            tokenizer.train(files, trainer)
            timer.notice('お疲れ様す。')

        # FastTokenizerの作成
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        adhoc.print('語彙', fast_tokenizer.get_vocab())
        # トークナイザーの保存
        if save_path:
            fast_tokenizer.save_pretrained(save_path)
            aargs.saved(save_path, f"保存されたTokenizerのパス save_path='{save_path}'")

        test_tokenizer(fast_tokenizer)

def train_unigram_cli(**kwargs):
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

    with adhoc.aargs_from(**kwargs) as aargs:
        files = list_filenames(aargs['files|!!'])
        save_path = aargs['save_path']

        # トークナイザーの初期化
        tokenizer = Tokenizer(models.Unigram(byte_fallback=True))

        # プレトークナイザーの設定（文字単位）
        tokenizer.pre_tokenizer = pre_tokenizers.UnicodeScripts()

        # デコーダーの設定
        tokenizer.decoder = decoders.ByteLevel()

        # トレーナーの設定
        trainer = trainers.UnigramTrainer(
            vocab_size=aargs['vocab_size|=32000'],
            n_sub_iterations=aargs['n_sub_iterations|=2'],
            max_piece_length=aargs['max_piece_length|=16'],
#            seed=aargs['seed|=42'],
# 
            unk_token="[UNK]",
            special_tokens=["[PAD]", 
                            "[UNK]", 
                            "[CLS]", 
                            "[SEP]", 
                            "[MASK]"]
        )

        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
        )

        with adhoc.start_timer() as timer:
            adhoc.notice('UnigramModelモデルの訓練', options=aargs)
            # トークナイザーのトレーニング
            tokenizer.train(files, trainer)
            timer.notice('お疲れ様')

        # デコーダーの設定
        tokenizer.decoder = decoders.Metaspace()

        # FastTokenizerの作成
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        print(fast_tokenizer.get_vocab())
        # トークナイザーの保存
        if save_path:
            fast_tokenizer.save_pretrained(save_path)
            aargs.saved(save_path, f"保存されたTokenizerのパス save_path='{save_path}'")

        test_tokenizer(fast_tokenizer)
