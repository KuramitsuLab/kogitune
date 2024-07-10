import os
import kogitune.adhocs as adhoc

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
    from kogitune.stores.files import list_filenames
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
        remove_input_file = os.path.exists(input_file)
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
            aargs.saved(save_path, f"変換されたTokenizerのパス save_path='{save_path}'")
