try:
    import hojichar

    _cleaner = hojichar.Compose([
    #    hojichar.document_filters.JSONLoader(key="text"),
        hojichar.document_filters.DocumentNormalizer(),
    #    hojichar.document_filters.AcceptJapanese(),
    #    hojichar.document_filters.DocumentLengthFilter(min_doc_len=0),
        hojichar.document_filters.DiscardAdultContentJa(), # アダルト用語除去
        hojichar.document_filters.DiscardAdultContentEn(), # 英語も除去
        hojichar.document_filters.DiscardDiscriminationContentJa(), # 差別用語
        hojichar.document_filters.DiscardViolenceContentJa() # 暴力用語
    ])

    def is_rejected_content(text: str) -> bool:
        doc = hojichar.Document(text)
        result = _cleaner.apply_filter(doc) # hojicharでフィルターする
        return result.is_rejected
except:
    import warnings

    def is_rejected_content(text):
        with warnings.catch_warnings():
            # このブロック内でのみ警告を一度だけ表示するように設定
            warnings.simplefilter('once', UserWarning)

            # ここで警告を出力（一度だけ表示される）
            warnings.warn('''もしコンテンツフィルタを使いたいなら
pip3 install hojichar''')
        return False

