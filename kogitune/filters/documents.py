from typing import Optional
import unicodedata
from .filters import TextFilter, ComposeFilter, adhoc

class UnicodeNormalization(TextFilter):
    """
    Unicode正規化フィルターする
    """
    def __init__(self, *args, **kwargs):
        """
        Unicode正規化フィルタを作る
        """
        super().__init__(*args, **kwargs)
        adhoc.aargs_from(**kwargs).record(
            'form|=NFKC',
            field=self, dic=self.rec,
        )

    def __call__(self, text: str, record: dict) -> Optional[str]:
        return unicodedata.normalize(self.form, text)

class DuplicatedLineFilter(TextFilter):
    """
    重複した行を取り除く
    """
    def __init__(self, *args, **kwargs):
        """
        重複した行を取り除くフィルタを作る
        :param prefix_length: 重複をチェックする先頭の文字数
        """
        super().__init__(*args, **kwargs)
        adhoc.aargs_from(**kwargs).record(
            'prefix_length|=8',
            field=self, dic=self.rec,
        )

    def __call__(self, text: str, record: dict) -> Optional[str]:
        lines = ['']
        for line in text.split('\n'):
            prev = lines[-1]
            if self.prefix_length < len(line) < 80 and prev.startswith(line[:self.prefix_length]):
                if len(line) > len(prev):
                    lines[-1] = line
                continue
            if len(line.strip()) == 0 and prev == '':
                continue
            if 1 < len(prev) < 40 and not prev.endswith('。') and len(line) > 80 and line.count('。') > 1:
                if lines[-1] != '':
                    lines[-1] = f'\n{prev}'
            lines.append(line)
        return '\n'.join(lines[1:])


## 行単位の処理

class LineByLineFilter(ComposeFilter):
    """
    行単位の処理をするフィルター
    """
    def __init__(self, *filters, **kwargs):
        """
        重複した行を取り除くフィルタを作る
        :param prefix_length: 重複をチェックする先頭の文字数
        """
        super().__init__(*filters, **kwargs)
        adhoc.aargs_from(**kwargs).record(
            'separator|sep|=\n',
            field=self, dic=self.rec,
        )

    def __call__(self, text: str, record: dict) -> Optional[str]:
        lines = []
        for line in text.split(self.separator):
            for f in self.filters:
                line = f(line, record)
                if line is None:
                    break
            if line:
                lines.append(line)
            else:
                lines.append('')
        if len(lines) == 0:
            return None
        return self.separator.join(lines)

def chunk_text(text, max_length=200):
    """
    指定されたルールに基づいてテキストを分割する関数。
    
    Parameters:
    text (str): 分割するテキスト
    max_length: 分割する単位

    Returns:
    list: 分割されたテキストのリスト
    """
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        if line.strip() == "":
            # 空行が来たら分割
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0
        elif current_length + len(line) + 1 > max_length:
            # 今までの行がmax_length文字を超えたら分割
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_length = len(line)
        else:
            # 行を追加
            current_chunk.append(line)
            current_length += len(line) + 1

    # 最後のチャンクを追加
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

class ChunkFilter(ComposeFilter):
    """
    行単位の処理をするフィルター
    """
    def __init__(self, *filters, **kwargs):
        """
        重複した行を取り除くフィルタを作る
        :param prefix_length: 重複をチェックする先頭の文字数
        """
        super().__init__(*filters, **kwargs)
        adhoc.aargs_from(**kwargs).record(
            'max_chunk_length|chunk_length|=200',
            field=self, dic=self.rec,
        )

    def __call__(self, text: str, record: dict) -> Optional[str]:
        chunks = []
        for chunk in chunk_text(text, self.max_chunk_length):
            for filter in self.filters:
                chunk = filter(chunk, record)
                if chunk is None:
                    break
            if chunk:
                chunks.append(chunk)
            else:
                chunks.append('')
        if len(chunks) == 0:
            return None
        return '\n'.join(chunks)


