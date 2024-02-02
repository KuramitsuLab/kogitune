import json
import unicodedata

from .base import TextFilter, ComposeFilter

class JSONConvertor(TextFilter):
    def __init__(self, target='text'):
        super().__init__(verbose=0)
        self.target = target

    def __call__(self, text):
        return json.loads(text)[self.target]

class JSONTemplateConvertor(TextFilter):
    def __init__(self, template:str):
        super().__init__(verbose=0)
        self.template = template

    def __call__(self, text):
        return self.template.format(**json.loads(text))

class UnicodeNormalization(TextFilter):

    def __init__(self, form = 'NFKC', verbose=0):
        super().__init__(verbose=verbose)
        self.form = form

    def filter(self, text):
        return unicodedata.normalize(self.form, text)

## 行単位の処理

class LineByLineFilter(ComposeFilter):
    """
    行単位の処理をするフィルター
    """
    def __init__(self, *filters, sep='\n'):
        """
        行単位の処理をするフィルターを作る
        :param sep: セパレータの調整
        """
        super().__init__(*filters)
        self.sep = sep

    def __call__(self, text):
        lines = []
        for line in text.split(self.sep):
            for f in self.filters:
                line = f(line)
                if line is None:
                    break
            if line:
                lines.append(line)
            else:
                lines.append('')
        if len(lines) == 0:
            return None
        return self.sep.join(lines)


class DuplicatedLineFilter(TextFilter):
    """
    重複した行を取り除く
    """
    def __init__(self, prefix_length=8, verbose=0):
        """
        重複した行を取り除くフィルタを作る
        :param prefix_length: 重複をチェックする先頭の文字数
        """
        super().__init__(verbose=verbose)
        self.prefix_length=prefix_length

    def filter(self, text):
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



