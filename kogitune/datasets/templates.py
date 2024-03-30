import re

from .local_commons import *
from .da import da
import numpy as np
import datasets

# パターンに一致するすべての部分を検索

SEC_IN = '### instruction\n'
SEC_OUT = '\n### output\n'

#まず、基本的な正規表現パターンを設計します。
# {}内の文字をマッチングするためには、{([^}]+)}のようなパターンを使用します。
# これは、{}で囲まれた、}を含まない1文字以上の任意の文字列にマッチします。
# 次に、抽出した文字列からフォーマット指定を除いてキー名のみを取り出すために、さらに処理を加えます。

def extract_format_keys(s):
    # 正規表現を使って{}で囲まれた部分を全て抽出
    matches = re.findall(r'\{([^}]+)\}', s)
    # フォーマット指定を考慮して、コロン以前の部分（キー名）のみを取り出す
    keys = [match.split(':')[0] for match in matches]
    # 重複を除いたリストを返す
    return list(set(keys))

## テスト
#test_string = "価格は{price:.2f}円で、数量は{quantity}個です。割引率は{discount:.2%}です。"
#extract_format_keys(test_string)



class TemplateProcessor:
    def __init__(self, **kwargs):
        self.prompt = kwargs.get('prompt', '')
        self.output = kwargs.get('output', kwargs.get('reference', ''))
        self.template_keys = extract_format_keys(self.prompt + self.output)
        self.SEC_IN = kwargs.get('section_in', SEC_IN)
        self.SEC_OUT = kwargs.get('section_out', SEC_OUT)
        da_policy = kwargs.get('da_policy', 'dynamic')
        if da_policy == 'dynamic':
            self.enforce_da = True
            self.random_choice = True
        elif da_policy == 'notion':
            self.enforce_da = False
        else: # static
            self.enforce_da = True
            self.random_choice = False
        self.options = kwargs

    def create_prompt(self, sample:dict):
        prompt = self.prompt.format(**sample)
        if self.enforce_da:
            prompt = da(prompt, random_choice=self.random_choice)
        return prompt
    
    def create_output(self, sample:dict):
        output = self.output.format(**sample)
        if self.enforce_da:
            output = da(output, random_choice=self.random_choice)
        return output

    def create_reference(self, sample:dict):
        return self.create_output(sample)

    def load_to_sample_list(self, datalist, sample_list):
        assert len(datalist) == len(sample_list)
        for i in range(len(datalist)):
            source = datalist[i]
            sample = sample_list[i]
            if 'input' not in sample:
                sample['input'] = self.create_prompt(source)
            if 'reference' not in sample:
                sample['reference'] = self.create_output(source)

    def create_instruction(self, sample:dict):
        prompt = self.create_prompt(sample)
        output = self.create_output(sample)
        return f'{self.SEC_IN}{prompt}{self.SEC_OUT}{output}'

    def formatting_for_trainer(self, example):
        output_texts = []
        sample = {}
        key = self.template_keys[0]
        for i in range(len(example[key])):
            for key in self.template_keys:
                sample[key] = example[key][i]
            output_texts.append(self.create_instruction(sample))
        return output_texts

    def test_template(self, sample:dict, verbose=True):
        try:
            prompt = self.create_prompt(sample)
        except KeyError as e:
            self.report_KeyError(e)
            adhoc.fatal('データセットとプロンプトテンプレートが一致しないよ')
        try:
            reference = self.create_output(sample)
        except KeyError as e:
            self.report_KeyError(e)
            adhoc.fatal('データセットと出力(参照)テンプレートが一致しないよ')
        if verbose:
            adhoc.print(f'プロンプトを確認してね\n{prompt}\n（期待される出力）\n{reference}')
 
    def report_KeyError(self, e):
        try:
            matches = re.findall(r"'(.*?)'",  f'{e}')
            key = matches[0]
            adhoc.perror(f'テンプレートの{key}がデータセットにないのが原因だよ')
        except:
            adhoc.perror(f'テンプレートのキーがデータセットにないのが原因だよ')

    def calc_length(self, dataset:List[dict], tokenizer=None, return_max_new_tokens=False, q=95):
        import pandas as pd
        tokenizer = tokenizer or configurable_tokenizer()
        total=[]
        prompt=[]
        output=[]
        for sample in dataset:
            prompt_length = len(tokenizer.encode(self.create_prompt(sample)))
            output_length = len(tokenizer.encode(self.create_output(sample)))
            prompt.append(prompt_length)
            output.append(output_length)
            total.append(prompt_length+output_length)
        adhoc.notice('データセットのトークン長を調べてみたよ')
        data = {'prompt': prompt, 'output': output, 'total': total}
        print(pd.DataFrame(data).describe(percentiles=[.8, .9, .95]))
        if q < 1:
            q = int(q*100)
        if return_max_new_tokens:
            max_new_tokens = max(output)
            if max_new_tokens < 512:
                return max_new_tokens
            return int(np.percentile(output, q=q))+1
        return int(np.percentile(total, q=q))+1

    # def extract(self, text:str) -> str:
    #     if isinstance(text, list):
    #         return [self.extract(t) for t in text]
    #     if self.begin or self.end:
    #         lines = text.splitlines()
    #         extracted = []
    #         inclusion = False if self.begin else True
    #         for line in lines:
    #             if self.end and line.startswith(self.end):
    #                 break
    #             if self.begin and line.startswith(self.begin):
    #                 inclusion = True
    #             if inclusion:
    #                 extracted.append(line)
    #         return '\n'.join(extracted)
    #     return text

    def filter(self, dataset, tokenizer, max_length=None, min_length=None, head=None, return_as_dict=False):
        sample_list = []
        for sample in dataset:
            prompt_length = len(tokenizer.encode(self.create_prompt(sample)))
            total_length = len(tokenizer.encode(self.create_output(sample))) + prompt_length
            if max_length:
                if prompt_length > max_length or total_length > max_length:
                    continue
            if min_length and total_length < min_length:
                continue
            if head and len(sample_list) >= head:
                break
            sample_list.append(sample)
        if return_as_dict:
            return sample_list
        data_columns = {key: [dic[key] for dic in sample_list] for key in sample_list[0]}
        return datasets.Dataset.from_dict(data_columns)

###

def has_schema(data: dict, keys:str):
    for key in keys.split('|'):
        if key not in data:
            return False
    return True

def guess_template(sample: dict):
    if has_schema(sample, 'instruction|input|output'):
        # Alpaca形式
        return {
            "prompt": "{instruction}\n{input}",
            "reference": "{output}",
        }
    if has_schema(sample, 'prompt|test|entry_point'):
        # HumanEval
        return {
            "prompt": "{prompt}",
            "reference": "\n{test}\n\ncheck({entry_point})\n",
        }
    if has_schema(sample, 'question|choice0|choice1|choice2|choice3|choice4|label'):
        # JCommonSenseQA
        return {
            "prompt": "{question}\n[選択肢|Choice]: [@(0)|0.|[0]] {choice0} [@(1)|1.|[1]] {choice1} [@(2)|2.|[2]] {choice2} [@(3)|3.|[3]] {choice3} [@(4)|4.|[4]] {choice4}\n",
            "reference": "{label}",
        }
    if has_schema(sample, 'in|out'):
        # KOGITUNE 標準形式
        return {
            "prompt": "{in}",
            "reference": "{out}",
        }
    return None

def load_template(**kwargs):
    """
    テンプレートエンジンをロードする
    :args sample: sampleを渡せば、推論してくれる。
    """
    sample = AdhocArguments.pop_from_kwargs('sample', kwargs)
    with AdhocArguments.from_main(**kwargs) as aargs:
        template_args = aargs['template_config|template_args|template']
        if template_args is None:
            template_args = dict(
                prompt = aargs['prompt_template|prompt'],
                reference = aargs['reference_template|reference'],
            )
        if template_args.get('prompt', None) is None:
            template_args = None
            if isinstance(sample, dict):
                template_args = guess_template(sample)
        if template_args is None:
            adhoc.fatal('templateの指定がないよ！このあと、何も出力されないよ')
        template = TemplateProcessor(**template_args)
    return template


