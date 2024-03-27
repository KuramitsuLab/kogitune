from typing import List
import re
from .local_utils import *

# パターンに一致するすべての部分を検索

class TemplateProcessor:
    def __init__(self, prompt, reference, **kwargs):
        self.prompt = prompt
        self.reference = reference
        self.options = kwargs

    def create_prompt(self, data:dict):
        prompt = self.prompt.format(**data)
        return prompt
    
    def create_reference(self, data:dict):
        reference = self.reference.format(**data)
        return reference

    def create_instruction(self, data):
        prompt = self.prompt.format(**data)
        reference = self.reference.format(**data)
        if not prompt.endswith('\n'):
            return f'{prompt}\n{reference}'
        return f'{prompt}{reference}'

    def create_instruction(self, data):
        prompt = self.prompt.format(**data)
        reference = self.reference.format(**data)
        if not prompt.endswith('\n'):
            return f'{prompt}\n{reference}'
        return f'{prompt}{reference}'

    def test_template(self, record, verbose=True):
        try:
            prompt = self.create_prompt(record)
        except KeyError as e:
            self.report_KeyError(e)
            adhoc.fatal('データセットとプロンプトテンプレートが一致しないよ')
        try:
            reference = self.create_reference(record)
        except KeyError as e:
            self.report_KeyError(e)
            adhoc.fatal('データセットと参照テンプレートが一致しないよ')
        if verbose:
            adhoc.print(f'プロンプトを確認してね\n{prompt}\n===\n{reference}')
 
    def report_KeyError(self, e):
        try:
            matches = re.findall(r"'(.*?)'",  f'{e}')
            key = matches[0]
            adhoc.perror(f'テンプレートの{key}がデータセットにないのが原因だよ')
        except:
            adhoc.perror(f'テンプレートのキーがデータセットにないのが原因だよ')

    def calc_max_tokens(self, datalist:List[dict], extra_length=20):
        import pandas as pd
        tokenizer = configurable_tokenizer()
        max_length=[]
        max_tokens=[]
        for data in datalist:
            prompt_length = len(tokenizer.encode(self.create_prompt(data)))
            output_length = len(tokenizer.encode(self.create_reference(data)))
            max_length.append(prompt_length+output_length)
            max_tokens.append(output_length)
        print(pd.DataFrame({'max_new_tokens': max_tokens, 'max_length': max_length}).describe(percentiles=[.8, .9, .95]))
        return max(max_tokens) + extra_length, max(max_length) + extra_length

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

###

def has_schema(data: dict, keys:str):
    for key in keys.split('|'):
        if key not in data:
            return False
    return True

def guess_template(data: dict, aargs):
    IN = aargs['instruction_section|=### Instruction\n']
    OUT = aargs['output_section|=### Output\n']
    if has_schema(data, 'prompt|test|entry_point'):
        return {
            "prompt": "{prompt}",
            "reference": "\n{test}\ncheck({entry_point})\n",
        }
    if has_schema(data, 'question|choice0|choice1|choice2|choice3|choice4|label'):
        return {
            "prompt": IN + "{question}\n選択肢: (0) {choice0} (1) {choice1} (2) {choice2} (3) {choice3} (4) {choice4}\n" + OUT,
            "reference": "{label}",
        }
    return None

# =====================
# Utility Function
# =====================

def load_template(datalist:List[dict], aargs: AdhocArguments):
    template_args = aargs['template_config|template']
    if template_args is None:
        template_args = dict(
            prompt = aargs['prompt_template|prompt'],
            reference = aargs['reference_template|reference'],
            extract_begin = aargs['extract_begin'],
            extract_end = aargs['extract_end'],
        )
    if template_args.get('prompt', None) is None:
        template_args = guess_template(datalist[0], aargs)
    if template_args is None:
        adhoc.fatal('templateの指定がないよ！このあと、何もできないね')

    template = TemplateProcessor(**template_args)

    return template
    