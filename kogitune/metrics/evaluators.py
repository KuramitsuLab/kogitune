import evaluate
import os
import re
import numpy as np

from .local_utils import *

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# =====================
# Base Class
# =====================

class Metric(object):
    """
    Base class for evaluators that use a model to obtain answers for generated prompts,
    evaluate them based on specified metrics, and calculate scores.
    """
    
    def __init__(self, **kwargs):
        self.name = 'nop'

    def __repr__(self):
        return self.name

    def eval_score(self, record:dict)->float:
        return 0.0

    def evaluate(self, result_list):
        scores = []
        for record in configurable_tqdm(result_list, desc=f'{self.name}'):
            if self.name not in record:
                record[self.name] = self.eval_score(record)
            scores.append(record[self.name])
        if len(scores) == 0:
            adhoc.warn(f'{self.name} スコアが一つもないよ')
            return None
        scores = np.array(scores)
        np.set_printoptions(precision=2)
        print(scores, scores.mean(), scores.min(), scores.max())
        return {'metric': self.name, 'mean': scores.mean(), 'scores': list(scores)}
            
class metric_exact_match(Metric):
    """
    コード評価用Evaluatorクラス。HuggingFaceのevaluate-metric/code_evalを使用してスコアを算出する。
    """

    def __init__(self, strict=True, **kwargs):
        self.name = f'exact_match'
        self.strict = strict

    def exact_match(self, output, reference):
        if (self.strict and reference == output) or reference in output:
            return 1
        return 0

    def eval_score(self, record:dict) -> float:
        outputs = record['outputs']
        reference = record['reference']
        scores = np.array([self.exact_match(output, reference) for output in outputs])
        return scores.mean()


# HumanEval pass@1
#

def humaneval_extract(prompt, generated_text):
    # if generated_text == '':
    #     return 'Empty Code!!'
    stop_sequences=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]
    min_stop_index = len(generated_text)
    for seq in stop_sequences:
        stop_index = generated_text.find(seq)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return prompt + "\n" + generated_text[:min_stop_index]


class metric_pass_at_k(Metric):
    """
    コード評価用Evaluatorクラス
    HuggingFaceのevaluate-metric/code_evalを使用してスコアを算出する。
    """

    def __init__(self, **kwargs):
        self.k = kwargs.get('k', 1)        
        self.name = f'pass@{self.k}'
        self.tool = evaluate.load('code_eval')  # code_eval

    def eval_score(self, record):
        test_cases = [record['reference']]
        extracted_code = [humaneval_extract(record['input'], x) for x in record['outputs']]
        record['generated_code'] = extracted_code
        candidates = [extracted_code]
        pass_at_k, results = self.tool.compute(references=test_cases, predictions=candidates, k=[self.k])
        record[f'{self.name}_results'] = results
        return pass_at_k[self.name]

metric_pass_at_1 = metric_pass_at_k

"""
# 日本語用のtokenizer
# Python: 正規表現による簡易版形態素解析
# https://qiita.com/kinoshita_yuri/items/e15f143981f1616994ed
    
def tokenize_japaneses(text):
    pJA = re.compile(r"/|[A-Z]+|[a-z]+|[ァ-ンー]+|[ぁ-ん-]+|[ァ-ヶ]+|[一-龍]+|[。、]|/")
    text_m = []
    m = pJA.findall(text)
    for row in m:
        if re.compile(r'^[あ-ん]+$').fullmatch(row):
            if row[0] in 'はがのにへともでを':
                prefix = row[0]
                token = row[1:]
                text_m.append(prefix)
                if (len(token) > 0):
                    text_m.append(token)
            elif row[-2:] in 'のでからまで':
                token = row[0:-2]
                suffix = row[-2:]
                text_m.append(token)
                text_m.append(suffix)
            elif row[-1:] in 'もはがでを':
                token = row[0:-1]
                suffix = row[-1:]
                text_m.append(token)
                text_m.append(suffix)
            else:
                text_m.append(row)
        else:
            text_m.append(row)
    return text_m

class BLEUEvaluator(Evaluator):
    # def calculate(self, dataset, record):

    #     # BLEUメトリック用のデータ準備
    #     references = [[d['reference'].split()] for d in dataset]  # リストのリストとして分割された参照文
    #     candidates = [d['model_output'].split() for d in dataset]  # 分割された予測文のリスト
    #     # BLEU スコアを計算
    #     score = self.metric.compute(predictions=candidates, references=references)['bleu']

    #     for data in dataset:
    #         data['bleu_score'] = score

    #     return score, dataset

    
    def score_item(self, data):
        predictions = [data['extracted_result']]
        references = [[data['reference']]]
        if output_lang == 'ja':
            item_score = self.metric.compute(predictions=predictions, references=references, tokenier=tokenize_ja, smooth=True)['bleu']
        else:
            item_score = self.metric.compute(predictions=predictions, references=references, smooth=True)['bleu']
        self.item_scores.append(item_score)
        return item_score        

class F1Evaluator(Evaluator):
    # def calculate(self, dataset, record):
        
    #     # F1スコアの計算に必要な正解ラベルと予測ラベルのリストを準備
    #     references = [d['reference'] for d in dataset]
    #     candidates = [d['model_output'] for d in dataset]
    #     # F1スコアを計算
    #     score = self.metric.compute(predictions=candidates, references=references)["f1"]
    #     # `score` には通常、precision, recall, f1 のキーが含まれている
    #     #f1_score = score['f1']
    #     #score = f1_score

    #     for data in dataset:
    #         data['f1_score'] = score

    #     return score, dataset
    def item_calculate(self, data, record, output_lang):
        return None
    
    def total_calculate(self, dataset, record, output_lang):
        predictions = [int(data['model_output']) for data in dataset]
        references = [int(data['reference']) for data in dataset]
        total_score = self.metric.compute(predictions=predictions, references=references)["f1"]
        return total_score
"""

#######################

def evaluate_metric(result_list, metric_path):
    metric_name, metric_args = parse_path_arguments(metric_path)
    name = metric_name.replace('@', '_at_')
    name = f'metric_{name}'
    if name not in globals():
        adhoc.warn(f'{metric_name}が見つかりません')
        return None
    metric = globals()[name](**metric_args)
    return metric.evaluate(result_list)
    
