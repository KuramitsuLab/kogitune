import evaluate
import os
import math 
import numpy as np
from scipy import stats

from .commons import *

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# =====================
# Base Class
# =====================

class Metric(object):
    """
    Base class for evaluators that use a model to obtain answers for generated prompts,
    evaluate them based on specified metrics, and calculate scores.
    """
    
    def __init__(self, name, **kwargs):
        self.name = name
        self.required_key = 'output'

    def __repr__(self):
        return self.name

    def eval_score(self, record:dict)->float:
        return 0.0

    def calc_scores(self, scores: np.ndarray, results: dict):
        return results

    def evaluate(self, result_list, force_eval=False):
        scores = []
        for record in adhoc.tqdm(result_list, desc=f'{self.name}'):
            if force_eval or self.name not in record:
                if self.required_key in record:
                    record[self.name] = self.eval_score(record)
            if self.name in record:
                scores.append(record[self.name])
        if len(scores) == 0:
            adhoc.notice(f'{self.name} スコアが一つもないよ')
            return None
        scores = np.array(scores)
        results = {
            'model': '', 
            'data': '', 
            'metric': self.name, 
            'mean': round(scores.mean(), 3), 
        }
        results = self.calc_scores(scores, results)
        if 'scores' not in results: 
            results['scores'] = list(round(v, 2) for v in scores)
        return results
        #         'scores': 
        # }
    
class metric_perplexity(Metric):
    def __init__(self, **kwargs):
        super(Metric).__init__('perplexity', **kwargs)
        self.required_key = 'loss'

    def eval_score(self, record:dict) -> float:
        return math.exp(record['loss'])

    def calc_scores(self, scores, results):
        results['std_dev'] = round(scores.std(), 3) 
        results['tmean'] = round(stats.trim_mean(scores, proportiontocut=0.1), 3),  # 上下10%をトリム
            
class metric_exact_match(Metric):
    """
    コード評価用Evaluatorクラス。HuggingFaceのevaluate-metric/code_evalを使用してスコアを算出する。
    """

    def __init__(self, **kwargs):
        super(Metric).__init__('exact_match', **kwargs)
        self.strict = kwargs.get('strict', True)

    def exact_match(self, output, reference):
        if (self.strict and reference == output) or reference in output:
            return 1
        return 0

    def eval_score(self, record:dict) -> float:
        outputs = listfy(record['output'])
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
        super(Metric).__init__(f'pass@{self.k}')
        self.required_key = 'output'
        self.tool = evaluate.load('code_eval')  # code_eval

    def eval_score(self, record):
        test_cases = [record['test']]
        extracted_code = [humaneval_extract(record['input'], x) for x in listfy(record['output'])]
        record['generated_code'] = extracted_code
        candidates = [extracted_code]
        pass_at_k, results = self.tool.compute(references=test_cases, predictions=candidates, k=[self.k])
        record[f'{self.name}_results'] = results
        return pass_at_k[self.name]

metric_pass_at_1 = metric_pass_at_k

class metric_f1(Metric):
    def __init__(self, **kwargs):
        super(Metric).__init__('perplexity', **kwargs)
        self.required_key = 'loss'

    def eval_score(self, record:dict) -> float:
        return math.exp(record['loss'])

    def calc_scores(self, scores, results):
        results['std_dev'] = round(scores.std(), 3) 
        results['tmean'] = round(stats.trim_mean(scores, proportiontocut=0.1), 3),  # 上下10%をトリム

from .similarites import (
    jaccard_similarity, cosine_similarity, 
    bleu_score, rouge_l, levenshtein_similarity, 
    simple_tokenize,
)

class metric_editsim(Metric):
    def __init__(self, **kwargs):
        super(Metric).__init__('editsim', **kwargs)

    def eval_score(self, record:dict) -> float:
        reference = record['reference']
        record['scores'] = [levenshtein_similarity(candidate, reference) 
                          for candidate in listfy(record['output'])]
        scores = np.array(record['scores'])
        return scores.mean()

    def calc_scores(self, scores, results):
        results['std_dev'] = round(scores.std(), 3) 


class metric_jaccard(Metric):
    def __init__(self, **kwargs):
        super(Metric).__init__('jaccard', **kwargs)
        self.tokenize = simple_tokenize

    def eval_score(self, record:dict) -> float:
        reference = record['reference']
        scores = np.array([jaccard_similarity(candidate, reference, tokenize=self.tokenize) 
                          for candidate in listfy(record['output'])])
        return scores.mean()

    def calc_scores(self, scores, results):
        results['std_dev'] = round(scores.std(), 3) 

class metric_cosine(Metric):
    def __init__(self, **kwargs):
        super(Metric).__init__('cosine', **kwargs)
        self.tokenize = simple_tokenize

    def eval_score(self, record:dict) -> float:
        reference = record['reference']
        scores = np.array([cosine_similarity(candidate, reference, tokenize=self.tokenize) 
                          for candidate in listfy(record['output'])])
        return scores.mean()

    def calc_scores(self, scores, results):
        results['std_dev'] = round(scores.std(), 3) 


"""
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

def evaluate_metric(result_list, metric_path, force_eval=False):
    metric_name, metric_args = adhoc.parse_path_args(metric_path)
    name = metric_name.replace('@', '_at_')
    name = f'metric_{name}'
    if name not in globals():
        adhoc.notice(f'{metric_name}が見つかりません')
        return None
    metric = globals()[name](**metric_args)
    return metric.evaluate(result_list, force_eval=force_eval)
    
