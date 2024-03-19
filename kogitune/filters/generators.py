from typing import List
import urllib.parse

from .base import *
from .documents import *
from .scores import MaxMinFilter

def urlencode(d:dict):
    return urllib.parse.urlencode(d)

def encode_arguments_without(path:str, args:dict, keys:list):
    if isinstance(path, str):
        if isinstance(keys, str):
            keys = keys.split('|')
        for key in keys:
            if key in args:
                del args[key]
        if len(args) > 0:
            return f'{path}?{urllib.parse.urlencode(args)}'
    return path

def maxmin(score, **kwargs):
    args = kwargs
    if 'min' in args and 'min_value' not in args:
        args['min_value'] = args.pop('min')
    if 'max' in args and 'max_value' not in args:
        args['max_value'] = args.pop('max')
    score_path = encode_arguments_without(score, args.copy(), 
                'min_value|max_value|record_key|histogram_sample|save_to|percentiles')
    return MaxMinFilter(score_path=score_path, **args)


def compose(*filters):
    if len(filters) == 1:
        return generate_filter(filters[0])
    return ComposeFilter(*(generate_filter(e) for e in filters))

def choice(*filters):
    if len(filters) == 1:
        return generate_filter(filters[0])
    return ChoiceFilter(*(generate_filter(e) for e in filters))

def each_line(*filters):
    return LineByLineFilter(*(generate_filter(e) for e in filters))


def filter(name, **kwargs):
    ns = globals()
    if name in ns:
        return ns[name](**kwargs)
    adhoc.print(f'フィルタ {name}が見つからないよ！ このままではフィルタリングされないよ')
    # FIXME をセットすることで修正箇所を記録する
    kwargs['FIXME'] = True
    return TextFilter(filter=name, **kwargs)

def generate_filter(expression):
    if isinstance(expression, TextFilter):
        return expression
    if isinstance(expression, list) and len(expression) > 0:
        if expression[0] == 'choice':
            filters = [generate_filter(e) for e in expression[1:]]
            return ChoiceFilter(*filters)
        elif expression[0] == 'each_line':
            filters = [generate_filter(e) for e in expression[1:]]
            return LineByLineFilter(*filters)
        else:
            filters = [generate_filter(e) for e in expression]
            return ComposeFilter(*filters)
    if isinstance(expression, dict):
        if 'score' in expression:
            return maxmin(expression.pop('score'), **expression)
        if 'maxmin' in expression:
            return maxmin(expression.pop('maxmin'), **expression)
        if 'filter' in expression:
            name = expression.pop('filter')
            return filter(name, **expression)
    return TextFilter(unknown_expression=f'{expression}')

def load_filter(json_filename):
    with open(json_filename) as f:
        return generate_filter(json.load(f))

