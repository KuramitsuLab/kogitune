from typing import List

from .commons import *
from .documents import *
from .scores import MaxMinFilter, maxmin
from .replaces import replace

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

def filter_from_path(path):
    path, kwargs = adhoc.parse_path_args(path)
    if 'class_name' not in kwargs:
        kwargs['class_name'] = path
    namespace = globals()
    return instantiate_json(kwargs, namespace)

def generate_filter(expression):
    if isinstance(expression, TextFilter):
        return expression
    if isinstance(expression, str):
        return filter_from_path(expression)
    if isinstance(expression, list) and len(expression) > 0:
        filters = [generate_filter(e) for e in expression]
        return ComposeFilter(*filters)
    return TextFilter(unknown_expression=f'{expression}')

def load_filter(json_filename):
    with open(json_filename) as f:
        namespace = globals()
        return instantiate_json(json.load(f), namespace)

