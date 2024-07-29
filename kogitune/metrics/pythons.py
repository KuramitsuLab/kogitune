import ast

def get_syntax_error_line(code):
    try:
        ast.parse(code)
        return None  # エラーがない場合はNoneを返す
    except SyntaxError as e:
        return e.lineno  # エラーが発生した行番号を返す

def clean_code(code):
    while True:
        error_lineno = get_syntax_error_line(code)
        if error_lineno is None:
            return code
        if '\n' not in code:
            break
        code, _, _ = code.rpartition('\n')
    return None

def extract_python_code(text):
    result = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        if lines[i].strip() == '':
            # 空行はスキップする
            i += 1
            continue
        code = '\n'.join(lines[i:])
        next = get_syntax_error_line(code)
        #print(i, next, code)
        if next == 1:
            # 先頭でエラーが発生したらスキップする
            i += 1
            continue
        if next is None:
            result.append(code)
            break
        code = clean_code('\n'.join(lines[i:i+next-1]))
        if code is not None:
            result.append(code)
        i += next
    return '\n'.join(result)
