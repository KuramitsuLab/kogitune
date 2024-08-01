import sys
import ast
import traceback

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

def extract_from_code_completion(prompt, generated_text):
    stop_sequences=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]
    min_stop_index = len(generated_text)
    for seq in stop_sequences:
        stop_index = generated_text.find(seq)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    code = prompt + "\n" + generated_text[:min_stop_index]
    return extract_python_code(code)

TEMPLATE_CODE_FIX = '''\
The following error has occurred. 
Please fix the code so that it can be executed without errors.

### Code
{code}

### Error
{error_message}
{stack_trace}
{error_message}

'''

def get_error_line_number():
    stack_trace = traceback.format_exc()
    # スタックトレースの最後の呼び出し部分から行番号を抽出
    tb_lines = stack_trace.splitlines()
    line_number = len(tb_lines)
    # print('@@@', stack_trace)
    for line in tb_lines:
        if 'File "<string>"' in line and ", line" in line:
            # 行番号を抽出
            try:
                _,_,linenum = line.partition(", line ")
                linenum,_,_ = linenum.partition(',')
                line_number = int(linenum)
            except:
                pass
    return line_number

def format_error_lines(code, line_number):
    code_lines = code.strip().split('\n')
    formatted_code = ""
    for i, line in enumerate(code_lines, 1):
        if i == line_number:
            formatted_code += f"----> {i} {line}\n"
        elif line_number - 2 <= i <= line_number + 1:
            formatted_code += f"      {i} {line}\n"
    return formatted_code

def get_code_fix_prompt(code_str, test_code):
    if isinstance(code_str, list):
        return [get_code_fix_prompt(x, test_code) for x in code_str]
    code = (code_str+test_code).strip()
    try:
        # コードを実行
        exec(code)
        return ''
    except Exception as e:
        # エラーが発生した場合、エラーメッセージとスタックトレースを回収
        error_message = f'{type(e).__name__}: {str(e)}'
        # _, _, tb = sys.exc_info()
        line_number = get_error_line_number()
        formatted_code = format_error_lines(code, line_number)
        prompt = TEMPLATE_CODE_FIX.format(
            error_message=error_message, 
            stack_trace=formatted_code, 
            code=code_str)
        return prompt

