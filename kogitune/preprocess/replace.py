from typing import List
import re
import regex

def RE(*patterns: List[str], flags=0):
    return regex.compile('|'.join(patterns), flags=flags)

def replace_pattern(pattern, text, replaced, group=None, allowable=0):
    if allowable > 0:
        if len(pattern.findall(text)) <= allowable:
           return text
    if group:
        def replaceStr(m):
            print(str(m))
            s = m.group(group)
            print(s, replaced)
            return replaced
        return pattern.sub(replaceStr, text)
    return pattern.sub(replaced, text)


date_pattern = RE (
    r'(?:19|20)\d{2}[-/\.]\d{1,2}[-/\.]\d{1,2}',
    r'(?:19|20)\d{2}\s?年\s?\d{1,2}\s?月\s?\d{1,2}\s?日(?![はにかまのも、])', # 否定先読み
    r'(?:令和|平成|昭和)\s?\d{1,2}\s?年\s?\d{1,2}\s?月\s?\d{1,2}\s?日(?![はにかまのも、])', # 否定先読み
    r'\d{1,2}[-/\.]\d{1,2}[-/\.](?:19|20)\d{2}',
    #アメリカ式  
    r'\bJan(?:uary)?\s+\d{1,2}\,?\s+\d{4}\b',
    r'\bFeb(?:ruary)?\s+\d{1,2}\,?\s+\d{4}\b',
    r'\bMar(?:ch)?\s+\d{1,2}\,?\s+\d{4}\b',
    r'\bApr(?:il)?\s+\d{1,2}\,?\s+\d{4}\b',
    r'\bMay\s+\d{1,2}\,?\s+\d{4}\b',
    r'\bJun(?:e)?\s+\d{1,2}\,?\s+\d{4}\b',
    r'\bJul(?:y)?\s+\d{1,2}\,?\s+\d{4}\b',
    r'\bAug(?:ust)?\s+\d{1,2}\,?\s+\d{4}\b',
    r'\bSep(?:tember)?\s+\d{1,2}\,?\s+\d{4}\b',
    r'\bOct(?:ober)?\s+\d{1,2}\,?\s+\d{4}\b',
    r'\bNov(?:ember)?\s+\d{1,2}\,?\s+\d{4}\b',
    r'\bDec(?:ember)?\s+\d{1,2}\,?\s+\d{4}\b',
    #英国  
    r'\d{1,2}\s+\bJan(?:uary)?\s+\d{4}\b',
    r'\b{1,2}\s+Feb(?:ruary)?\s+\d{4}\b',
    r'\b{1,2}\s+Mar(?:ch)?\s+\d{4}\b',
    r'\b{1,2}\s+Apr(?:il)?\s+\d{4}\b',
    r'\b{1,2}\s+May\s+\d{4}\b',
    r'\b{1,2}\s+Jun(?:e)?\s+\d{4}\b',
    r'\b{1,2}\s+Jul(?:y)?\s+\d{4}\b',
    r'\b{1,2}\s+Aug(?:ust)?\s+\d{4}\b',
    r'\b{1,2}\s+Sep(?:tember)?\s+\d{4}\b',
    r'\b{1,2}\s+Oct(?:ober)?\s+\d{4}\b',
    r'\b{1,2}\s+Nov(?:ember)?\s+\d{4}\b',
    r'\b{1,2}\s+Dec(?:ember)?\s+\d{4}\b',
    r'\<date\>',
    r'\(date\)',
    flags=re.IGNORECASE
)

time_pattern = RE (
    r'\d{1,2}:\d{1,2}(:\d{1,2})?(\.\d{2,3})?(\s*(AM|PM))?',
    r'\(\d{1,2}:\d{1,2}(:\d{1,2})?\)',
    r'\d{1,2}[時]\s?\d{1,2}\s?[分]\s?\d{1,2}\s?[秒]',
    r'\d{1,2}[時]\s?\d{1,2}\s?[分](?![はにかまのも、])',
    r'\(time\)(?:\.\d{2,3})',
    r'\<time\>(?:\.\d{2,3})',
)

datetime_pattern = RE(
    r'\(date\).{,8}\(time\)',
    r'\<date\>.{,8}\<date\>',
    r'\(date\).{,8}\n',
    r'\<date\>.{,8}\n',
)

def replace_date(text, date='<date>', time='<time>', allowable=0):
    if date:
        text = replace_pattern(date_pattern, text, date, allowable=allowable)
    if time:
        text = replace_pattern(time_pattern, text, time, allowable=allowable)
    if date and time:
        text = replace_pattern(datetime_pattern, text, date)
    return text

if __name__ == '__main__':
    sample_text = """
Contact me at 1999年12月24日(土) 12時 6分 or follow me 1999/3/2 12:00 AM on +81-3-3333-3333 2019.12.19
1999年12月24日 19時11分9秒 返信・引用 FEB 12, 2004
東京工業大学は令和1年12月24日からORCIDメンバーとなっています
"""
    replaced_text = replace_date(sample_text)
    print(replaced_text)

url_pattern = RE(
    r'https?://[\w/:%#\$&\?~\.=\+\-\\_]+', # \(\)
    r'\%[0-9A-F](\%[0-9A-F])+',
    r'\(url\)~[\w/:%#\$&\?\(\)~\.=\+\-]+',
    r'\(url\)(?:\s+\(url\))?',
    r'\<url\>(?:\s+\<url\>)?',
)

def replace_url(text, url='<url>', allowable=0):
    if url:
        text = replace_pattern(url_pattern, text, url, allowable=allowable)
    return text

base64_pattern = RE(
    r'\b(?:[A-Za-z0-9+/]{4}){3,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\b'
)

uuid_pattern = RE(r'\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b')

hash_pattern = RE(
    r'\b[a-fA-F]+[0-9]+[a-fA-F]+[0-9]+[a-fA-F0-9]{7,}\b',
    r'\b[0-9]+[a-fA-F]+[0-9]+[a-fA-F]+[a-fA-F0-9]{7,}\b',
)

def replace_uuid(text):
    text = replace_pattern(uuid_pattern, text, '<uuid>')
    # text = replace_pattern(base64_pattern, text, '<base64>')
    text = replace_pattern(hash_pattern, text, '<hash>')
    return text

email_pattern = RE(
    r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}',
    r'\(email\)', r'\<email\>', 
    flags=re.IGNORECASE,
)

def replace_email(text, email='<email>', allowable=0):
    if email:
        text = replace_pattern(email_pattern, text, email, allowable=allowable)
    return text

userid_pattern = RE(
    r'@[A-Za-z0-9_]+',
    r'\b[Ii][Dd]:[A-Za-z0-9\-+_.\/]+',
    r'\d{4} \d{4} \d{4} \d{4}', # クレジットカード？
)

idlike_pattern = RE(    
    r'\b[a-z]+[0-9\-_]+[a-z0-9\-_]*',
    r'\bby [A-Za-z\-_]+\b', 
    r'\b[A-Za-z0-9_/]{12,}\b',  # 長めの英単語と重なる
    r'\d{4} \d{4} \d{4} \d{4}', # クレジットカード？
)

def replace_name(text, user='<name>', strong=False, allowable=0):
    text = replace_pattern(userid_pattern, text, user, allowable=allowable)
    if strong:
        text = replace_pattern(idlike_pattern, text, user, allowable=allowable)
    return text

if __name__ == '__main__':
    sample_text = """
Contact me at https://chat.openai.com/~あ or follow me on http://a.com/?%E3%81%82.
32 M子 id:OvB44Wm. 現代書館@gendaishokanさん
by sono at (time)
そして id:rhythnnさん (ペパボ)
hairs7777 (date)« ランキングの好み
より詳しい日本語の説明はこちら。
Click here for a more detailed explanation in Japanese.
"""
    replaced_text = replace_url(sample_text)
    print(replaced_text)

phone_pattern = RE(
    r'\(0\d{1,4}\)\s*\d{2,4}-\d{3,4}',
    r'0\d{1,4}-\d{2,4}-\d{3,4}',
    r'\+81-\d{1,4}-\d{2,4}-\d{3,4}',
    r'\+\d{10,}' #+819012345678
    r'\(phone\)', r'\<phone\>',
)

address_pattern = RE(
    r'〒\d{3}-\d{4}[\s|\n][^\d]{,40}\d{1,4}(?:[−\-ー]\d{1,4}(?:[−\-ー]\d{1,4})?)?',
#    r'(?:北海道|青森県|岩手県|宮城県|秋田県|山形県|福島県|茨城県|栃木県|群馬県|埼玉県|千葉県|東京都|神奈川県|新潟県|富山県|石川県|福井県|山梨県|長野県|岐阜県|静岡県|愛知県|三重県|滋賀県|京都府|大阪府|兵庫県|奈良県|和歌山県|鳥取県|島根県|岡山県|広島県|山口県|徳島県|香川県|愛媛県|高知県|福岡県|佐賀県|長崎県|熊本県|大分県|宮崎県|鹿児島県|沖縄県){,40}\d{1,4}(?:[−\-ー]\d{1,4}(?:[−\-ー]\d{1,4})?)?',
)

def replace_phone(text, allowable=0):
    text = replace_pattern(phone_pattern, text, "<phone>", allowable=allowable)
    text = replace_pattern(address_pattern, text, "<address>", allowable=allowable)
    return text

if __name__ == '__main__':
    sample_text = """
Contact me at 0532-55-2222 or follow me on +81-3-3333-3333.
茨城県潮来市辻232−2
〒311-2421 茨城県潮来市辻232−2  NVVKx1T0QuvSfCR
〒803-0835
福岡県北九州市小倉北区井掘1-26-17 田園ビル2F
_lvddYekQdoQliubkd_avozze___wYYs

"""
    replaced_text = replace_phone(sample_text)
    print(replaced_text)

bar_pattern = RE(
    r'(.)\1{3,}',
)

def replace_bar(text, allowable=0):
    text = replace_pattern(bar_pattern, text, '\1\1\1')
    return text

article_pattern = RE(
    r'(?P<prefix>\n#?)\d{2,}[\:\.]?\b',
    r'(?P<prefix>\>\s*)\d{2,}[\:\.]?\b',
)

number_pattern = RE(
    r'\d{5,}', 
    r'\d*\.\d{4,}'
    r'\(num\)',
)

def replace_number(text, allowable=0):
    text = replace_pattern(bar_pattern, text, r'\1\1\1')
    text = replace_pattern(article_pattern, text, '\g<prefix><article>')
    text = replace_pattern(number_pattern, text, "<N>", allowable=allowable)
    return text

if __name__ == '__main__':
    sample_text = """
wwwwww
=====================
-----*-----*-----*
600.投稿日:(date) (time)
――――――――――――――――――――――――――――【緊急速報】
>>> 7084 およ
123次へ >>>
"""
    replaced_text = replace_number(sample_text)
    print(replaced_text)



menu_pattern = RE(
    r'\s*[\|｜>＞/／«»].{,256}\n',
    r'\D12345678910\D',
)

enclose_pattern = RE(
    r'(\[[^\]]+?\])',
    r'(【[^】]+?】)',
    r'(（[^）]+?）)',
    r'(〔[^〕]+?〕)',
    r'(\([\w\d ]+\))', 
    r'(\<[\w\d ]+\>)'
)

def replace_menu(text, allowable=0):
    text = replace_pattern(enclose_pattern, text, '(*)')
    text = replace_pattern(menu_pattern, text, '(menu)\n')
    return text


def replace_full(text):
    text=replace_menu(text)
    text=replace_date(text)
    text=replace_url(text)
    text=replace_phone(text)
    text=replace_number(text)
    return text


cleanup_pattern = RE(
    r'\(\*\).{,8}\(\*\)',
)

extra_newlines_pattern = RE(r'\n{3,}')

def cleanup(text):
    text = replace_pattern(cleanup_pattern, '(*)'),
    text = replace_pattern(extra_newlines_pattern, '\n\n')
    return text.replace('(*)', '')

ignore_line_pattern = RE(
    r'\(menu\)|copyright|\(c\)|\s[\|\/]\s|[:,]', 
    flags=re.IGNORECASE
)

def filter_mc4(text, prefix_size=8):
    text=replace_menu(text)
    lines = ['']
    for line in text.split('\n'):
        if len(ignore_line_pattern.findall(line)) > 0:
            if len(line) > 128 and '。' in line:
                if '(menu)' not in line:
                    lines.append(line)
                    continue
                # 一行が長い場合は、たぶん改行ミスなので、最後の(menu)だけ除去
                lines.append('。'.join(line.split('。')[:-1])+'。')
            lines.append('')
            continue
        if prefix_size < len(line) < 80 and lines[-1].startswith(line[:prefix_size]):
            # print('@A', lines[-1])
            # print('@B', line)
            if len(line) > len(lines):
                lines[-1] = line
            continue          
        lines.append(line)
    text = '\n'.join(lines)
    text=replace_date(text)
    text=replace_url(text)
    text=replace_phone(text)
    text=replace_number(text)
    text=replace_bar(text)
    text = cleanup(text)
    return text

if __name__ == '__main__':
    text = '''
2、社会資本の維持管理費等について。
(1)長寿命化計画等の策定について。
我が国の社会資本は、
'''

    print(filter_mc4(text))
