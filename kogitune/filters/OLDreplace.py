from typing import List
import re
import regex
from .filters import TextFilter

def RE(*patterns: List[str], flags=0):
    return regex.compile('|'.join(patterns), flags=flags)

def replace_pattern(pattern, text, replaced, max_allowed_num=0):
    if max_allowed_num > 0:
        if len(pattern.findall(text)) <= max_allowed_num:
           return text
    return pattern.sub(replaced, text)

## URL

url_pattern = RE(
    r'https?://[\w/:%#\$&\?~\.\,=\+\-\\_]+', # \(\)
    # r'\%[0-9A-F](\%[0-9A-F])+',
    # r'\(url\)~[\w/:%#\$&\?\(\)~\.=\+\-]+',
    # r'\(url\)(?:\s+\(url\))?',
    # r'\<url\>(?:\s+\<url\>)?',
)

def replace_url(text, replaced='<url>', max_allowed_num=0):
    """
    text 中のURLを<url>に置き換える

    >>> replace_url("http://www.peugeot-approved.net/UWS/WebObjects/UWS.woa/wa/carDetail?globalKey=uwsa1_1723019f9af&currentBatch=2&searchType=1364aa4ee1d&searchFlag=true&carModel=36&globalKey=uwsa1_1723019f9af uwsa1_172febeffb0, 本体価格 3,780,000 円")
    '<url> uwsa1_172febeffb0, 本体価格 3,780,000 円'

    >>> replace_url("「INVADER GIRL!」https://www.youtube.com/watch?v=dgm6-uCDVt0")
    '「INVADER GIRL!」<url>'

    >>> replace_url("http://t.co/x0vBigH1Raシグネチャー")
    '<url>'

    >>> replace_url("(http://t.co/x0vBigH1Ra)シグネチャー")
    '(<url>)シグネチャー'

    >>> replace_url("kindleにあるなーw http://www.amazon.co.jp/s/ref=nb_sb_noss?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&url=search-alias%3Ddigital-text&field-keywords=%E3%82%A2%E3%82%B0%E3%83%8D%E3%82%B9%E4%BB%AE%E9%9D%A2")
    'kindleにあるなーw <url>'

    >>> replace_url("http://live.nicovideo.jp/watch/lv265893495 #nicoch2585696")
    '<url> #nicoch2585696'
    
    """

    text = replace_pattern(url_pattern, text, replaced, max_allowed_num=max_allowed_num)
    return text

# 日付

date_pattern = RE (
    r'(?:19|20)\d{2}[-/\.]\d{1,2}[-/\.]\d{1,2}',
    r'(?:19|20)\d{2}\s?年\s?\d{1,2}\s?月\s?\d{1,2}\s?日(?![はにかまのも、])', # 否定先読み
    r'(?:令和|平成|昭和)\s?\d{1,2}\s?年\s?\d{1,2}\s?月\s?\d{1,2}\s?日(?![はにかまのも、])', # 否定先読み
    r'[HRS]\d{1,2}[-/\.]\d{1,2}[-/\.]\d{1,2}',  # Matches 'H27.9.18'
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
    #英国式  
    r'\d{1,2}\s+Jan(?:uary)?\s+\d{4}\b',
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

def replace_date(text, replaced='<date>', max_allowed_num=0):
    """
    text中の日付を<date>に置き換える

    >>> replace_date("March 20, 2016")
    '<date>'

    >>> replace_date("March 20, 2016 13:24")
    '<date> 13:24'

    >>> replace_date("Posted on: Tuesday, 01 May, 2018 18:14")
    'Posted on: Tuesday, <date> 18:14'

    >>> replace_date("返信・引用 FEB 12, 2004")
    '返信・引用 <date>'

    >>> replace_date("|18-Feb-2013|")
    '|<date>|'

    >>> replace_date("2016/08/16 17:37 - みすずのつぶやき")
    '<date> 17:37 - みすずのつぶやき'

    >>> replace_date("2007-11-14 Wed")
    '<date> Wed'

    >>> replace_date("| 2016-03-08 12:18")
    '| <date> 12:18'

    >>> replace_date("HIT 2001-01-09 レス")
    'HIT <date> レス'

    >>> replace_date("HIT 2001-1-9 レス")
    'HIT <date> レス'

    >>> replace_date("(2016.3.8. 弁理士 鈴木学)")
    '(<date>. 弁理士 鈴木学)'

    >>> replace_date("Posted: 2005.06.22")
    'Posted: <date>'

    >>> replace_date("HIT 2019/03/23(土) レス")
    'HIT <date>(土) レス'

    >>> replace_date("35: 名刺は切らしておりまして 2017/01/26(木) 23:22:16.16")
    '35: 名刺は切らしておりまして <date>(木) 23:22:16.16'

    >>> replace_date("2013年01月15日 本澤二郎の「日本の風景」(1252) ")
    '<date> 本澤二郎の「日本の風景」(1252) '

    >>> replace_date("2009年 08月22日 10:08 (土)")
    '<date> 10:08 (土)'

    >>> replace_date("2018年1月30日 at 22:37")
    '<date> at 22:37'

    >>> replace_date("1972年12月19日生まれ")
    '<date>生まれ'

    >>> replace_date("H27.9.18 9月議会最終日。")
    '<date> 9月議会最終日。'

    >>> replace_date("令和元年10月6日(日)")
    '令和元年10月6日(日)'

    >>> replace_date("平成22年12月22日の時点で")
    '平成22年12月22日の時点で'

    >>> replace_date("anond:20190414004605")
    'anond:20190414004605'

    >>> replace_date("その後、1988年秋に日経産業新聞から")
    'その後、1988年秋に日経産業新聞から'

    >>> replace_date("月を選択 2019年8月 (246) ")
    '月を選択 2019年8月 (246) '

    >>> replace_date("東京工業大学は令和1年12月24日からORCIDメンバーとなっています")
    '東京工業大学は令和1年12月24日からORCIDメンバーとなっています'

    >>> replace_date("http://tanakaryusaku.jp/2016/10/00014719")
    'http://tanakaryusaku.jp/2016/10/00014719'
    """
    return replace_pattern(date_pattern, text, replaced, max_allowed_num=max_allowed_num)

time_pattern = RE (
    r'\d{1,2}:\d{1,2}(:\d{1,2})?(\.\d{2,3})?(\s*(AM|PM))?',
    r'\(\d{1,2}:\d{1,2}(:\d{1,2})?\)',
    r'\d{1,2}[時]\s?\d{1,2}\s?[分]\s?\d{1,2}\s?[秒]',
    r'\d{1,2}[時]\s?\d{1,2}\s?[分](?![はにかまのも、])',
    r'\(time\)(?:\.\d{2,3})',
    r'\<time\>(?:\.\d{2,3})',
)

def replace_time(text, replaced='<time>', max_allowed_num=0):
    """
    text中の時刻を<time>に置き換える

    >>> replace_time("35: 名刺は切らしておりまして 2017/01/26(木) 23:22:16.16")
    '35: 名刺は切らしておりまして 2017/01/26(木) <time>'

    >>> replace_time("2009年 08月11日 11:04 (火)")
    '2009年 08月11日 <time> (火)'

    >>> replace_time("2017年1月9日 8:12 PM")
    '2017年1月9日 <time>'

    """
    return replace_pattern(time_pattern, text, replaced, max_allowed_num=max_allowed_num)

datetime_pattern = RE(
    r'\<date\>[^\n]{,8}\<time>',
    r'\(date\)[^\n]{,8}\n',
    r'\<date\>[^\n]{,8}\n',
)

def replace_datetime(text, replaced='<date>'):
    """
    text中の日時時刻を<date>に置き換える

    >>> replace_datetime("35: 名刺は切らしておりまして 2017/01/26(木) 23:22:16.16")
    '35: 名刺は切らしておりまして <date>'

    >>> replace_datetime("2009年 08月11日 11:04 (火)")
    '<date> (火)'

    >>> replace_datetime("1999年12月24日 19時11分9秒 返信・引用")
    '<date> 返信・引用'

    """

    text = replace_date(text, replaced)
    text = replace_time(text)
    text = replace_pattern(datetime_pattern, text, replaced)
    return text


email_pattern = RE(
    r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}',
    r'\<EMAIL\>', # StarCoder
    flags=re.IGNORECASE,
)

def replace_email(text, replaced='<email>', max_allowed_num=0):
    """
    text中のメールアドレスを<email>に置き換える

    >>> replace_email("Chian-Wei Teo +81-3-3201-3623　cwteo@bloomberg.net 香港")
    'Chian-Wei Teo +81-3-3201-3623　<email> 香港'

    """

    text = replace_pattern(email_pattern, text, replaced, max_allowed_num=max_allowed_num)
    return text

phone_pattern = RE(
    r'\(0\d{1,4}\)\s*\d{2,4}-\d{3,4}',
    r'0\d{1,4}-\d{2,4}-\d{3,4}',
    r'\+81-\d{1,4}-\d{2,4}-\d{3,4}',
    r'\+\d{10,}' #+819012345678
    r'\(phone\)', r'\<phone\>',
)

#Chian-Wei Teo +81-3-3201-3623　cwteo@bloomberg.net 香港

def replace_phone(text, replaced='<phone>', max_allowed_num=0):
    """
    text中の電話番号を<phone>に置き換える

    >>> replace_phone("Contact me at 0532-55-2222")
    'Contact me at <phone>'

    >>> replace_phone("TEL0120-350-108")
    'TEL<phone>'

    >>> replace_phone("大阪市中央区日本橋1-21-20-604ATEL:06-7860-2088")
    '大阪市中央区日本橋1-21-20-604ATEL:<phone>'

    >>> replace_phone("Chian-Wei Teo +81-3-3201-3623　cwteo@bloomberg.net 香港")
    'Chian-Wei Teo <phone>　cwteo@bloomberg.net 香港'

    """
    text = replace_pattern(phone_pattern, text, replaced, max_allowed_num=max_allowed_num)
    return text

address_pattern = RE(
    r'〒\d{3}-\d{4}[\s|\n][^\d]{,40}\d{1,4}(?:[−\-ー]\d{1,4}(?:[−\-ー]\d{1,4})?)?',
#    r'(?:北海道|青森県|岩手県|宮城県|秋田県|山形県|福島県|茨城県|栃木県|群馬県|埼玉県|千葉県|東京都|神奈川県|新潟県|富山県|石川県|福井県|山梨県|長野県|岐阜県|静岡県|愛知県|三重県|滋賀県|京都府|大阪府|兵庫県|奈良県|和歌山県|鳥取県|島根県|岡山県|広島県|山口県|徳島県|香川県|愛媛県|高知県|福岡県|佐賀県|長崎県|熊本県|大分県|宮崎県|鹿児島県|沖縄県){,40}\d{1,4}(?:[−\-ー]\d{1,4}(?:[−\-ー]\d{1,4})?)?',
)

def replace_address(text, replaced='<address>', max_allowed_num=0):
    """
    text中の電話番号を<address>に置き換える

    >>> replace_address("〒311-2421 茨城県潮来市辻232−2  NVVKx1T0QuvSfCR")
    '<address>  NVVKx1T0QuvSfCR'

    """
    text = replace_pattern(address_pattern, text, replaced, max_allowed_num=max_allowed_num)
    return text

## ID

account_pattern = RE(
    r'(@[A-Za-z0-9_]+)',
    r'(\b[Ii][Dd]:[A-Za-z0-9\-+_\.\/]+)',
    r'(\b[a-z]+[0-9_][0-9a-z_]*\b)',
    r'([0-9]+[a-z][0-9a-z_]+)',
    r'(\<NAME\>)',
    r'(\d{4}[ \-\/]\d{4}[ \-\/]\d{4}[ \-\/]\d{4})', # クレジットカード？
)

product_id_pattern = RE(
    r'(\b[Nn][Oo][:\.][A-Za-z0-9\-+_\.\/]{3,})',
    r'(\b[A-Z]+[_\-0-9][A-Z0-9_\-]+)',
    r'(\b[0-9]{2,}[A-Z\-][A-Z0-9_\-]+)',
    r'(\b[A-Z0-9]+\-[0-9]{5,}[A-Z0-9\-]*)',
    r'([0-9]+[A-Z_\/\-]+[A-Z0-9]+)',
    r'(\b[A-Z]{4,}[_\/\.0-9][A-Z0-9_\/\.=]*)',
    r'([0-9]{6,})',
)

base64_pattern = RE(
    r'(\b[0-9\+/]+[a-z]+[0-9\+/A-Z]+[a-z]+[0-9\+/A-Z]+[0-9a-zA-Z\+/]*={0,2}\b)',
    r'(\b[0-9\+/]+[A-Z]+[0-9\+/a-z]+[A-Z]+[0-9\+/a-z]+[0-9a-zA-Z\+/]*={0,2}\b)',
    r'(\b[0-9a-zA-Z+/]{4,}={1,2}\b)',
)

uuid_pattern = RE(r'\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b')

hash_pattern = RE(
    r'(\b[a-f]+[0-9]+[a-f]+[0-9]+[a-f0-9]{3,}\b)',
    r'(\b[0-9]+[a-f]+[0-9]+[a-f]+[a-f0-9]{3,}\b)',
    r'(\b[A-F]+[0-9]+[A-F]+[0-9]+[A-F0-9]{3,}\b)',
    r'(\b[0-9]+[A-F]+[0-9]+[A-F]+[A-F0-9]{3,}\b)',
)

def replace_id(text, replaced='<name>', max_allowed_num=0):
    """
    text中の名前らしい識別子を<name>に置き換える

    >>> replace_id("藍猫 (@0309LOVE) Twitter")
    '藍猫 (<name>) Twitter'

    >>> replace_id("32 M子 id:OvB44Wm. 現代書館@gendaishokanさん")
    '32 M子 <name> 現代書館<name>さん'

    >>> replace_id("hairs7777 <date>« ランキングの好み")
    '<name> <date>« ランキングの好み'

    >>> replace_id("SLRカメラBS真鍮1201S563")
    'SLRカメラBS真鍮<id>'
        
    >>> replace_id("ZMCQV102741286153207☆おすすめ商品")
    '<id>☆おすすめ商品'

    >>> replace_id("大柴邦彦|DSTD-09705 アニメ")
    '大柴邦彦|<id> アニメ'

    >>> replace_id("S3 No.447984906")
    'S3 <id>'

    >>> replace_id("FX-TL3-MA-0006 FOX フォックス")
    '<id> FOX フォックス'

    >>> replace_id("入数100 03505325-001")
    '入数100 <id>'

    >>> replace_id("2311H-40BOX EA436BB-12 03073939-001")
    '<id> <id> <id>'

    >>> replace_id("LC500 フロント左右セット URZ100 T980DF")
    '<id> フロント左右セット <id> <id>'

    >>> replace_id("アイボリー 500043532")
    'アイボリー <id>'

    >>> replace_id("着丈79.580.581.582838485")
    '着丈79.580.581.<id>'

    >>> replace_id("1641 2.0 rakutenblog facebook_feed fc2_blog")
    '1641 2.0 rakutenblog <name> <name>'

    >>> replace_id("ed8ufce1382ez0ag7k 71lje_pxldbfa3f6529gjq9xwyv1mbw 801stws0r7dfqud905aedaln-a0ik29")
    '<name> <name> <name>-<name>'

    >>> replace_id("550e8400-e29b-41d4-a716-446655440000")
    '<uuid>'

    >>> replace_id("d41d8cd98f00b204e9800998ecf8427e")
    '<hash>'
    
    >>> replace_id("YWJjZGVmZw==")  #FIXME
    'YWJjZGVmZw=='
    
    >>> replace_id("#f1f1f1")  #FIXME
    '#<name>'
    
    >>> replace_id("Induction Certificate")
    'Induction Certificate'

    """
    text = replace_pattern(uuid_pattern, text, '<uuid>')
    text = replace_pattern(hash_pattern, text, '<hash>')
#    text = replace_pattern(account_pattern, text, replaced)
    text = replace_pattern(base64_pattern, text, '<base64>')
    text = replace_pattern(product_id_pattern, text, '<id>')
    return text

def replace_uuid(text):
    return replace_pattern(uuid_pattern, text, '<uuid>')

# idlike_pattern = RE(
#     r'\b[A-Za-z0-9_\,]\b',
# )

# def replace_id(text, a):
#     for idlike in idlike_pattern.findall(text):
#         cs = list(idlike)

## アーティクル用

bar_pattern = RE(
    r'([^\s])\1{4,}',
)

def replace_bar(text):
    """
    text中の-----などの４連続以上の文字を短くする

    >>> replace_bar("-----------")
    '---'

    >>> replace_bar("――――――――――――――――――――――――――――【緊急速報】")
    '―――【緊急速報】'

    >>> replace_bar("-----*-----*-----*-----")
    '---*---*---*---'

    >>> replace_bar("    a=1")   # インデント
    '    a=1'

    >>> replace_bar("乗っ取りだ～wwwwww おおお!")
    '乗っ取りだ～www おおお!'

    >>> replace_bar("FF13の戦闘やべえｗｗｗｗｗｗｗｗｗｗｗｗｗｗｗｗｗｗおおお!")
    'FF13の戦闘やべえｗｗｗおおお!'
    """

    text = replace_pattern(bar_pattern, text, r'\1\1\1')
    return text

# enclose

double_enclose_pattern = RE(
    r'(\<\<[\s\S]+?\>\>)',
    r'(\[\[[\s\S]+?\]\])',
    r'(\{\{[\s\S]+?\}\})',
    r'(\(\([\s\S]+?\)\))',
    r'(【【[\s\S]+?】】)',
)

enclose_pattern = RE(
    r'(\[[^\]]+?\])',
    r'(\([^\)]+?\))', 
    r'(\{[^\}]+?\})', 
    r'(【[^】]+?】)',
    r'(（[^）]+?）)',
    r'(〔[^〕]+?〕)',
    r'(《[^》]+?》)',
)

close_pattern = RE(
    r'(\&[\#\w\d]+\;\s*)',
)

def replace_enclose(text, replaced='', code=False, max_allowed_num=0):
    """
    text中から()などを取り除く

    >>> replace_enclose("仕様(デザイン、サイズ、カラーなど)に多少のバラツキが")
    '仕様に多少のバラツキが'

    >>> replace_enclose("サイズ外形:(約)幅 39 × 奥行 41")
    'サイズ外形:幅 39 × 奥行 41'

    >>> replace_enclose("リアリティ・ショー乱造の背景【米国テレビ事情】 - OMASUKI FIGHT")
    'リアリティ・ショー乱造の背景 - OMASUKI FIGHT'

    >>> replace_enclose("《街の口コミあり》白市駅")
    '白市駅'
   
    >>> replace_enclose("賃貸住宅[賃貸マンション・賃貸一軒家]で部屋探")
    '賃貸住宅で部屋探'

    >>> replace_enclose("2020年2月 (44) 2020年1月 (54) ")
    '2020年2月  2020年1月  '

    >>> replace_enclose("ｶﾞｸｶﾞｸ {{ (>_<) }} ﾌﾞﾙﾌﾞﾙ状態")
    'ｶﾞｸｶﾞｸ  ﾌﾞﾙﾌﾞﾙ状態'
    
    >>> replace_menu("邪神〈イリス〉覚醒")    # 除去しない
    '邪神〈イリス〉覚醒'

    """
    text = replace_pattern(double_enclose_pattern, text, replaced)
    text = replace_pattern(enclose_pattern, text, replaced)
    text = replace_pattern(close_pattern, text, replaced)
    return text


## アーティクル用

menu_pattern = RE(
    r'\s+[\|｜>＞/／«»].{,256}\n',
    r'[\|｜＞／«»].{,256}\n',
    r'\b12345678910\b',
)

def replace_menu(text, max_allowed_num=0):
    """
    textの中のメニューらしいところを取り出す

    >>> replace_menu("生八つ橋のタグまとめ | エキサイトブログ\\n生八つ橋のタグまとめ\\n")
    '生八つ橋のタグまとめ<menu>\\n生八つ橋のタグまとめ\\n'

    >>> replace_menu("ガメラ３／邪神〈イリス〉覚醒\\n邪神〈イリス〉覚醒")
    'ガメラ３<menu>\\n邪神〈イリス〉覚醒'

    """
    text = replace_pattern(menu_pattern, text, '<menu>\n')
    return text

article_pattern = RE(
    r'(?P<prefix>\n#?)\d{2,}[\:\.]?\b',
    r'(?P<prefix>\>\s*)\d{2,}[\:\.]?\b',
    # No.447984906
)

def replace_article(text):
    """
    textの中の記事番号を短くする

    >>> replace_article("\\n99: ナナシさん")
    '\\n<article>: ナナシさん'

    >>> replace_article("\\n600.投稿日:<date> <time>")
    '\\n<article>投稿日:<date> <time>'

    >>> replace_article(">>> 7084 およ")
    '>>> <article> およ'

    """

    text = replace_pattern(article_pattern, text, '\g<prefix><article>')
    return text


number_pattern = RE(
    r'\d{5,}', 
    r'\d*\.\d{4,}'
    r'\(num\)',
)

def replace_number(text, max_allowed_num=0):
    text = replace_pattern(number_pattern, text, "<N>", max_allowed_num=max_allowed_num)
    return text

### コード用

float_pattern = RE(
    r'(?P<prefix>\d*[\.]\d{3})\d{2,}',
)

def replace_float(text):
    """
    text中の少数点数を短くする

    >>> replace_float("大さじ 1.8942041666667 ")
    '大さじ 1.894 '
    """
    text = replace_pattern(float_pattern, text, r'\g<prefix>')
    return text

string_pattern = RE(
    r'(?P<prefix>\"[\w/_\.\-]{8})[^\"\n\s]{16,}(?P<suffix>[\w\._/\-]{8}\")',
    r"(?P<prefix>\'[\w/_\.\-]{8})[^\'\n\s]{16,}(?P<suffix>[\w\._/\-]{8}\')",
)

def replace_longstring(text):
    """
    コード中の文字列を短くする

    >>> replace_longstring("'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'")
    "'aaaaaaaa(...)aaaaaaaa'"

    >>> replace_longstring('"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"')
    '"aaaaaaaa(...)aaaaaaaa"'
    """
    text = replace_pattern(string_pattern, text, r'\g<prefix>(...)\g<suffix>')
    return text


cleanup_pattern = RE(
    r'\<\*\>.{,8}\<\*\>',
)

extra_newlines_pattern = RE(
    r'\n{3,}',
)

ignore_line_pattern = RE(
    r'\<menu\>|copyright|\(c\)|\s[\|\/]\s', 
    flags=re.IGNORECASE
)

def cleanup(text):
    lines = ['']
    for line in text.split('\n'):
        if len(line) < 40 and line.count('<') > 1:
            continue
        if len(ignore_line_pattern.findall(line)) > 0:
            if len(line) > 128 and '。' in line:
                if '<menu>' not in line:
                    lines.append(line)
                    continue
                # 一行が長い場合は、たぶん改行ミスなので、最後の(menu)だけ除去
                lines.append('。'.join(line.split('。')[:-1])+'。')
            lines.append('')
            continue
        lines.append(line)
    text = '\n'.join(lines)
    text = replace_pattern(extra_newlines_pattern, text, '\n\n')
    return text.replace('<url>', '')


def CCFilter(text):
    text = replace_url(text)
    text = replace_email(text)
    text = replace_datetime(text)
    text = replace_phone(text)
    text = replace_address(text)
    text = replace_enclose(text)
    text = replace_id(text)
    text = replace_float(text)
    text = replace_article(text)
    text = replace_menu(text)
    text = replace_bar(text)
    return cleanup(text)


def find_replace_func(pattern:str):
    func = globals().get(f'replace_{pattern}')
    if func is None:
        patterns = [s.replace('replace_', '') for s in globals() if s.startswith('replace_')]
        raise ValueError(f'replace_{pattern} is not found. Select pattern from {patterns}')
    return func




class Replacement(TextFilter):
    """
    置き換えフィルター
    :patterns: 'url:<URL>|date:<date>'
    """

    def __init__(self, patterns: List[str]):
        """
        置き換えフィルターを作る
        :param patterns: 置き換える文字列パターンのリスト
        """
        if isinstance(patterns,str):
            patterns = patterns.split('|')
        self.replace_funcs = []
        for pattern in patterns:
            pattern, _, w = pattern.partition(':')
            self.replace_funcs.append((find_replace_func(pattern), w))

    def __call__(self, text):
        for replace_fn, w in self.replace_funcs:
            text = replace_fn(text, w)
            if text is None:
                break
        return cleanup(text)

if __name__ == '__main__':
    import doctest # doctestのための記載
    print(doctest.testmod())
