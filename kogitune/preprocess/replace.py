import re


def replace_pattern(pattern, text, replaced, allowable=0):
    if allowable > 0:
        if len(pattern.findall(text)) <= allowable:
           return text
    return pattern.sub(replaced, text)

date_pattern = r'(?:19|20)\d{2}[-/\.]\d{1,2}[-/\.]\d{1,4}|' + \
              r'(?:19|20)\d{2}\s?年\s?\d{1,2}\s?月(\s?\d{1,2}\s?日)?(\s*[\(（][月火水木金土日][\)）])?|' + \
              r'\d{2}[-/]\d{1,2}[-/]\d{2,4}|\(\d{2}/\d{2}\)|' + \
                r'(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|' + \
                r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|' + \
                r'Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b)|' + \
                r'(\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|' + \
                r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|' + \
                r'Dec(?:ember)?)\s+\d{4}\b)'
date_pattern = re.compile(date_pattern)

datetime_pattern = r'\(date\)[\w\d \(\)\[\]\-:]+\n' + \
              r'|\(date\)\s+\d{1,2}時\d{1,2}分(\d{1,2}秒)?'
datetime_pattern = re.compile(datetime_pattern)

time_pattern = r'\(\d{1,2}:\d{1,2}(:\d{1,2})?\)|\d{1,2}:\d{1,2}(:\d{1,2})?'

time_pattern = re.compile(time_pattern)

def replace_date(text, allowable=0):
    text = replace_pattern(date_pattern, text, "(date)", allowable=allowable)
    text = replace_pattern(datetime_pattern, text, "(date)", allowable=allowable)
    text = replace_pattern(time_pattern, text, "(time)", allowable=allowable)
    return text
# 
sample_text = "Contact me at 1999年12月24日(土) or follow me 1999/3/2 12:00 on +81-3-3333-3333."
replaced_text = replace_date(sample_text)
print(replaced_text)


url_pattern = re.compile(r'(?:https?:\/\/)?(?:[\da-z\.-]+)\.(?:[a-z\.]{2,6})(?:[\/\w\.-]*)*\/?(?:\?[^\s]*)?(?:#[^\s]*)?')

def replace_url(text, allowable=0):
    return replace_pattern(url_pattern, text, "(url)", allowable=allowable)

# sample_text = "Contact me at https://chat.openai.com/c/7f5e95bc-aba2-4a62-b6d6-3242788065ae or follow me on Twitter @john_doe."
# replaced_text = replace_url(sample_text)
# print(replaced_text)

email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b|@[A-Za-z0-9_]+\b')
twitter_pattern = re.compile(r'@[A-Za-z0-9_]+\b|\b[B|b]y\s*@[A-Za-z0-9_\-\.]\b')

def replace_email(text, allowable=0):
    text = replace_pattern(email_pattern, text, "(email)", allowable=allowable)
    return replace_pattern(twitter_pattern, text, "(contact)", allowable=allowable)

number_pattern = re.compile(r'[\[(]\d{2,4}[\])]|\b\d{5,}\b')

def replace_number(text, allowable=0):
    text = replace_pattern(number_pattern, text, "(num)", allowable=allowable)
    return text

# sample_text = "Contact me at john.doe@example.com or follow me on Twitter @john_doe."
# replaced_text = replace_email(sample_text)
# print(replaced_text)

phone_pattern = re.compile(r'\(0\d{1,4}\)\s*\d{2,4}-\d{4}|(?:0|\+81-)\d{1,4}-\d{2,4}-\d{4}')

def replace_phone(text, allowable=0):
    return replace_pattern(phone_pattern, text, "(phone)", allowable=allowable)

#sample_text = "Contact me at 0532-55-2222 or follow me on +81-3-3333-3333."
#replaced_text = replace_phone(sample_text)
#print(replaced_text)

base64_pattern = re.compile(r'(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?')
uuid_pattern = re.compile(r'\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b')
hash_pattern = re.compile(r'\b[a-fA-F0-9]{7,}\b')

def replace_data(text, allowable=0):
    text = replace_pattern(base64_pattern, text, allowable=allowable)
    text = replace_pattern(uuid_pattern, text, allowable=allowable)
    text = replace_pattern(hash_pattern, text, allowable=allowable)
    return text

def replace_full(text):
    text=replace_date(text)
    text=replace_email(text)
    text=replace_url(text)
    text=replace_phone(text)
    text=replace_number(text)
    return text
