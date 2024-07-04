

import re
from collections import Counter

# languages that require word segmentation

def compile_with_word_segmentation(words):
    return re.compile(r'\b(?:' + '|'.join(map(re.escape, words)) + r')\b', re.IGNORECASE)

frequent_english_words = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not", "on", "with", "he",
    "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about",
    "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our",
    "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
]

pattern_en = compile_with_word_segmentation(frequent_english_words)

# 頻出フランス語単語100個のリスト
frequent_french_words = [
    "le", "de", "un", "à", "être", "et", "en", "avoir", "que", "pour", "dans", "ce", "il", "qui", "ne", "sur",
    "se", "pas", "plus", "pouvoir", "par", "je", "avec", "tout", "faire", "son", "mettre", "autre", "on", "mais",
    "nous", "comme", "ou", "si", "leur", "y", "dire", "elle", "avant", "deux", "bien", "où", "même", "prendre",
    "aussi", "celui", "donner", "du", "lui", "cela", "mon", "rien", "encore", "voir", "enfin", "aucun", "très",
    "savoir", "sans", "sous", "voilà", "un", "peu", "falloir", "votre", "quand", "quelque", "cela", "ton", "comme",
    "vers", "moins", "aimer", "comme", "alors", "autre", "beaucoup", "devoir", "là", "tout", "vouloir", "venir",
    "pendant", "ainsi", "cela", "là", "notre", "depuis", "quand", "autour", "chez", "sous", "près", "ainsi", "savoir"
]

pattern_fr = compile_with_word_segmentation(frequent_french_words)

frequent_spanish_words = [
    "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un", "por", "con", "no", "una", "su", 
    "para", "es", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta", 
    "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", 
    "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", 
    "mí", "antes", "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", 
    "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", 
    "mis", "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros", "vosotras", "os", "mío", "mía", "míos", 
    "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas"
]

pattern_es = compile_with_word_segmentation(frequent_spanish_words)

frequent_portuguese_words = [
    "de", "a", "que", "e", "o", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na", 
    "por", "mais", "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", 
    "quando", "muito", "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso", "ela", 
    "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", 
    "você", "tinha", "foram", "essa", "num", "nem", "suas", "meu", "minha", "numa", "pelos", "elas", "havia", 
    "sejam", "qual", "será", "nós", "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse", "dele", 
    "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa", "nossos", 
    "nossas", "dela", "delas", "esta", "estes", "estas"
]

pattern_pt = compile_with_word_segmentation(frequent_portuguese_words)

frequent_italian_words = [
    "di", "che", "e", "il", "la", "a", "per", "un", "in", "è", "io", "non", "sono", "una", "le", "si", "con", "mi", 
    "ma", "ti", "ci", "lo", "gli", "ha", "no", "ho", "questo", "tu", "quello", "al", "lui", "come", "del", "loro", 
    "della", "delle", "questi", "cosa", "molto", "quella", "su", "nel", "tutto", "questa", "alla", "ne", "essere", 
    "da", "ho", "tra", "me", "quando", "io", "ancora", "più", "della", "anche", "queste", "quelli", "là", "di", 
    "perché", "bene", "ma", "tu", "tutti", "ora", "suo", "hai", "nostro", "sempre", "se", "fa", "fatto", "poi", 
    "senza", "pochi", "dove", "grazie", "chi", "quanto", "ciò", "adesso", "allora", "dire", "quindi", "sto", "stai", 
    "stata", "stato", "ogni", "posso", "dopo", "quasi", "fare", "vostra", "nostra", "sopra", "loro"
]

pattern_it = compile_with_word_segmentation(frequent_italian_words)


frequent_german_words = [
    "der", "die", "und", "in", "zu", "den", "das", "nicht", "von", "sie", "ist", "des", "sich", "mit", "dem", 
    "dass", "er", "es", "ein", "ich", "auf", "so", "eine", "auch", "als", "an", "nach", "wie", "im", "für", 
    "man", "aber", "aus", "durch", "wenn", "nur", "war", "noch", "werden", "bei", "hat", "wir", "was", "wird", 
    "sein", "einen", "welche", "sind", "oder", "zur", "um", "haben", "einer", "mir", "über", "ihm", "diese", 
    "einem", "ihr", "uns", "da", "zum", "kann", "doch", "vor", "dieser", "mich", "ihn", "du", "hatte", "seine", 
    "mehr", "am", "denn", "nun", "unter", "sehr", "selbst", "schon", "hier", "bis", "habe", "ihre", "dann", 
    "ihnen", "seiner", "alle", "wieder", "meine", "Zeit", "gegen", "vom", "ganz", "einzelne", "wo", "muss", 
    "ohne", "eines", "können", "dieses", "hatten", "allen", "waren"
]

pattern_de = compile_with_word_segmentation(frequent_german_words)

frequent_finnish_words = [
    "ja", "on", "oli", "että", "ei", "se", "en", "hänen", "mutta", "tai", "joka", "kuin", "että", "niin", "hän", 
    "tässä", "nyt", "oli", "minä", "sinä", "kun", "jossa", "mitä", "kun", "mikä", "yksi", "se", "olen", "sitten", 
    "me", "kuten", "te", "he", "niin", "tämä", "nämä", "siinä", "vain", "vielä", "jo", "johon", "jolla", "voi", 
    "vain", "jotta", "jos", "koko", "saada", "jolloin", "tehdä", "tulee", "tämä", "sanoi", "kun", "niiden", "ollut", 
    "tässä", "ole", "voin", "ollut", "nyt", "ehkä", "tuli", "olla", "meidän", "kaikki", "sinun", "ovat", "ettei", 
    "mitään", "heitä", "olemme", "voi", "ehkä", "nyt", "niiden", "teitä", "juuri", "nyt", "vaikka", "niin", 
    "siksi", "niin", "näin", "koska", "kaksi", "omaa", "siten", "sitten", "olemme", "koska", "joskus", "kaikki", 
    "olla", "sitten", "seuraava", "samoin"
]

pattern_fi = compile_with_word_segmentation(frequent_finnish_words)

frequent_turkish_words = [
    "ve", "bir", "bu", "da", "ne", "için", "ile", "de", "mi", "ben", "o", "ama", "gibi", "çok", "daha", "var", "sen", 
    "diye", "bana", "benim", "oldu", "ki", "biraz", "olabilir", "bütün", "biz", "beni", "her", "şey", "siz", "beni", 
    "bile", "şu", "kadar", "o", "bunu", "çünkü", "artık", "neden", "bu", "nasıl", "olduğunu", "diğer", "zaman", "şimdi", 
    "hep", "aynı", "herkes", "onun", "çok", "kendine", "sadece", "sonra", "olduğunu", "diye", "olarak", "önce", "bunu", 
    "ancak", "senin", "bir şey", "üzerinde", "etmek", "çok", "ya", "vardı", "olmak", "orada", "şeyi", "beni", "kendi", 
    "iki", "birlikte", "herhangi", "olarak", "hem", "onlar", "kadar", "birisi", "önce", "arkadaşlar", "bile", "başka", 
    "sürekli", "büyük", "çoğu", "altında", "dışında", "herhangi", "kendini", "bunun", "bir şey", "hala", "ancak", 
    "yeni", "neden", "onun", "önemli", "nasıl", "üzerine", "birisi", "zaman", "kadar", "başka", "bunlar", "üzerinde"
]

pattern_tr = compile_with_word_segmentation(frequent_turkish_words)

frequent_indonesian_words = [
    "dan", "di", "yang", "untuk", "dengan", "pada", "ke", "dari", "adalah", "ini", "itu", "sebagai", "dalam", 
    "tidak", "bahwa", "oleh", "akan", "atau", "juga", "saya", "kami", "anda", "mereka", "kami", "kita", 
    "bisa", "telah", "lebih", "banyak", "sudah", "kalau", "harus", "ada", "tersebut", "agar", "tetapi", 
    "karena", "jadi", "serta", "hanya", "bagi", "saat", "seorang", "bawah", "atas", "tahun", "kali", "sama", 
    "seperti", "lalu", "setelah", "kemudian", "hingga", "kembali", "lain", "bisa", "begitu", "apabila", 
    "namun", "bila", "sejak", "sedang", "belum", "sambil", "sehingga", "yakni", "hanya", "malah", "meski", 
    "antara", "setiap", "dua", "dulu", "hal", "baik", "tanpa", "baru", "waktu", "orang", "maupun", "masih", 
    "lagi", "pun", "terhadap", "para", "selama", "menjadi", "terutama", "maka", "sekitar", "ambil", "buat", 
    "hingga", "mana", "dalam", "apakah", "kecil", "cukup", "setiap", "oleh", "milik", "hingga", "sudah", "telah"
]

pattern_id = compile_with_word_segmentation(frequent_indonesian_words)

frequent_vietnamese_words = [
    "và", "của", "là", "có", "trong", "được", "đến", "này", "cho", "một", "với", "như", "cũng", "còn", "khi", "người", 
    "năm", "để", "anh", "về", "đã", "tôi", "không", "nhiều", "thì", "mà", "hơn", "ra", "sẽ", "bị", "rất", "nhưng", 
    "đi", "nếu", "lại", "có", "đây", "nhất", "các", "nhiệm", "hoặc", "vì", "học", "nước", "ở", "vào", "phải", "em", 
    "ông", "bà", "gì", "chỉ", "nơi", "đầu", "họ", "chưa", "chúng", "từ", "bạn", "trên", "nữa", "điều", "tại", "ta", 
    "làm", "đó", "đang", "những", "vui", "đấy", "sao", "giờ", "sự", "đều", "bởi", "gần", "sau", "thật", "giữa", 
    "của", "bao", "ngay", "ai", "gặp", "thế", "nó", "vẫn", "về", "đều", "để", "đâu", "thì", "vào", "rồi", "mình", 
    "có", "lắm", "với", "vậy", "khác", "còn", "trước", "bây", "ngay", "nên", "đôi", "thành", "đầu", "vô", "phải"
]

pattern_vi = compile_with_word_segmentation(frequent_vietnamese_words)

frequent_russian_words = [
    "и", "в", "не", "на", "я", "что", "тот", "быть", "с", "он", "как", "это", "по", "но", "они", "к", "из", "у", 
    "от", "о", "со", "за", "так", "весь", "она", "вы", "ты", "мы", "же", "который", "мочь", "этот", "говорить", 
    "для", "вот", "делать", "если", "её", "наш", "их", "его", "только", "себя", "ещё", "один", "сказать", "кто", 
    "уже", "когда", "сам", "что-то", "без", "чтобы", "там", "потому", "много", "во", "теперь", "люди", "нет", 
    "мой", "здесь", "такой", "даже", "знать", "тоже", "другой", "раз", "чем", "первый", "через", "теперь", 
    "перед", "после", "день", "жизнь", "очень", "еще", "время", "работа", "свой", "можно", "потом", "дом", 
    "слово", "пойти", "тогда", "место", "надо", "вообще", "каждый", "новый", "над", "ни", "стать", "тебя", 
    "мир", "также", "сегодня", "где", "говорить", "всегда", "почему", "об", "любить"
]

pattern_ru = compile_with_word_segmentation(frequent_russian_words)

frequent_hebrew_words = [
    "של", "הוא", "על", "לא", "זה", "את", "אני", "מה", "היא", "עם", "כל", "כי", "היה", "גם", "אבל", "אם", "או", 
    "יש", "אחד", "הם", "הן", "מי", "כמו", "עוד", "רק", "כבר", "הזה", "כך", "היה", "בא", "בין", "ב", "אני", "היו", 
    "הייתה", "אחר", "לה", "אשר", "ל", "ה", "הייתי", "היום", "כן", "יהיה", "היו", "זאת", "ב", "שלא", "אתה", 
    "שהוא", "הם", "שבו", "כדי", "כ", "שלה", "שלהם", "מ", "בכל", "ולא", "אותו", "יותר", "כלום", "כמו", "משום", 
    "שום", "לפני", "מאוד", "כאן", "לכל", "אין", "היה", "הייתה", "עצמו", "אל", "איך", "עכשיו", "בגלל", 
    "למה", "אם", "למה", "משום", "שלה", "שלהם", "ל", "רק", "הוא", "ב", "מתי", "ש", "לא", "איך", "מי", "הוא", 
    "אין", "משהו", "כמו", "אבל", "איך", "אל", "רק", "שלה", "שלהם", "יש", "אני"
]

pattern_he = compile_with_word_segmentation(frequent_russian_words)

python_operators_and_keywords = [
    "+", "-", "*", "/", "//", "%", "**",         # 算術演算子
    "==", "!=", ">", "<", ">=", "<=",           # 比較演算子
    "=", "+=", "-=", "*=", "/=", "%=", "//=", "**=",   # 代入演算子
    "&", "|", "^", "~", "<<", ">>",              # ビット演算子
    "and", "or", "not",                          # 論理演算子
    "is", "is not",                              # 同一性演算子
    "in", "not in",                              # メンバーシップ演算子
    "[", "]", "{", "}", "(", ")",                # 括弧
    ":", ",", ".", ";", "@",                     # その他の記号
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally", "for",
    "from", "global", "if", "import", "lambda", "nonlocal", "not", "or", "pass",
    "raise", "return", "try", "while", "with", "yield"
]

pattern_py = compile_with_word_segmentation(python_operators_and_keywords)


def dedup(words):
    dup = set()
    for w in words:
        for w2 in words:
            if w != w2 and w in w2:
                dup.add(w2)
    return [w for w in words if w not in dup]

def compile_without_word_boundaries(words):
    return re.compile(r'|'.join(map(re.escape, dedup(words))))

# 頻出日本語単語100個のリスト
frequent_japanese_words = [
    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ", "ある", "いる", "も", "する", "から", "な", "こと",
    "として", "い", "や", "など", "なっ", "ない", "この", "ため", "その", "あっ", "よう", "また", "もの", "という", "あ", "さらに",
    "でも", "なら", "なり", "なり", "でき", "これ", "さん", "して", "それ", "その", "どう", "どの", "わたし", "わたし",
    "あなた", "あの", "どれ", "いい", "あるいは", "しかし", "そう", "そして", "ただ", "だが", "だけど", "だから", "たとえば",
    "ちゃんと", "どうも", "だれか", "どうして", "どこか", "どんな", "どれか", "なぜなら", "やっぱり", "よい", "どうぞ",
    "ようやく", "それから", "すぐ", "あまり", "いちばん", "おおい", "おかしい", "おもい", "かならず", "かなり", "すべて",
    "とても", "または", "まず", "もっと", "やすい", "よく", "らしい", "か", "さ", "たち", "っぽい", "いくつ", "いくら",
    "いけない", "いっぱい", "うえ", "うまく", "おおぜい", "おなじ", "かのう", "きみ", "きゅう", "けっして", "こそ", "これら",
    "ご", "さらに", "すでに", "そのうち", "ただし", "たち", "ためし", "つい", "て", "どの", "なか", "など", "にたいして",
    "において", "にっぽん", "につれて", "のなか", "ほんとう", "ほんと", "まるで", "みたい", "むずかしい", "めったに",
    "もしくは", "やっぱり", "ようするに", "より", "わざと", "わたし", "わたしたち", "わかる", "わたし", "わたし"
]

pattern_ja = compile_without_word_boundaries(frequent_japanese_words)

# 頻出中国語単語100個のリスト
frequent_chinese_words = [
    "的", "一", "是", "在", "不", "了", "有", "和", "人", "这", "中", "大", "为", "上", "个", "国", "我", "以", "要", "他",
    "时", "来", "用", "们", "生", "到", "作", "地", "于", "出", "就", "分", "对", "成", "会", "可", "主", "发", "年", "动",
    "同", "工", "也", "能", "下", "过", "子", "说", "产", "种", "面", "而", "方", "后", "多", "定", "行", "学", "法", "所",
    "民", "得", "经", "十", "三", "之", "进", "着", "等", "部", "度", "家", "电", "力", "里", "如", "水", "化", "高", "自",
    "二", "理", "起", "小", "物", "现", "实", "加", "量", "都", "两", "体", "制", "机", "当", "使", "点", "从", "业", "本"
]

pattern_zh = compile_without_word_boundaries(frequent_chinese_words)

frequent_korean_words = [
    "이", "그", "저", "안", "있다", "없다", "좋다", "나쁘다", "크다", "작다", "많다", "적다", "새롭다", "오래되다", "높다", "낮다",
    "빠르다", "느리다", "쉽다", "어렵다", "같다", "다르다", "같이", "다르게", "그렇다", "그러다", "이렇다", "저렇다", "너무", "매우",
    "정말", "진짜", "아주", "조금", "좀", "많이", "적게", "또", "그리고", "그러나", "하지만", "그래서", "그런데", "그러니까", "즉", "그러면",
    "왜", "어떻게", "어디", "언제", "누구", "무엇", "어떤", "어느", "얼마나", "얼마", "몇", "어때", "이런", "그런", "저런", "이것", "그것",
    "저것", "여기", "거기", "저기", "이제", "오늘", "어제", "내일", "지금", "어떻게", "어디", "누구", "뭐", "왜", "얼마나", "몇", "나",
    "너", "그", "우리", "너희", "그들", "나의", "너의", "그의", "우리의", "너희의", "그들의", "이", "그", "저", "안", "있다"
]

pattern_ko = compile_without_word_boundaries(frequent_korean_words)

frequent_thai_words = [
    "ที่", "เป็น", "และ", "การ", "ใน", "มี", "ของ", "ได้", "ให้", "ว่า", "นี้", "ไม่", "ไป", "จะ", "มา", "ด้วย", "เรา",
    "แต่", "ก็", "เมื่อ", "หรือ", "จาก", "โดย", "เขา", "คุณ", "กัน", "นั้น", "ซึ่ง", "อย่าง", "ดี", "ต้อง", "แล้ว", "ถึง",
    "มาก", "คน", "อีก", "อยู่", "ทั้ง", "วัน", "ทำ", "บอก", "เข้า", "ดู", "เพื่อ", "รัก", "ออก", "ครั้ง", "แม่", "เงิน",
    "บ้าน", "ตอน", "ใจ", "พ่อ", "ถูก", "เด็ก", "น้ำ", "เรา", "รอ", "ให้", "พี่", "พูด", "คุณ", "ฟัง", "เอง", "ชื่อ",
    "ใช้", "เจอ", "ใคร", "เสีย", "ความ", "จริง", "เมือง", "ดีกว่า", "หลัง", "ให้", "เห็น", "มา", "ทาง", "หัว",
    "ข่าว", "เจ็บ", "เพราะ", "ทำไม", "เรื่อง", "มัน", "เกิด", "กลาย", "ตอบ", "ก่อน", "ร้าย", "หวัง", "ขอ",
    "พบ", "ถือ", "ลืม", "เจอ", "ใน", "ครั้ง", "ลูก", "เมื่อ", "กลับ", "ขาด", "เล็ก", "ใหญ่", "ตาม", "อัน",
    "หนี", "เดิน", "รัก"
]

pattern_th = compile_without_word_boundaries(frequent_thai_words)


# 正規表現パターンを事前にコンパイル

def patterns():
    d = {}
    for name, pat in globals().items():
        if name.startswith('pattern_'):
            d[name[8:]] = pat
    return d

all_patterns = patterns()

def count_all(text, patterns=all_patterns):
    results = Counter()
    for lang, pattern in patterns.items():
        matches = pattern.findall(text)
        # 一致する単語の数を数える
        results[lang] = len(matches)
    return results

def split_text(text, max_length=200):
    """
    指定されたルールに基づいてテキストを分割する関数。
    
    Parameters:
    text (str): 分割するテキスト
    max_length: 分割する単位

    Returns:
    list: 分割されたテキストのリスト
    """
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        if line.strip() == "":
            # 空行が来たら分割
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0
        elif current_length + len(line) + 1 > max_length:
            # 今までの行がmax_length文字を超えたら分割
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_length = len(line)
        else:
            # 行を追加
            current_chunk.append(line)
            current_length += len(line) + 1

    # 最後のチャンクを追加
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def recongnize(text):
    for text in split_text(text):
        c = count_all(text)
        print(text)
        print(c)
