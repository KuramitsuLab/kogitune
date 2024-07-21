from collections import Counter
import math
import unicodedata

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from scipy.spatial.distance import cosine
except:
    # ない場合は無視する
    pass

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def levenshtein_similarity(candidate, reference):
    distance = levenshtein_distance(candidate, reference)
    max_length = max(len(candidate), len(reference))
    return 1 - (distance / max_length)

def char_type(char):
    if ord(char) < 256:
        if char.isalpha():
            return 'ALPHA'
        if char.isdigit():
            return 'DIGIT'
        if char in '+-*/=<>|&~^_':
            return 'OP'
        return char
    else:
        cat = unicodedata.category(char)
        name = unicodedata.name(char)
        if cat.startswith('P'):
            return 'PUNCT'
        if cat.startswith('S'):
            return 'EMOJI'
        if name.startswith('HIRAGANA'):
            return 'HIRAGANA'
        if name.startswith('KATAKANA'):
            return 'KATAKANA'
        if name.startswith('CJK UNIFIED IDEOGRAPH'):
            return 'KANJI'
        if name.startswith('FULL'):
            return 'FULL'
        return 'OTHER'

def simple_tokenize(text):
    token=[]
    result = []
    def append():
        nonlocal token, result
        if len(token) > 0:
            s = ''.join(token)
            if s != ' ':
                result.append(s)
            token=[]

    prev_type = None
    for char in text:
        current_type = char_type(char)
        if prev_type and current_type != prev_type:
            if len(token) > 0:
                append()
        token.append(char)
        prev_type = current_type
    append()
    return result

def jaccard_similarity(candidate, reference, tokenize=simple_tokenize):
    # テキストを単語に分割してセットに変換
    set1 = set(tokenize(candidate))
    set2 = set(tokenize(reference))
    
    # 積集合と和集合を計算
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # Jaccard係数を計算
    return len(intersection) / len(union)


def cosine_similarity(candidate, reference, tokenize=simple_tokenize):
    # テキストを単語に分割
    words1 = tokenize(candidate)
    words2 = tokenize(reference)

    # 単語の出現回数をカウント
    word_count1 = Counter(words1)
    word_count2 = Counter(words2)

    # 共通の単語を抽出
    common_words = set(word_count1.keys()) & set(word_count2.keys())

    # 内積を計算
    numerator = sum(word_count1[word] * word_count2[word] for word in common_words)

    # ベクトルの大きさを計算
    sum1 = sum(word_count1[word]**2 for word in word_count1.keys())
    sum2 = sum(word_count2[word]**2 for word in word_count2.keys())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return numerator / denominator

## BLEU

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def count_ngrams(tokens, n):
    return Counter(ngrams(tokens, n))

def clip_count(candidate_ngrams, reference_ngrams):
    return {ngram: min(count, reference_ngrams[ngram]) for ngram, count in candidate_ngrams.items()}

def modified_precision(candidate, reference, n):
    candidate_ngrams = count_ngrams(candidate, n)
    reference_ngrams = count_ngrams(reference, n)
    clipped_counts = clip_count(candidate_ngrams, reference_ngrams)
    
    total_clipped_count = sum(clipped_counts.values())
    total_candidate_count = sum(candidate_ngrams.values())
    
    if total_candidate_count == 0:
        return 0
    
    return total_clipped_count / total_candidate_count

def brevity_penalty(candidate, reference):
    c = len(candidate)
    r = len(reference)
    
    if c > r:
        return 1
    else:
        return math.exp(1 - r/c)

def bleu_score(candidate:str, reference:str, n=4, tokenize=simple_tokenize):
    candidate = tokenize(candidate)
    reference = tokenize(reference)
    
    precisions = [modified_precision(candidate, reference, i) for i in range(1, n+1)]
    
    if any(p == 0 for p in precisions):
        return 0
    
    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / n)
    
    bp = brevity_penalty(candidate, reference)
    
    return bp * geometric_mean

# ROUGE-L

def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])

    return L[m][n]

def rouge_l(candidate, reference, tokenize=simple_tokenize):
    # 単語単位でトークン化
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)

    lcs_length = lcs(candidate_tokens, reference_tokens)

    # Precision, Recall, F1-scoreの計算
    precision = lcs_length / len(candidate_tokens) if len(candidate_tokens) > 0 else 0
    recall = lcs_length / len(reference_tokens) if len(reference_tokens) > 0 else 0
    
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    # return {
    #     "precision": precision,
    #     "recall": recall,
    #     "f1_score": f1_score
    # }
    return f1_score

# BERTScore

def get_bert_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.squeeze(0)

def bert_score(candidate, reference, model_type="cl-tohoku/bert-base-japanese-v2"):
    try:
        from bert_score import score
        P, R, F1 = score([candidate], [reference], lang="ja", model_type=model_type, verbose=True)        
        # return {
        #     "precision": P.mean().item(),
        #     "recall": R.mean().item(),
        #     "f1": F1.mean().item()
        # }
        return F1.mean().item()
    except:
        print('PLEASE pip install bert_score')

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModel.from_pretrained(model_type)
    ref_embeddings = get_bert_embeddings(reference, model, tokenizer)
    cand_embeddings = get_bert_embeddings(candidate, model, tokenizer)
    
    def cosine_similarity(a, b):
        return 1 - cosine(a, b)

    # Compute R-precision, R-recall, and F1
    precision_scores = torch.max(cosine_similarity(cand_embeddings[:, None], ref_embeddings[None, :]), dim=1)[0]
    recall_scores = torch.max(cosine_similarity(ref_embeddings[:, None], cand_embeddings[None, :]), dim=1)[0]
    
    precision = precision_scores.mean().item()
    recall = recall_scores.mean().item()
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0    
    # return {
    #     "precision": precision,
    #     "recall": recall,
    #     "f1": f1
    # }
    return f1


