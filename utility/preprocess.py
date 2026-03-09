import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords

import re, html
import json
import pandas as pd

stop_words = set(thai_stopwords())

def tokenizer(text):
    tokens = word_tokenize(text, engine="newmm")
    return [t for t in tokens if t not in stop_words]

def clean_eng(text):
    text = re.sub(r'[A-Za-z]', '', text)
    return text

def has_thai(text):
    return bool(re.search(r'[ก-๙]', text))

def handle_negation(tokens):
    new_tokens = []
    skip = False

    for i in range(len(tokens)):
        if skip:
            skip = False
            continue
            
        if tokens[i] == "ไม่" and i+1 < len(tokens):
            new_tokens.append("ไม่" + tokens[i+1])
            skip = True
        else:
            new_tokens.append(tokens[i])
    return new_tokens

# positive_words = ["ดี", "เยี่ยม", "สุดยอด", "คุ้ม", "อร่อย", "ชอบ", "ส่งเสริม", "สุข"]
# negative_words = ["แย่", "ห่วย", "พัง", "แพง", "ช้า", "ไม่ดี"]

# def lexicon_features(text):
#     pos = sum(word in text for word in positive_words)
#     neg = sum(word in text for word in negative_words)
#     return [pos, neg]