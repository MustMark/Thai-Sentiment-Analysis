import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.util import normalize
import re, html
import json
import pandas as pd

def preprocess1(text: str):
    tokens = [" ".join(word_tokenize(text))]
    return tokens

stop_words = set(thai_stopwords())

def preprocess2(text: str):
    tokens = tokenizer(text)
    tokens = [" ".join(tokens)]
    return tokens

def tokenizer(text):
    tokens = word_tokenize(text, engine="newmm")
    return [t for t in tokens if t not in stop_words]

def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # text = re.sub(r'[^ก-๙\s]', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def clean_eng(text):
    text = re.sub(r'[A-Za-z]', '', text)
    return text

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