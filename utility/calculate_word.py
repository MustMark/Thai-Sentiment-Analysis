from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from collections import Counter
import re, html
import json
import pandas as pd

with open("dataset/train_sentiment.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

thai_stops = list(thai_stopwords())

custom_stops = [" ", "  ", "\n", "\t", "ๆ", "ฯ"]
all_stops = set(thai_stops + custom_stops)

def get_top_words(text_series, top_n=20):
    all_words = []
    for text in text_series:
        text = str(text)
        
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^ก-๙a-zA-Z0-9\s]', ' ', text)
        
        words = word_tokenize(text, engine="newmm")
        
        words = [w.strip() for w in words if w.strip() not in all_stops and len(w.strip()) > 1]
        all_words.extend(words)
        
    return Counter(all_words).most_common(top_n)

for sentiment_class in df["sentiment"].unique():
    print(f"\n{'='*30}")
    print(f"Top 50 Word : {sentiment_class.upper()}")
    print(f"{'='*30}")
    
    texts_in_class = df[df["sentiment"] == sentiment_class]["text"]
    top_words = get_top_words(texts_in_class, top_n=50)
    
    for rank, (word, count) in enumerate(top_words, 1):
        print(f"{rank:>2}. {word:<15} (Found : {count})")