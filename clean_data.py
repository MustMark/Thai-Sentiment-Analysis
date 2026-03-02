import re, html
import json
import pandas as pd

def clean_text(text):
    
    text = html.unescape(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()

    return text

with open("dataset/train_sentiment.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print(df.head())

df["clean_text"] = df["text"].apply(clean_text)

df.to_json(
    "cleaned_dataset.json",
    orient="columns",
    force_ascii=False,
    indent=2
)