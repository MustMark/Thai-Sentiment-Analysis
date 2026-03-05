import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.util import normalize

def preprocess1(text: str):
    tokens = [" ".join(word_tokenize(text))]
    return tokens

stopwords = set(thai_stopwords())

def preprocess2(text: str):
    tokens = tokenizer(text)
    tokens = [" ".join(tokens)]
    return tokens

def tokenizer(text: str):
    # 1️⃣ Normalize Thai text (fix duplicated vowels/tones)
    text = normalize(text)
    # 2️⃣ Lowercase (important if mixed English)
    text = text.lower()
    # 3️⃣ Remove numbers
    text = re.sub(r'\d+', '', text)
    # 4️⃣ Remove special characters
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z\s]', '', text)
    # 5️⃣ Tokenize
    tokens = word_tokenize(text, engine="newmm")
    # 6️⃣ Remove stopwords
    tokens = [word for word in tokens if word not in stopwords]
    # # 7️⃣ Remove short words (optional)
    # tokens = [word for word in tokens if len(word) > 1]
    # tokens = handle_negation(tokens)
    
    return tokens

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