import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pythainlp.tokenize import word_tokenize

def thai_text_processor(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'ไม่\s*', 'ไม่_', text)
    words = word_tokenize(text, engine='newmm')
    return " ".join(words)

print("Loading and Preprocessing data...")
df = pd.read_json("../dataset/train_sentiment_cleaned.json")
df['clean_text'] = df['text'].apply(thai_text_processor)

X_train, X_val, y_train, y_val = train_test_split(
    df['clean_text'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)

word_vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 3),
    max_features=6000,   
    min_df=2,            
    sublinear_tf=True
)

char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 4),  
    max_features=10000,   
    min_df=2,
    sublinear_tf=True
)

combined_features = FeatureUnion([
    ('word_tfidf', word_vectorizer),
    ('char_tfidf', char_vectorizer)
])

clf1 = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)

svc = LinearSVC(C=0.4, class_weight='balanced', random_state=42)
clf2 = CalibratedClassifierCV(svc)

clf3 = LGBMClassifier(
    n_estimators=400,      
    learning_rate=0.03, 
    num_leaves=25,         
    max_depth=6,           
    min_child_samples=40,  
    reg_alpha=1.0,         
    reg_lambda=1.5,        
    random_state=42
)

voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('svc', clf2), ('lgbm', clf3)],
    voting='soft',
    weights=[1.5, 1.0, 1.8]
)

pipeline = Pipeline([
    ('features', combined_features),
    ('classifier', voting_clf)
])

print("Training Final Traditional ML Pipeline (Sweet Spot)...")
pipeline.fit(X_train, y_train)

print("\n" + "="*50)
print("FINAL MODEL EVALUATION")
print("="*50)

y_train_pred = pipeline.predict(X_train)
y_val_pred = pipeline.predict(X_val)

print("\n[TRAINING SET RESULTS]:")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print("Classification Report:\n", classification_report(y_train, y_train_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

print("-" * 50)

print("\n[VALIDATION SET RESULTS]:")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print("Classification Report:\n", classification_report(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

joblib.dump(pipeline, "weight.pkl")
print("\nFinal Model saved as 'weight.pkl'")