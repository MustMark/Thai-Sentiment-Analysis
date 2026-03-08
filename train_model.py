import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from pythainlp.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

MAX_VOCAB_SIZE = 12000
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 64

def thai_text_processor(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'ไม่\s*', 'ไม่_', text) 
    words = word_tokenize(text, engine='newmm')
    return " ".join(words)

print("Loading and Preprocessing data...")
df = pd.read_json("dataset/train_sentiment.json") 
df['clean_text'] = df['text'].apply(thai_text_processor)

label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
reverse_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
df['label'] = df['sentiment'].map(label_mapping)

X_train_text, X_val_text, y_train, y_val = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights_array))
print(f"Class weights applied to handle imbalanced data: {class_weights_dict}")

print("Tokenizing text...")
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_val_seq = tokenizer.texts_to_sequences(X_val_text)

X_train = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_val = pad_sequences(X_val_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

print("Building Anti-Overfit C-LSTM Model...")
model = Sequential([
    Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    Dropout(0.45),
    Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(32, return_sequences=False)),
    Dropout(0.45),
    Dense(16, activation='relu'),
    Dropout(0.3), 
    Dense(3, activation='softmax')
])

optimizer = Adam(learning_rate=0.00037)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

print("\nTraining Model...")
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=30, 
    batch_size=64,
    validation_data=(X_val, y_val),
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)

print("\n" + "="*50)
print("ANTI-OVERFIT C-LSTM MODEL EVALUATION")
print("="*50)

y_train_pred_prob = model.predict(X_train)
y_train_pred = np.argmax(y_train_pred_prob, axis=1)

y_val_pred_prob = model.predict(X_val)
y_val_pred = np.argmax(y_val_pred_prob, axis=1)

y_train_true_labels = [reverse_mapping[val] for val in y_train]
y_train_pred_labels = [reverse_mapping[val] for val in y_train_pred]

y_val_true_labels = [reverse_mapping[val] for val in y_val]
y_val_pred_labels = [reverse_mapping[val] for val in y_val_pred]

print("\n[TRAINING SET RESULTS]:")
print(f"Accuracy: {accuracy_score(y_train_true_labels, y_train_pred_labels):.4f}")
print("Classification Report:\n", classification_report(y_train_true_labels, y_train_pred_labels))
print("Confusion Matrix:\n", confusion_matrix(y_train_true_labels, y_train_pred_labels))

print("-" * 50)

print("\n[VALIDATION SET RESULTS]:")
print(f"Accuracy: {accuracy_score(y_val_true_labels, y_val_pred_labels):.4f}")
print("Classification Report:\n", classification_report(y_val_true_labels, y_val_pred_labels))
print("Confusion Matrix:\n", confusion_matrix(y_val_true_labels, y_val_pred_labels))

model.save("sentiment_clstm_anti_overfit.keras")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("\nModel saved as 'sentiment_clstm_anti_overfit.keras' and 'tokenizer.pkl'")