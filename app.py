import streamlit as st
import pandas as pd
import json
import re
import pickle
import numpy as np
from pythainlp.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 150
REVERSE_MAPPING = {0: 'negative', 1: 'neutral', 2: 'positive'}

def thai_text_processor(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'ไม่\s*', 'ไม่_', text) 
    words = word_tokenize(text, engine='newmm')
    return " ".join(words)

@st.cache_resource
def load_resources():
    try:
        model = load_model("sentiment_clstm_anti_overfit.keras")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Model file not found. Please ensure 'sentiment_clstm_anti_overfit.keras' and 'tokenizer.pkl' exist: {e}")
        return None, None

model, tokenizer = load_resources()

st.set_page_config(page_title="Deep Thai Sentiment", layout="centered")
st.title("Deep Thai Sentiment Analyzer")
st.markdown("**AI Assignment:** Powered by **Anti-Overfit C-LSTM (Deep Learning)**")
st.divider()

if model and tokenizer:
    mode = st.radio("Select mode:", ("Single Text", "Batch Upload (JSON)"))

    def predict_sentiment(texts):
        clean_texts = [thai_text_processor(t) for t in texts]
        seqs = tokenizer.texts_to_sequences(clean_texts)
        padded_seqs = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        probs = model.predict(padded_seqs)
        preds = np.argmax(probs, axis=1)
        return [REVERSE_MAPPING[p] for p in preds]

    if mode == "Single Text":
        user_input = st.text_area("Enter news or complaint text here:", height=150)
        
        if st.button("Analyze Sentiment"):
            if user_input.strip():
                with st.spinner('Analyzing context with C-LSTM...'):
                    prediction = predict_sentiment([user_input])[0]
                
                if prediction == 'positive':
                    st.success(f"**Prediction:** {prediction.upper()} (Positive / Compliment)")
                    st.balloons()
                elif prediction == 'negative':
                    st.error(f"**Prediction:** {prediction.upper()} (Negative / Complaint)")
                else:
                    st.info(f"**Prediction:** {prediction.upper()} (Neutral / General News)")
            else:
                st.warning("Please enter text before analyzing.")

    elif mode == "Batch Upload (JSON)":
        st.info("The JSON file must have a key named 'text'.")
        uploaded_file = st.file_uploader("Upload JSON file for prediction", type=["json"])
        
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                if 'text' not in data:
                    st.error("Key 'text' not found in JSON file.")
                else:
                    text_dict = data['text']
                    keys = list(text_dict.keys())
                    raw_texts = [text_dict[k] for k in keys]
                    
                    st.write(f"Found **{len(raw_texts)}** texts. Sending to Deep Learning Pipeline...")
                    
                    with st.spinner('Processing through C-LSTM network...'):
                        predictions = predict_sentiment(raw_texts)
                    
                    data['sentiment'] = {}
                    for i, k in enumerate(keys):
                        data['sentiment'][k] = predictions[i]
                        
                    st.success("Data analysis complete!")
                    
                    preview_df = pd.DataFrame({
                        'ID': keys, 
                        'Text': raw_texts, 
                        'Predicted': predictions
                    })
                    st.dataframe(preview_df)
                    
                    json_string = json.dumps(data, ensure_ascii=False, indent=4)
                    st.download_button(
                        label="Download JSON file with predictions",
                        file_name="sentiment_predictions_clstm.json",
                        mime="application/json",
                        data=json_string
                    )
            except Exception as e:
                st.error(f"An error occurred: {e}")