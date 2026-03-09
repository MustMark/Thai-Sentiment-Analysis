import streamlit as st
import pandas as pd
import json
import re
import joblib
from pythainlp.tokenize import word_tokenize

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
        model = joblib.load("weight.pkl")
        return model
    except Exception as e:
        st.error(f"Model file not found: {e}")
        return None

pipeline_model = load_resources()

st.set_page_config(page_title="Thai Sentiment Analysis", layout="centered")
st.title("Thai Sentiment Analyzer 🇹🇭")
st.markdown("**AI Assignment:** Powered by **Voting Ensemble (Sweet Spot Version)**")
st.divider()

if pipeline_model:
    mode = st.radio("Select mode:", ("Single Text", "Batch Upload (JSON)"))

    def predict_sentiment(texts):
        clean_texts = [thai_text_processor(t) for t in texts]
        predictions = pipeline_model.predict(clean_texts)
        return predictions

    if mode == "Single Text":
        user_input = st.text_area("Enter text here:", height=150)
        
        if st.button("Analyze Sentiment"):
            if user_input.strip():
                with st.spinner('Analyzing...'):
                    prediction = predict_sentiment([user_input])[0]
                
                if prediction == 'positive':
                    st.success(f"Prediction: {prediction.upper()}")
                elif prediction == 'negative':
                    st.error(f"Prediction: {prediction.upper()}")
                else:
                    st.info(f"Prediction: {prediction.upper()}")
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
                    
                    st.write(f"Found {len(raw_texts)} texts. Processing...")
                    
                    with st.spinner('Analyzing...'):
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
                        file_name="sentiment_predictions_final.json",
                        mime="application/json",
                        data=json_string
                    )
            except Exception as e:
                st.error(f"An error occurred: {e}")