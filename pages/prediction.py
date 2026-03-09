import streamlit as st
from utility.json_handler import handle_json_file
from models.test import test_model
import json
import pandas as pd

# intro
st.title("Text Sentiment Prediction")
"this project is a part of :rainbow-background[:rainbow[Artificial Intelligence course]]. The model used in this project is SVM"
st.link_button("github", "https://github.com/MustMark/Thai-Sentiment-Analysis", icon=":material/open_in_new:")
""

# inputs
with st.container(horizontal=True, vertical_alignment='bottom', horizontal_alignment="right"):
    text_input = st.text_area(label="Enter text", value="", placeholder="Enter text here")
    predict_btn = st.button("predict", key="text", type="primary")


# model result
model_container = st.container()

if predict_btn: 
    with model_container:
        with st.spinner("generating result..."):
            output_text = test_model([text_input])[0]
        if output_text == "positive":
            st.success("Positive", icon=":material/sentiment_excited:")
        elif output_text == "neutral":
            st.warning("Neutral", icon=":material/sentiment_neutral:")
        elif output_text == "negative":
            st.error("Negative", icon=":material/sentiment_sad:")

with st.container(horizontal=True, vertical_alignment="center"):
    st.divider()
    st.text("or")
    st.divider()

with st.container(horizontal_alignment="right"):
    json_file = st.file_uploader("Upload a json file", type='json', accept_multiple_files=False)
    json_predict_btn = st.button("predict", key="json", type="primary")
    if json_predict_btn and json_file:
        with st.spinner("generating result"):
            result_json = handle_json_file(json_file)
            
        st.success("Prediction complete")
        
        json_data = json.dumps(result_json, ensure_ascii=False, indent=2)

        st.download_button(
            label="Download result JSON",
            data=json_data,
            file_name="sentiment_result.json",
            mime="application/json"
        )
        st.json(json_data)