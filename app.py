import streamlit as st
import pandas as pd # type: ignore
from models.logistic_regression.test import test_model as test_model1
from models.SVM.test import test_model as test_model2


"""
# Text Sentiment Prediction
"""

text_input = st.text_input(label="Text", value="", placeholder="Enter text here")
predict_btn = st.button("predict")

model_container = st.container(horizontal=True)

model1, model2 = model_container.columns(2, border=True)
        
with model1:   
    st.write("## Logistic Regression")
    model1_output = st.empty()

with model2:   
    st.write("## SVM")
    model2_output = st.empty()

if predict_btn:
    with model1_output.container():
        with st.spinner("generating result..."):
            output_text = test_model1(text_input)
        model1_output.write(output_text)
        
    with model2_output.container():
        with st.spinner("generating result..."):
            output_text = test_model2(text_input)
        model2_output.write(output_text)