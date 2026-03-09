import streamlit as st
import pandas as pd

st.title("Text Sentiment Prediction")
"this project is a part of :rainbow-background[:rainbow[Artificial Intelligence course]]. The model used in this project is SVM"
""

st.subheader("Group Members")

members = {
    "ID": [
        "66010346",
        "66010850",
        "66010948",
        "66011442",
    ],
    "First Name": [
        "ธนัตถ์",
        "สิทธิวรรธน์",
        "แอมมีลี",
        "พีรวัส",
    ],
    "Last Name": [
        "ชัยพานนท์วิชญ์",
        "กุลชนะเจริญ",
        "โจว",
        "อิงคสันตติกุล",
    ],
    "Responsibility": [
        "None",
        "model study, frontend",
        "model study, frontend",
        "model study, deployment",
    ]
}
df = pd.DataFrame(members)
st.dataframe(df, hide_index=True, selection_mode=None, on_select="ignore")