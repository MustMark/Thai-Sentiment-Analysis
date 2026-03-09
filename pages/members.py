import streamlit as st
import pandas as pd

st.title("Text Sentiment Prediction")
"This project is a part of the :rainbow-background[:rainbow[Artificial Intelligence Course]] in Computer Engineering at King Mongkut's Institute of Technology Ladkrabang."
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
        "clean dataset",
        "preprocess, model study",
        "model training, frontend",
        "model training, deployment",
    ]
}
df = pd.DataFrame(members)
st.dataframe(df, hide_index=True, selection_mode=None, on_select="ignore")