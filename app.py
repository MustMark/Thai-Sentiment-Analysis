import streamlit as st
import pandas as pd # type: ignore


pages = [
    st.Page(
        "pages/prediction.py",
        title="Prediction",
        icon=":material/home:"
    ),
    st.Page(
        "pages/members.py",
        title="Group members",
        icon=":material/group:"
    ),
    st.Page(
        "pages/model.py",
        title="model",
        icon=":material/graph_1:"
    ),
]

page = st.navigation(pages)
page.run()