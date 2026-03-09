import streamlit as st
import pandas as pd # type: ignore


pages = [
    st.Page(
        "pages/prediction.py",
        title="Prediction",
        icon=":material/graph_1:"
    ),
    st.Page(
        "pages/members.py",
        title="Group members",
        icon=":material/group:"
    ),
]

page = st.navigation(pages)
page.run()