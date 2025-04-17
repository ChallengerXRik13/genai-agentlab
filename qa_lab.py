import streamlit as st
import pandas as pd

st.title("🧪 QA Lab - Edit or Add Questions")
st.write("Modify synthetic QA pairs or input your own")

if "qa_data" not in st.session_state:
    st.session_state.qa_data = pd.DataFrame({
        "question": ["What is the document about?"],
        "answer": ["This document discusses..."]
    })

edited = st.data_editor(st.session_state.qa_data, num_rows="dynamic")
st.session_state.qa_data = edited
