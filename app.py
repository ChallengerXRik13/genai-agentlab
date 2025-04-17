import streamlit as st
from backend.api import query_agent_lab

st.title("🤖 GenAI AgentLab")
uploaded_file = st.file_uploader("Upload a document (PDF)", type=["pdf"])

if uploaded_file:
    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Thinking..."):
            answer = query_agent_lab(uploaded_file, query)
            st.success(answer)
