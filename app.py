import streamlit as st
from langchain.prompts import PromptTemplate

st.set_page_config(
    page_title="여기는 AI국국",
    page_icon="😘"
)

st.title("반갑습니다. 낯선이여")

st.markdown("""
#Hellow
    
- [ ] [Document](/DocumentGPT)
- [ ] [Private](/PrivateGPT)
- [ ] [Quiz](/QuizGPT)
""")