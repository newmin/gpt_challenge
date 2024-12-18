import streamlit as st
from langchain.prompts import PromptTemplate
from datetime import datetime
import time

st.set_page_config(
    page_title="AI",
    page_icon="😘"
)

st.title("title")

if "message" not in st.session_state:
    st.session_state["message"]=[]

# st.write(st.session_state["message"])

# st.markdown("""
# #Hellow
    
# - [ ] [Document](/DocumentGPT)
# - [ ] [Private](/PrivateGPT)
# - [ ] [Quiz](/QuizGPT)
# """)


# with st.chat_message("human"):
#     st.write('Hi')

# with st.chat_message("ai"):
#     st.write('Fuck')

# with st.status("Embedding...",expanded=True) as status:
#     time.sleep(1)
#     "Gett"
#     time.sleep(1)
#     "embeding"
#     time.sleep(1)
#     "Caching"
#     status.update(label="Error", state="error")


def send_msg(msg,r, save=True):
    with st.chat_message(r):
        st.write(msg)
    if save:
        st.session_state["message"].append({"message":msg,"role":r})

for message in st.session_state["message"]:
    send_msg(message["message"],message["role"],save=False)

msg = st.chat_input("말걸어봐")
if msg:
    send_msg(msg,"human")
    time.sleep(2)
    send_msg(f"감히 {msg}라 했겠다?","ai")


with st.sidebar:
    st.title("side")
    st.text_input("입력")

    st.write(st.session_state)