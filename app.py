# import streamlit as st

# st.set_page_config(
#     page_title="여기는 AI국국",
#     page_icon="😘"
# )

# st.title("반갑습니다. 낯선이여")

# st.markdown("""
# #Hellow
    
# - [x] [Document](/DocumentGPT)
# - [ ] [Private](/PrivateGPT)
# - [x] [Quiz](/QuizGPT)
# """)

import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import os
import nltk
import pickle

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

st.set_page_config(
    page_title="Document GPT",
    page_icon="💕"
)
# openai_api_key = st.secrets["openai_api_key"]
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# llm = ChatOpenAI(openai_api_key=openai_api_key,model="gpt-3.5-turbo", temperature=0)
# llm = ChatOpenAI(openai_api_key=openai_api_key,model_name="gpt-3.5-turbo",temperature=0)
# llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)

st.title("Document GPT")

if "message" not in st.session_state:
    st.session_state["message"]=[]

@st.cache_data(show_spinner=True)
@st.cache_resource
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)

    with open(file_path,"wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    # return pickle.dumps(vectorstore)
    return retriever

def embed_file2(file):
    vectorstore_data = embed_file_data(file)
    vectorstore = pickle.loads(vectorstore_data)
    retriever = vectorstore.as_retriever()
    return retriever

def send_message(message,role,save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["message"].append({"message":message,"role":role})

def paint_history():
    for message in st.session_state["message"]:
        send_message(message["message"],message["role"],save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [("system","""
     All answer in Korean. Answer the question using ONLY the following context.
     If you don't know the answer just say you dont't know. Don't make anything up.

        Context: {context}
        """),
    ("human","{question}")]
)


st.markdown("""
6번째 챌린지에 오신 것을 환영합니다.\n
사이드바에서 본인의 OPENAI_API_KEY를 사용하실 수 있습니다.
""")

file=False
with st.sidebar:

    st.write("https://github.com/newmin/gpt_challenge/blob/main/app.py")
    
    key = st.text_input('OPENAI_API_KEY를 입력하세요')
    st.markdown(f"{key}")
    if key:
        llm = ChatOpenAI(openai_api_key=key, model="gpt-3.5-turbo", temperature=0)
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    file = st.file_uploader('파일을 올려주세요',type=["pdf","docx","txt"])

if file:
    retriever = embed_file(file)
    
    send_message("질문할 준비됏나!","ai",save=False)
    paint_history()
    
    message = st.chat_input('질문해봐라!')
    if message:
        send_message(message,'human',save=True)
        chain = {
            "context":retriever | RunnableLambda(format_docs),
            "question" : RunnablePassthrough()
        } | prompt | llm
        rep = chain.invoke(message)
        send_message(rep.content,'ai')
else:
    st.session_state["message"]=[]
