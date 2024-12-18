import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from datetime import datetime

st.set_page_config(
    page_title="Document GPT",
    page_icon="💕"
)
st.title("Document GPT")

if "message" not in st.session_state:
    st.session_state["message"]=[]

@st.cache_data(show_spinner=True)
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
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
    return retriever

def send_message(message,role,save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["message"].append({"message":message,"role":role})

def paint_history():
    for message in st.session_state["message"]:
        send_message(message["message"],message["role"],save=False)

template = ChatPromptTemplate.from_messages(
    ("system","""
        All answer in Korean. Answer the question using ONLY the following context.
        If you don't know the answer just say you dont't know. Don't make anything up.

        Context: {context}
        """),
    ("human","{question}"),
)


st.markdown("""
6번째 챌린지에 오신 것을 환영합니다.
""")


with st.sidebar:
    file = st.file_uploader('파일을 올려주세요',type=["pdf","docx","txt"])

if file:
    retriever = embed_file(file)
    
    send_message("질문할 준비됏나!","ai",save=False)
    paint_history()
    
    message = st.chat_input('질문해봐라!')
    if message:
        send_message(message,'human',save=True)
        docs = retriever.invoke(message)
        docs = "\n\n".join(document.page_content for document in docs)
        prompt = template.format_messages(context=docs, question=message)
        prompt
else:
    st.session_state["message"]=[]