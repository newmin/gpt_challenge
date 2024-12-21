import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnableMap
from langchain.schema import BaseOutputParser, output_parser
import json

class JsonOutputParser(BaseOutputParser):
    def parse(self,text):
        text = text.replace("```json","").replace("```","")
        return json.loads(text)

output_parser = JsonOutputParser()

st.set_page_config(
    page_title='Quiz생성AI',
    page_icon='❤️'
)
st.title("Quiz생성AI")

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}
st.sidebar.write("https://github.com/newmin/gpt_challenge/blob/main/app.py")
openai_api_key = st.sidebar.text_input("당신의 OPENAI API KEY를 입력하십시오")
if openai_api_key:
    try:
        llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-3.5-turbo-1106",
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=openai_api_key
        ).bind(
            function_call={
                "name": "create_quiz",
            },
            functions=[
                function,
            ],
        )
    except Exception as e:
        st.error("Check your OpenAI API Key or File")
        st.write(e)

    level = None

    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)

    question_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
            You are a helpful assistant that is role playing as a teacher.
                    
            Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

            Each question should have 4 answers, three of them must be incorrect and one should be correct.
                    
            Use (o) to signal the correct answer.
                    
            단, 난이도의 경우 {level}가 풀 수 있는 난이도로 생성하시오.

            Question examples:
                    
            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(o)
                    
            Question: What is the capital or Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut
                    
            Question: When was Avatar released?
            Answers: 2007|2001|2009(o)|1998
                    
            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o)|Painter|Actor|Model
                    
            단, 모든 Question과 Answers은 한글로 작성해.
            Your turn!
                    
            Context: {{context}}
            """,
            )
        ]
    )


    question_chain = RunnableMap({"context":format_docs}) | question_prompt | llm

    formatting_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are a powerful formatting algorithm.
        
        You format exam questions into JSON format.
        Answers with (o) are the correct ones.
        
        Example Input:
        Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue(o)
            
        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi(o)|Manila|Beirut
            
        Question: When was Avatar released?
        Answers: 2007|2001|2009(o)|1998
            
        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o)|Painter|Actor|Model
        
        
        Example Output:
        
        ```json
        {{ "questions": [
                {{
                    "question": "What is the color of the ocean?",
                    "answers": [
                            {{
                                "answer": "Red",
                                "correct": false
                            }},
                            {{
                                "answer": "Yellow",
                                "correct": false
                            }},
                            {{
                                "answer": "Green",
                                "correct": false
                            }},
                            {{
                                "answer": "Blue",
                                "correct": true
                            }},
                    ]
                }},
                            {{
                    "question": "What is the capital or Georgia?",
                    "answers": [
                            {{
                                "answer": "Baku",
                                "correct": false
                            }},
                            {{
                                "answer": "Tbilisi",
                                "correct": true
                            }},
                            {{
                                "answer": "Manila",
                                "correct": false
                            }},
                            {{
                                "answer": "Beirut",
                                "correct": false
                            }},
                    ]
                }},
                            {{
                    "question": "When was Avatar released?",
                    "answers": [
                            {{
                                "answer": "2007",
                                "correct": false
                            }},
                            {{
                                "answer": "2001",
                                "correct": false
                            }},
                            {{
                                "answer": "2009",
                                "correct": true
                            }},
                            {{
                                "answer": "1998",
                                "correct": false
                            }},
                    ]
                }},
                {{
                    "question": "Who was Julius Caesar?",
                    "answers": [
                            {{
                                "answer": "A Roman Emperor",
                                "correct": true
                            }},
                            {{
                                "answer": "Painter",
                                "correct": false
                            }},
                            {{
                                "answer": "Actor",
                                "correct": false
                            }},
                            {{
                                "answer": "Model",
                                "correct": false
                            }},
                    ]
                }}
            ]
        }}
        ```
        Your turn!
        Questions: {context}
    """,
            )
        ]
    )

    formatting_chain = formatting_prompt | llm

    @st.cache_data(show_spinner="Loading file...")
    def split_file(file):
        file_content = file.read()
        file_path = f"./.cache/quiz_files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        return docs

    @st.cache_data(show_spinner='퀴즈가 생성 중입니다. 잠시만 기다려주십시오...')
    def run_quiz_chain(_docs,topic,level):
        chain = question_chain 
        # chain = {"context":question_chain} | formatting_chain | output_parser
        response = chain.invoke(_docs)
        # st.write(response)
        arguments = json.loads(response.additional_kwargs["function_call"]["arguments"])
        # st.write(arguments)
        
        return arguments

    @st.cache_data(show_spinner='wiki 검색중입니다. 잠시만 기다려주십시오...')
    def wiki_search(term):
        retriver = WikipediaRetriever(top_k_results=3)
        return retriver.invoke(term)

    with st.sidebar:
        docs = None
        if openai_api_key:

            choice = st.selectbox("사용 방식",[
                "파일",
                "위키피디아"
            ])

            if choice == "파일":
                file = st.file_uploader(
                    "파일 업로드",
                    type=['pdf','docx','txt']
                )
                if file:
                    docs = split_file(file)

            else :
                topic = st.text_input('위키피디아 검색어')
                if topic:
                    docs = wiki_search(topic)

    if not docs :
        st.markdown(
            """
    그 후 문제를 생성하기 위한 방식을 선택하고, 자료 첨부 혹은 검색어를 입력해주십시오.
    """
        )
    else :
        def initStatus():
            st.session_state.isPerfect = False

        level = st.sidebar.radio(
            '난이도를 선택해주십시오',
            ['쫄보','일반인','전문가','아인슈타인'],
            index=None,
            key='level_radio',
            on_change=initStatus
            )
        if not level:
            st.markdown("""
            문제의 난이도를 선택해주십시오.
                        """)
        else :
            o = 0
            x = 0
            # if "isPerfect" not in st.session_state:
            #     st.session_state.isPerfect = False

            def checkPerfect(question_length,correct_cnt):
                if question_length==correct_cnt:
                    st.session_state.isPerfect = True
                    st.balloons()

            response = run_quiz_chain(docs, topic if topic else file.name, level)
            # st.write(response)
            with st.form("question_form"):
                for idx, question in enumerate(response["questions"]):
                    st.write(idx+1,')',question["question"])
                    value = st.radio(
                        label="맞춰봐",
                        options=[answers["answer"] for answers in question["answers"]],
                        index=None,
                        key=f'answer_radio_{idx}'
                        )
                    if {"answer":value,"correct":True} in question["answers"]:
                        st.success('딩동댕동')
                        o += 1
                    elif value is not None:
                        st.error('내동댕이')
                        x += 1
                        
                button = st.form_submit_button(
                    label='제출',
                    on_click=checkPerfect(len(response["questions"]),o),
                    disabled=st.session_state.isPerfect,
                )
                # if len(response["questions"]) == o:
                #     st.session_state.isPerfect = True
                #     st.balloons()
else :
    st.markdown("""
    좌측 사이드바에서 당신의 openai_api_key를 입력하세요.
                """)