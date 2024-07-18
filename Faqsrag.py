import streamlit as st
from streamlit_mic_recorder import mic_recorder, speech_to_text
from streamlit_chat import message
import numpy as np
import pyttsx3
from dotenv import load_dotenv
import time
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
import os
import getpass
from htmlTemplates2 import new, bot_template, user_template, background_image, page_bg_img
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,LLMChain
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
import openai
from langchain_community.vectorstores import FAISS

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='')
x = FAISS.load_local("D:\\sem6\\DL\\Innovative\\faiss_index", embeddings, allow_dangerous_deserialization=True)

def get_conversation_chain(x):
    llm = GoogleGenerativeAI(model="gemini-1.0-pro",google_api_key='')
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer",return_messages=True)
    system_template = r"""
    You're a helpful AI assistant focused on answering FAQs about Apple devices. Given a user query about a issue or FAQ about any 
    Apple product . Answer the questions correctly. If the query is not related to the Apple product just say that you didn't know 
    about this. If the query is related to greetings respond them correctly. 
    
    Here are the questions related to the product:
    --------
    {context}
    --------
    """
    user_template = "Question:```{question}```"
    messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template),
            HumanMessage(
            content=(
                "Tips: If you don't find any related query find the most similar ones to the description."
                "and don't answer any question which does not fall in context"
                    )
                        )
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=x.as_retriever(),
        combine_docs_chain_kwargs={'prompt':prompt})
    # st.conversation=rag_chain
    return rag_chain
    
state = st.session_state

def handle_user_input(prompt):
    response = st.conversation({'question': prompt})
    st.chat_history = response['chat_history']
    mes = st.chat_history[-1]
    typewriter(mes, user_template, 10)
    # engine.say(mes)
    # engine.runAndWait()
    # st.write(mes.content)

def typewriter(text, template, speed):
   
    tokens = (text.content).split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(curr_full_text)
        time.sleep(1 / speed)
        
def submit():
        st.text_received=st.session_state.widget
        st.session_state.widget=""
        st.conversation = get_conversation_chain(x)
        handle_user_input(st.text_received)
def submit1():
        # st.text_received=st.session_state.widget
        # st.session_state.widget=""
        st.conversation = get_conversation_chain(x)
        handle_user_input(st.text_received)

def main():
    load_dotenv()
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.write(new, unsafe_allow_html=True)
    st.title("Appple! 🛒")
    st.subheader("Your Apple FAQs AI assistant")

    st.option = st.selectbox('Select the mode of input?', ('Text', 'Voice'))
    if st.option=='Voice':
        st.write('You selected:', st.option)

        st.text_received = ""
        c1, c2 = st.columns(2)
    
        with c1:
            st.write("Tell me, I'm listening:")
        with c2:
            text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
        # st.write(text)
        st.text_received=text
        inp1=st.text_area(label="Your Query",value=text,key="widget", disabled=True)
        st.button(on_click=submit1, label="Submit")
        
    elif st.option=='Text':
        inp=st.text_input(label="Your Query",key='widget',on_change=submit, value='')


if __name__ == "__main__":
    main()