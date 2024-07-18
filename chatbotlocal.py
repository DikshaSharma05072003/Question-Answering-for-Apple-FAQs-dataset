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
from htmlTemplates2 import new, bot_template, user_template, background_image, page_bg_img
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,LLMChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

#embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='')
embedding = OllamaEmbeddings(model="mistral") 
x = FAISS.load_local("D:\\sem6\\DL\\Innovative\\local_faiss_index", embedding, allow_dangerous_deserialization=True)

def get_conversation_chain(x):
    #llm = ChatOllama(model="llama2:7b")
    #llm = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    llm = Ollama(model="mistral")
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer",return_messages=True)
    system_template = r"""
    You're a helpful AI assistant focused on answering FAQs about Apple devices. Given a user query about a issue or FAQ about any 
    Apple product . Answer the questions correctly. If the query is not related to the Apple product just say that you didn't know 
    about this. If the query is related to greetings respond them correctly and shortly. 
    
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
        combine_docs_chain_kwargs={'prompt':prompt}
        #chain_type_kwargs={"prompt": prompt},
        )
    # st.conversation=rag_chain
    return rag_chain
    
state = st.session_state

def get_conversation_chain(x):
    # LLM and memory setup (unchanged)
    llm = Ollama(model="mistral")
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer", return_messages=True)
    # ... rest of the function (unchanged)

def handle_user_input(prompt):
    response = st.conversation({'question': prompt})
    st.chat_history = response['chat_history']

    # Display chat history on the left side
    with st.container():
        st.header("Chat History")
        for message in st.chat_history:
            if message.role == "user":
                st.markdown(user_template.format(question=message.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.format(answer=message.content), unsafe_allow_html=True)

    # Handle displaying and processing the latest message
    mes = st.chat_history[-1]
    typewriter(mes, user_template, 10)

def typewriter(text, template, speed):
    tokens = (text.content).split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(curr_full_text)
        time.sleep(1 / speed)


def submit():
    st.text_received = st.session_state.widget
    st.session_state.widget = ""
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
    st.title("Appple! ")
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