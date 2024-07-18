from langchain_community.vectorstores import FAISS
import os
import getpass
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain,LLMChain
from langchain.docstore.document import Document
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import OllamaEmbeddings

import openai

#os.environ['GOOGLE_API_KEY'] = ''
def get_chunks(j):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(j)
    return chunks


def get_vectorstore(chunks):
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embedd = OllamaEmbeddings(model="mistral") 
    d=Document(page_content=chunks[0].page_content)
    d1=[d]
    vectorstore = FAISS.from_documents(documents=d1, embedding=embedd)
    for x in range(1,len(chunks)):
      y=Document(page_content=chunks[x].page_content)
      y1=[y]
      v=FAISS.from_documents(documents=y1, embedding=embedd)
      vectorstore.merge_from(v)
      time.sleep(1)
    return vectorstore

loader = CSVLoader("D:\sem6\DL\Innovative\Apple_QandA\QandA.csv", encoding="windows-1252")
documents = loader.load()

#print(documents[0])

chunks=get_chunks(documents)
x=get_vectorstore(chunks)
x.save_local("local_faiss_index")