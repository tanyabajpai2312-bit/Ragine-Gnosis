 # Phase1 imports
import streamlit as st

 # Phase2 imports
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()  # Load env vars from .env

 # Phase3 imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

st.title("RAG Chatbot")

# Setup a session state variable to hold old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
# Disply all the historicl messages
for message in st.session_state.messages:
      st.chat_message(message['role']).markdown(message['content'])  
      
@st.cache_resource
def get_vectorstore():
        pdf_name="C:\Users\tanya\OneDrive\Documents\Desktop\Data\Gale Encyclopedia of Medicine 1.pdf"
        loaders=[PyPDFLoader(pdf_name)]
          # Create chunks,ka vectors(Chromadb)
        index=VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name='ll-MiniLM-L12-v2'),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
            ).from_loaders(loaders)
        return index.vectorstore
    
prompt = st.chat_input("pass your prompt here")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    
    groq_sys_prompt=ChatPromptTemplate.from_template("""You are very smart at everything.answer the following question: {user_prompt}Start the answer directly.No small talk""")
    model="llama3-8b-8192"
    
    groq_chat=ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model
    )
    try:
        vectorstore=get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load the document")
    
        chain=RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs=({"k: 3"})),
            return_source_documents=True
        )
            
        result=chain({"query":prompt})
        response=result["result"]
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
     


