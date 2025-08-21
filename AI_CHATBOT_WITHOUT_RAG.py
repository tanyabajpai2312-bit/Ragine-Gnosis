# Phase1 imports
import streamlit as st
 # Phase2 imports
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()  # Load env vars from .env

st.title("RAG Chatbot")

# Setup a session state variable to hold old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
# Disply all the historicl messages
for message in st.session_state.messages:
      st.chat_message(message['role']).markdown(message['content'])  
      
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
    chain=groq_sys_prompt | groq_chat | StrOutputParser()
    response = chain.invoke({"user_prompt":prompt})
    
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content': response})
     


