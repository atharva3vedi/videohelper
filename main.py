# main.py
import streamlit as st
from streamlit_chat import message
from query_processing import retrieve_documents
import logging
from response_generation import generate_response
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

st.title("RAG-based Chatbot")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize memory in session state if not already done
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

if 'responses' not in st.session_state:
    st.session_state['responses'] = []

user_query = st.text_input("Ask me anything:")

if user_query:
    with st.spinner("Thinking..."):
        context = retrieve_documents(str(user_query), index_name="mro")
        logger.info("context",context)
        response = generate_response(user_query,context, st.session_state.memory)     
        st.session_state['responses'].append({"query": user_query, "response": response})

for i, chat in enumerate(st.session_state['responses']):
    message(chat["response"], is_user=False, key=f"response_{i}")
    message(chat["query"], is_user=True, key=f"query_{i}")

if st.button("Clear Chat"):
    st.session_state['responses'] = []
    st.session_state.memory.clear()  # Clear the memory as well
    st.rerun()
