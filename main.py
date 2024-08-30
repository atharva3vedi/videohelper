import os
import streamlit as st
import streamlit.components.v1 as components
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from markdown2 import markdown
import logging
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from query_processing import retrieve_documents
from response_generation import generate_response


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Aircraft Maintenance Copilot", page_icon="✈️", layout="wide")

# Model options for the sidebar
model_options = {
    "LLaMA 8B": "llama3-8b-8192",
    "LLaMA 8B Instant": "llama-3.1-8b-instant",
    "LLaMA 70B": "llama3-70b-8192",
    "LLaMA 70B Versatile": "llama-3.1-70b-versatile",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "LLaMA 504B Reasoning": "llama-3.1-405b-reasoning",
    "Gemma 7B IT": "gemma-7b-it",
    "Gemma 2 9B IT": "gemma2-9b-it"
}

# Sidebar for model selection
model = st.sidebar.selectbox("Select Model", list(model_options.keys()))
selected_model = model_options[model]

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("API key not found. Please set the GROQ_API_KEY in your .env file.")
    st.stop()

groq_chat = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name=selected_model,
)

# Ensure memory is maintained across user interactions
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

# Set up the Streamlit interface
st.title("Aircraft Maintenance Copilot")

# Placeholder for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize user input in session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Function to handle user input and generate response
def handle_user_input():
    user_input = st.session_state.user_input
    
    if user_input:
        with st.spinner("Thinking..."):
            context = retrieve_documents(str(user_input), index_name="mro")
            logger.info("context",context)
            response = generate_response(user_input,context, st.session_state.memory)     
        
        # Format the chat messages using Markdown
        user_message = f"**You:** {user_input}\n"
        bot_message = f"**Bot:** {response}\n\n---\n"
        
        # Convert to HTML
        user_message_html = markdown(user_message)
        bot_message_html = markdown(bot_message)
        
        # Update chat history with proper HTML formatting
        st.session_state.chat_history.append(user_message_html)
        st.session_state.chat_history.append(bot_message_html)
        
        # Clear the input field
        st.session_state.user_input = ""  # Reset the input value to empty

# Chat UI setup
chat_container = st.container()
with chat_container:
    # Convert chat history to a single HTML string
    chat_messages = "\n".join(st.session_state.chat_history[-50:])  # Show only the last 50 messages for performance

    # Display the chat messages with custom CSS for scrolling and text styling
    components.html(
        f"""
        <div id="chat-window" style="height:500px; width:95%;color: black; overflow-y:auto; padding:10px; border:1px solid #ccc; border-radius:5px; font-family: 'Arial', sans-serif; font-size:16px; line-height:1.6;">
            {chat_messages}
        </div>
        <script>
        var chatWindow = document.getElementById('chat-window');
        chatWindow.scrollTop = chatWindow.scrollHeight;
        </script>
        """,
        height=500
    )

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()  # Clear the memory as well
        st.rerun()  # Rerun the script to refresh the chat window

# User input handling with on_change trigger
st.text_input("You:", value=st.session_state.user_input, key="user_input", on_change=handle_user_input)
