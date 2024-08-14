# response_generation.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage


load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

model_name = "llama3-8b-8192"  # Adjust based on your requirement

# Initialize Groq LLM
groq_chat = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=model_name,
    temperature=0.2,
    max_tokens=1024
)

# Create the system prompt template
system_prompt = '''
You are an AI assistant that helps users with their queries. You provide concise and helpful information. If the user asks for information outside of your knowledge base, please politely let them know you cannot assist.
'''

# Function to generate response
def generate_response(human_input, memory):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )
    
    conversation_chain = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        memory=memory,
    )
    
    return conversation_chain.predict(human_input=human_input)
