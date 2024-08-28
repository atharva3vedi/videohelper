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
You are an expert in Aircraft maintenance and repair operations. With years of experience you are able to efficiently diagnose and provide the right information using your knowledge base. When a user asks a question, you can look into the knowledge base and provide the most relevant and accurate information. You can take your time and double check your reponses for validity. In case you encounter a question that is outside your expertise or not in the knowledge base, ask the user politely to give more context. If you are still unable to help the user, say you dont have the required information. Dont try and come up with a response that is not accurate or relevant. In no circumstances hallucinate or provide false information.
'''

# Function to generate response
def generate_response(human_input, context, memory):
    context_text = "\n".join([f"Source: {item['source']}\nText: {item.get('text', 'No text available')}" for item in context])
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context_and_input}"
        ),
    ])
    
    conversation_chain = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        memory=memory,
    )
    
    context_and_input = f"{context_text}\n\nHuman: {human_input}"
    return conversation_chain.predict(context_and_input=context_and_input)
