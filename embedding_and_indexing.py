# embedding_and_indexing.py
import os
import uuid
import logging
import pinecone
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_HOST =  os.getenv('PINECONE_HOST')

# Initialize Cohere and Pinecone
cohere_embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-multilingual-v3.0")
pc=pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "mro"

# Load and split documents
def load_and_split_documents(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(documents)

def generate_id(text):
    return str(uuid.uuid4())

# Index documents in Pinecone
def index_documents(docs):
    index = pc.Index("mro",host=PINECONE_HOST)
    for doc in docs:
        text = doc.page_content
        embedding = cohere_embeddings.embed([text],input_type="classification")[0]
        print(len(embedding))    
        print(type(doc))
        doc_id = generate_id(text)
        print("log1")
        index.upsert(vectors =  [(doc_id, embedding, doc.metadata)],namespace="ns1")
        print("log2")
