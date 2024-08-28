# query_processing.py
import os
import pinecone
import logging
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
PINECONE_HOST = os.getenv('PINECONE_HOST')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Initialize Cohere
cohere_embeddings = CohereEmbeddings(cohere_api_key= COHERE_API_KEY, model="embed-multilingual-v3.0")
pc=pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Retrieve documents
def retrieve_documents(query, index_name, top_k=5):
    logger.info(query)
    index = pc.Index(index_name, host=PINECONE_HOST)
    query_embedding = cohere_embeddings.embed(texts=[query], input_type="search_query")[0]
    logger.info(query_embedding)
    results = index.query(vector=query_embedding, top_k=top_k,include_metadata=True,namespace="ns1")
    logger.info(results)
    matches = results['matches']
    logger.info(matches)
    return [match['metadata'] for match in matches]

