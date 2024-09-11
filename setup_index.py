# setup_index.py
import os
import traceback
import logging
from pinecone import pinecone
from dotenv import load_dotenv
from embedding_and_indexing import load_and_split_documents, index_documents

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_HOST =  os.getenv('PINECONE_HOST')

def setup_index(directory='data', index_name='mro4'):
    try:
        # Step 1: Load and split the documents
        logger.info("Loading and splitting documents from directory: %s", directory)
        documents = load_and_split_documents(directory)
        logger.info("Loaded and split %d documents.", len(documents))
        logger.info(type(documents))
        # Step 2: Index the documents in Pinecone
        logger.info("Indexing documents in Pinecone under index name: %s", index_name)
        index_documents(documents)
        logger.info("Indexing completed successfully.")

        # Step 3: Verify the index (optional but recommended)
        #verify_index(index_name)

    except Exception as e:
        logger.error("An error occurred during the indexing process: %s", e)

def verify_index(index_name):
    # Initialize Pinecone
    pinecone_api_key = PINECONE_API_KEY
    pc=pinecone.Pinecone(api_key=pinecone_api_key)

    try:
        # Connect to the index
        index = pc.Index(index_name,host=PINECONE_HOST)
        
        # Fetch the number of vectors stored in the index
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)

        logger.info("Index '%s' contains %d vectors.", index_name, vector_count)
        
        # Optionally, perform a simple query to test
        if vector_count > 0:
             try:
                logger.info("trying")

                namespaces = stats.get('namespaces', {})
                list_namespace = list(namespaces.keys())
                logger.info(list_namespace)

                results = index.query(namespace = list_namespace[0],id="8c63baa2-d77e-46c0-9bc8-da88d6c8b884",top_k=1,include_values=True)
                logger.info(results)

                if results['matches']:
                    logger.info("Test query successful. Index is ready for use.")
                else:
                    logger.warning("Test query returned no results.")
             except KeyError as ke:
                logger.error("Key error encountered: %s", ke)
        else:
            logger.warning("The index is empty. Please check the document ingestion process.")
            
    except Exception as e:
        logger.error("An error occurred during index verification: %s", e)
        logger.error("Traceback: %s", traceback.format_exc())

if __name__ == "__main__":
    setup_index('data')
