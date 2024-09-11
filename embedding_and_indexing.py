import re
import os
import uuid
import logging
import pinecone
import pdfplumber
import base64
import time
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_HOST = os.getenv('PINECONE_HOST')

# Initialize Cohere and Pinecone
cohere_embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-multilingual-v3.0")
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "mro4"

# Regular expressions for figure and page number detection
figure_pattern = re.compile(r'Fig\.\s*\d+-\d+', re.IGNORECASE)
page_number_pattern = re.compile(r'Page\s*\d+-\d+', re.IGNORECASE)

# Load and split documents
def load_and_split_documents(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    split_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for doc in documents:
        if doc.metadata.get('source', '').endswith('.pdf'):  # Ensure the file is a PDF
            with pdfplumber.open(doc.metadata['source']) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    figure_references = figure_pattern.findall(page_text) if page_text else []

                    # If the page contains figure references or very little text, convert it to an image
                    if figure_references or (page_text and len(page_text.strip()) < 200):  # Low-text page
                        page_metadata = doc.metadata.copy()
                        page_metadata['figure_reference'] = figure_references if figure_references else "Low-text page"
                        page_metadata['page_number'] = f"Page {page_num}"

                        # Store base64 image in the metadata
                        page_doc = Document(page_content=page_text or '', metadata=page_metadata)
                        split_docs.append(page_doc)

                        logger.info(f"Figure reference or low-text page found on page {page_num}: {figure_references}")
                    else:
                        # Split the text page normally
                        page_metadata = doc.metadata.copy()
                        page_metadata['page_number'] = f"Page {page_num}"
                        page_doc = Document(page_content=page_text, metadata=page_metadata)
                        split_chunks = text_splitter.split_documents([page_doc])
                        split_docs.extend(split_chunks)

    return split_docs

# Generate unique document ID
def generate_id(text):
    return str(uuid.uuid4())

# Index documents in Pinecone
def index_documents(docs):
    index = pc.Index("mro4", host=PINECONE_HOST)
    for doc in docs:
        text = doc.page_content
        
        #time.sleep(2)
        # Generate embedding and document ID
        embedding = cohere_embeddings.embed([text], input_type="classification")[0]
        doc_id = generate_id(text)

        # Add page number, figure reference, and text to metadata
        metadata = {
            "source": doc.metadata.get('source', 'unknown'),
            "page_number": doc.metadata.get('page_number', 'unknown'),
            "figure_reference": doc.metadata.get('figure_reference', 'none'),
            "text": text  # Store the entire text with the embedding
        }

        # Upsert to Pinecone with embedding and metadata
        index.upsert(vectors=[(doc_id, embedding, metadata)], namespace="ns1")
        logger.info(f"Document indexed with ID: {doc_id}")

