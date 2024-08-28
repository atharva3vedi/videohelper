import pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_HOST =  os.getenv('PINECONE_HOST')

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "mro"
index = pc.Index(index_name, host=PINECONE_HOST)

# Fetch specific vectors by ID from the specified namespace
response = index.query(namespace='ns1',id=['b5480bd5-8c4b-442c-a1ed-fada1d4dadbf'], top_k=1, include_values=True, include_metadata=True)

# Print the index stats
stats = index.describe_index_stats()
print(stats)

# Print the fetched metadata
for match in response.get('vectors', {}).values():
    print(match['metadata'])  
