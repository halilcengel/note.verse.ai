import os

from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings


embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
)

qdrant_url = os.getenv("QDRANT_URL", "")
qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="school_data",
        url=qdrant_url,
        api_key=qdrant_api_key,
)


