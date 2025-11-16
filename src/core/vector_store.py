from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings


embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
)

store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="school_data",
        url="http://localhost:6333",
)


