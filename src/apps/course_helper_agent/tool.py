import os
from typing import Optional
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import Filter, FieldCondition, MatchValue
import logging

logger = logging.getLogger(__name__)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

qdrant_url = os.getenv("QDRANT_URL", "")
qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

course_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="courses",
    url=qdrant_url,
    api_key=qdrant_api_key,
)


def retrieve_course_documents(
    query: str,
    course_id: str,
    k: Optional[int] = 5,
    score_threshold: Optional[float] = 0.1
) -> dict:
    """
    Retrieve relevant documents from vector store filtered by course_id.
    """
    k = max(1, min(k, 20))

    try:
        logger.info(f"Searching course '{course_id}' for: '{query}' (top {k} results)")

        course_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.course_id",
                    match=MatchValue(value=course_id)  # Changed to MatchValue
                )
            ]
        )

        results = course_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=course_filter
        )

        formatted_results = []
        for doc, score in results:
            similarity_score = 1 / (1 + score)

            if similarity_score >= score_threshold:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(similarity_score),
                    "distance": float(score)
                })

        logger.info(f"✓ Found {len(formatted_results)} relevant documents (threshold: {score_threshold})")

        return {
            "query": query,
            "course_id": course_id,
            "num_results": len(formatted_results),
            "results": formatted_results,
            "score_threshold": score_threshold
        }

    except Exception as e:
        error_msg = f"Error querying vector store: {str(e)}"
        logger.error(f"✗ {error_msg}")
        return {
            "error": error_msg,
            "query": query,
            "course_id": course_id,
            "num_results": 0,
            "results": []
        }