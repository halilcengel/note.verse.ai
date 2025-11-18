import logging
from typing import Any, Dict
from langchain_core.messages import HumanMessage, AIMessage

from ..state import State
from ..tool import retrieve_course_documents

logger = logging.getLogger(__name__)


def retrieve_node(state: State) -> Dict[str, Any]:
    """
    Retrieval node for RAG pipeline.

    Extracts the user's query from messages, retrieves relevant documents
    from vector store filtered by course_id, and formats context.

    Args:
        state: Current agent state

    Returns:
        Updated state with retrieved documents and formatted context
    """
    logger.info("=== Retrieval Node ===")

    messages = state.get("messages", [])
    if not messages:
        logger.warning("No messages found in state")
        return {
            "retrieved_documents": [],
            "context": "",
            "needs_retrieval": False
        }

    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)

    course_id = state.get("course_id")
    if not course_id:
        logger.error("No course_id provided in state")
        return {
            "retrieved_documents": [],
            "context": "Error: No course_id provided",
            "needs_retrieval": False
        }

    logger.info(f"Query: '{query}'")
    logger.info(f"Course ID: '{course_id}'")

    retrieval_result = retrieve_course_documents(
        query=query,
        course_id=course_id,
        k=5,
        score_threshold=0.5
    )

    if "error" in retrieval_result:
        logger.error(f"Retrieval error: {retrieval_result['error']}")
        return {
            "retrieved_documents": [],
            "context": f"Error retrieving documents: {retrieval_result['error']}",
            "needs_retrieval": False
        }

    documents = retrieval_result.get("results", [])
    logger.info(f"Retrieved {len(documents)} documents")

    if not documents:
        context = "No relevant course materials found for this query."
    else:
        context_parts = ["# Retrieved Course Materials\n"]

        for i, doc in enumerate(documents, 1):
            content = doc["content"]
            metadata = doc.get("metadata", {})
            score = doc.get("relevance_score", 0)

            context_parts.append(f"## Document {i} (Relevance: {score:.2f})")

            if metadata:
                context_parts.append("**Metadata:**")
                for key, value in metadata.items():
                    if key != "course_id":
                        context_parts.append(f"- {key}: {value}")

            context_parts.append(f"\n**Content:**\n{content}\n")
            context_parts.append("---\n")

        context = "\n".join(context_parts)

    logger.info(f"Context length: {len(context)} characters")

    return {
        "retrieved_documents": documents,
        "context": context,
        "needs_retrieval": False
    }
