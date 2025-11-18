from typing import Annotated, TypedDict, Optional

from langgraph.graph import add_messages


class State(TypedDict):
    """State for course helper RAG agent.

    Attributes:
        messages: Conversation history
        course_id: ID of the course to filter documents by
        retrieved_documents: Documents retrieved from vector store
        context: Formatted context for LLM
        needs_retrieval: Flag to determine if retrieval is needed
    """
    messages: Annotated[list, add_messages]
    course_id: str
    retrieved_documents: Optional[list]
    context: Optional[str]
    needs_retrieval: bool