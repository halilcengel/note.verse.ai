"""
Utility functions for the Course Helper Agent.
"""

from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


def create_initial_state(
    course_id: str,
    question: str,
    conversation_history: Optional[List[BaseMessage]] = None
) -> Dict[str, Any]:
    """
    Create initial state for the course helper agent.

    Args:
        course_id: The course ID to query
        question: The user's question
        conversation_history: Optional previous messages for context

    Returns:
        Initial state dictionary
    """
    messages = conversation_history or []
    messages.append(HumanMessage(content=question))

    return {
        "messages": messages,
        "course_id": course_id,
        "retrieved_documents": None,
        "context": None,
        "needs_retrieval": True
    }


def create_session_config(session_id: str, course_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create configuration for agent session with conversation memory.

    Args:
        session_id: Unique identifier for the conversation session
        course_id: Optional course ID to include in thread ID

    Returns:
        Configuration dictionary for the agent
    """
    thread_id = f"session-{session_id}"
    if course_id:
        thread_id += f"-{course_id}"

    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


def format_conversation_history(messages: List[BaseMessage]) -> str:
    """
    Format conversation history for display.

    Args:
        messages: List of messages from the conversation

    Returns:
        Formatted string of the conversation
    """
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"Student: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
        else:
            formatted.append(f"System: {msg.content}")

    return "\n\n".join(formatted)


def extract_sources_from_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract source information from retrieved documents.

    Args:
        documents: List of retrieved document dictionaries

    Returns:
        List of source information dictionaries
    """
    sources = []
    for doc in documents:
        metadata = doc.get("metadata", {})
        sources.append({
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page"),
            "section": metadata.get("section"),
            "relevance_score": doc.get("relevance_score", 0),
        })

    return sources


def estimate_token_count(text: str) -> int:
    """
    Rough estimation of token count for text.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    # Rough estimation: ~4 characters per token for English text
    return len(text) // 4


def should_retrieve(question: str, conversation_history: Optional[List[BaseMessage]] = None) -> bool:
    """
    Determine if retrieval is needed for a question.

    Simple heuristic: retrieval is needed unless it's a very basic greeting
    or acknowledgment. You can enhance this with an LLM call for better classification.

    Args:
        question: The user's question
        conversation_history: Optional conversation history

    Returns:
        Boolean indicating if retrieval should be performed
    """
    # Simple heuristics - can be enhanced with LLM classification
    greetings = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"]
    question_lower = question.lower().strip()

    # If it's just a greeting or thanks, no retrieval needed
    if question_lower in greetings:
        return False

    # Short acknowledgments
    if len(question.split()) <= 2 and any(g in question_lower for g in ["ok", "sure", "yes", "no"]):
        return False

    # For everything else, retrieve
    return True


def validate_course_id(course_id: str) -> bool:
    """
    Validate course ID format.

    Args:
        course_id: Course ID to validate

    Returns:
        Boolean indicating if course ID is valid
    """
    if not course_id or not isinstance(course_id, str):
        return False

    # Basic validation - adjust based on your course ID format
    if len(course_id) < 2 or len(course_id) > 20:
        return False

    return True


def truncate_context(context: str, max_tokens: int = 4000) -> str:
    """
    Truncate context to fit within token limit.

    Args:
        context: Context text to truncate
        max_tokens: Maximum number of tokens

    Returns:
        Truncated context
    """
    estimated_tokens = estimate_token_count(context)

    if estimated_tokens <= max_tokens:
        return context

    # Truncate to approximate character count
    max_chars = max_tokens * 4
    truncated = context[:max_chars]

    # Try to truncate at a sentence boundary
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:  # Only if we don't lose too much
        truncated = truncated[:last_period + 1]

    return truncated + "\n\n[Context truncated due to length...]"


def create_error_response(error_message: str) -> Dict[str, Any]:
    """
    Create standardized error response.

    Args:
        error_message: Error message to include

    Returns:
        Error response dictionary
    """
    return {
        "messages": [AIMessage(content=f"I apologize, but I encountered an error: {error_message}")],
        "retrieved_documents": [],
        "context": "",
        "needs_retrieval": False
    }


def get_answer_from_result(result: Dict[str, Any]) -> Optional[str]:
    """
    Extract answer text from agent result.

    Args:
        result: Result dictionary from agent execution

    Returns:
        Answer string or None if not found
    """
    messages = result.get("messages", [])
    if not messages:
        return None

    last_message = messages[-1]
    if hasattr(last_message, "content"):
        return last_message.content

    return str(last_message)
