import logging
from typing import Any, Dict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..state import State
from src.core.llm import llm

logger = logging.getLogger(__name__)


RAG_SYSTEM_PROMPT = """You are a helpful course assistant. Your role is to answer student questions based on the course materials provided.

**Instructions:**
1. Answer questions using ONLY the information from the retrieved course materials provided below
2. If the materials don't contain enough information to answer the question, clearly state this
3. Provide specific references to the source materials when possible
4. Be concise but comprehensive
5. Use a friendly, educational tone
6. If you're uncertain, acknowledge it rather than making assumptions
7. Break down complex topics into understandable explanations

**Important:**
- Do NOT make up information not present in the course materials
- Do NOT use knowledge outside of the provided context
- If the question is outside the scope of the course materials, politely redirect to the course materials or instructor
"""


def generate_node(state: State) -> Dict[str, Any]:
    """
    Generation node for RAG pipeline.

    Uses LLM to generate answer based on retrieved context and conversation history.

    Args:
        state: Current agent state with retrieved documents and context

    Returns:
        Updated state with AI response
    """
    context = state.get("context", "")
    messages = state.get("messages", [])
    course_id = state.get("course_id", "unknown")

    if not messages:
        logger.warning("No messages found in state")
        return {"messages": []}

    last_user_message = messages[-1]
    user_query = last_user_message.content if hasattr(last_user_message, "content") else str(last_user_message)

    logger.info(f"Generating answer for: '{user_query[:100]}...'")
    logger.info(f"Context available: {len(context)} characters")

    # Build the prompt with context and conversation history
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("system", "Course ID: {course_id}\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chat_history = messages[:-1] if len(messages) > 1 else []

    try:
        chain = prompt_template | llm

        response = chain.invoke({
            "course_id": course_id,
            "context": context,
            "chat_history": chat_history,
            "question": user_query
        })

        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)

        logger.info(f"Generated answer: {len(answer)} characters")

        retrieved_docs = state.get("retrieved_documents", [])
        if retrieved_docs and len(retrieved_docs) > 0:
            citations = "\n\n---\n*Based on {} course material(s)*".format(len(retrieved_docs))
            answer += citations

        return {
            "messages": [AIMessage(content=answer)]
        }

    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)

        return {
            "messages": [AIMessage(content=f"I apologize, but I encountered an error while generating the response. Please try again.")]
        }
