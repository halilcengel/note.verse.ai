from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

import logging
from apps.course_helper_agent.state import State
from apps.course_helper_agent.nodes.retrieval import retrieve_node
from apps.course_helper_agent.nodes.generate import generate_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_course_helper_graph():
    """
    Create the course helper RAG agent graph.

    The graph follows this flow:
    1. START -> retrieve_node: Retrieve relevant course documents
    2. retrieve_node -> generate_node: Generate answer using retrieved context
    3. generate_node -> END: Return answer to user

    Returns:
        Compiled LangGraph with checkpointing enabled
    """
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Define edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    logger.info("Course helper graph created successfully")

    return graph


graph = create_course_helper_graph()