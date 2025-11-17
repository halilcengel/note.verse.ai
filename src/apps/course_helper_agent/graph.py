from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

import logging
from state import State

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.llm import llm

workflow = StateGraph(State)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)