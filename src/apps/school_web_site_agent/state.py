from typing import Optional

from langchain.agents import AgentState

class State(AgentState):
    announcements: list = []
    related_announcement: Optional[dict] = None
    related_announcement_doc: Optional[list] = None
