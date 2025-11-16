from typing import Optional

from langchain.agents import AgentState

class State(AgentState):
    announcements: list = []
    related_announcement: Optional[dict] = None
    related_announcement_doc: Optional[list] = None
    regulation_search_results: Optional[dict] = None
    last_regulation_query: Optional[str] = None
