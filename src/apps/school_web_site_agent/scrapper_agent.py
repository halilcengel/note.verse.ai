from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from core.llm import llm
from apps.school_web_site_agent.tools import scrape_announcements
from apps.school_web_site_agent.context import Context

SYSTEM_PROMPT = """
Sen Üniversite Duyurularına erişebilen bir asistansın. Bunun için scrape_announcements tool'unu kullanabilirsin.
"""

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[scrape_announcements],
    context_schema=Context,
    checkpointer=checkpointer
)