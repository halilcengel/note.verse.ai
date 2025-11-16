from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Literal, TypedDict, Annotated
from langgraph.graph import add_messages
from enum import Enum
import logging
from langgraph.config import get_stream_writer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from apps.school_web_site_agent.scrapper_agent import agent as announcement_agent
from apps.school_web_site_agent.yonetmelik_agent import agent as yonetmelik_agent
from core.llm import llm


class State(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    routing_reason: str
    current_agent: str


class AgentType(str, Enum):
    ANNOUNCEMENT = "announcement_agent"
    YONETMELIK = "yonetmelik_agent"


ORCHESTRATOR_SYSTEM_PROMPT = """
Sen bir üniversite asistan orkestratörüsün. Kullanıcının sorusunu analiz edip doğru uzman asistana yönlendiriyorsun.

Kullanabileceğin uzman asistanlar:

1. ANNOUNCEMENT_AGENT (Duyuru Asistanı):
   - Güncel duyurular, haberler, etkinlikler
   - Sınav tarihleri, kayıt dönemleri, son tarihler
   - Departman duyuruları ve güncellemeler
   - Web sitesindeki anlık bilgiler

2. YONETMELIK_AGENT (Yönetmelik Asistanı):
   - Okul yönetmelikleri ve yönergeler
   - Kalıcı kurallar ve prosedürler
   - Akademik politikalar
   - Resmi belgeler ve düzenlemeler

Görevin:
1. Kullanıcının sorusunu dikkatlice analiz et
2. Hangi asistanın daha uygun olduğuna karar ver
3. Sadece şu formatla yanıt ver: "ANNOUNCEMENT" veya "YONETMELIK"

Karar kriterleri:
- Zaman belirten ifadeler (son, bu hafta, bugün) → ANNOUNCEMENT
- "Duyuru", "haber", "etkinlik" kelimeleri → ANNOUNCEMENT  
- Kural, yönetmelik, prosedür soruları → YONETMELIK
- "Nasıl yapılır", "şartlar", "gereklilikler" → YONETMELIK
- Belirsiz durumlarda içerik tipine göre en mantıklısını seç

Sadece "ANNOUNCEMENT" veya "YONETMELIK" kelimelerinden birini döndür, başka bir şey yazma.
"""


def router_node(state: State, llm) -> State:
    """
    Router node that determines which agent should handle the query.
    Stores the decision in state for transparency and debugging.

    This node does NOT add messages to the state to prevent streaming output.
    """
    messages = state["messages"]

    # Find the last user message
    last_user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break

    if not last_user_message:
        logger.warning("No user message found, defaulting to announcement agent")
        state["next_agent"] = AgentType.ANNOUNCEMENT.value
        state["routing_reason"] = "No user message found"
        return state

    try:
        routing_messages = [
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=last_user_message)
        ]

        # Call LLM but don't add its response to the state messages
        response = llm.invoke(routing_messages)
        decision = response.content.strip().upper()

        if "YONETMELIK" in decision:
            agent = AgentType.YONETMELIK.value
        elif "ANNOUNCEMENT" in decision:
            agent = AgentType.ANNOUNCEMENT.value
        else:
            logger.warning(f"Unclear routing decision: {decision}, defaulting to announcement")
            agent = AgentType.ANNOUNCEMENT.value
            decision = f"UNCLEAR: {decision} -> DEFAULT: ANNOUNCEMENT"

        state["next_agent"] = agent
        state["routing_reason"] = decision
        state["current_agent"] = agent

        # Optional: Stream metadata about routing without streaming the actual response
        writer = get_stream_writer()
        writer({"agent": agent, "routing_reason": decision})

        logger.info(f"🔀 Routing to {agent.upper()} for query: '{last_user_message[:50]}...'")

    except Exception as e:
        logger.error(f"Error during routing: {e}", exc_info=True)
        state["next_agent"] = AgentType.ANNOUNCEMENT.value
        state["routing_reason"] = f"ERROR: {str(e)}"
        state["current_agent"] = AgentType.ANNOUNCEMENT.value
        logger.error(f"❌ Routing error, defaulting to announcement agent: {e}")

    # Return state without adding any AI messages
    return state


def route_to_agent(state: State) -> AgentType:
    """
    Simple conditional edge that reads routing decision from state.
    """
    next_agent = state.get("next_agent", AgentType.ANNOUNCEMENT.value)
    return AgentType(next_agent)


def create_orchestrator_graph(llm, announcement_agent, yonetmelik_agent):
    """
    Create the orchestrator graph that routes between specialized agents.
    """
    workflow = StateGraph(State)

    workflow.add_node("router", lambda state: router_node(state, llm))
    workflow.add_node(AgentType.ANNOUNCEMENT.value, announcement_agent)
    workflow.add_node(AgentType.YONETMELIK.value, yonetmelik_agent)

    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            AgentType.ANNOUNCEMENT: AgentType.ANNOUNCEMENT.value,
            AgentType.YONETMELIK: AgentType.YONETMELIK.value
        }
    )

    workflow.add_edge(AgentType.ANNOUNCEMENT.value, END)
    workflow.add_edge(AgentType.YONETMELIK.value, END)

    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    logger.info("Orchestrator graph compiled successfully")

    return graph


orchestrator = create_orchestrator_graph(llm, announcement_agent, yonetmelik_agent)