from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

from apps.school_web_site_agent.state import State
from core.llm import llm
from apps.school_web_site_agent.tools import scrape_announcements, scrape_announcement, get_document_from_url
from apps.school_web_site_agent.context import Context

SYSTEM_PROMPT = """
Sen Üniversite Duyurularına erişebilen bir asistansın. Kullanıcılara duyurular hakkında bilgi vermek için tasarlandın.

Kullanabileceğin araçlar:
1. scrape_announcements: Duyuru listesini çeker. Zaman filtreleme yapabilirsin (1d, 1w, 1m, 3m, 6m, 1y, all)
2. scrape_announcement: Tek bir duyurunun detaylarını çeker (başlık, tarih, içerik, linkler)
3. get_document_from_url: PDF dokümanlarını indirir ve içeriğini çıkarır

Görevlerin:
- Kullanıcı duyurular hakkında soru sorduğunda önce scrape_announcements ile listeyi çek
- İlgili duyuruları bul ve gerekirse scrape_announcement ile detayları al
- PDF linkleri varsa ve kullanıcı içeriği istiyorsa get_document_from_url kullan
- Türkçe ve anlaşılır şekilde yanıt ver
- Tarihleri ve detayları doğru aktar

Örnekler:
- "Son 1 haftanın duyurularını göster" -> scrape_announcements(time_range="1w")
- "Sınav takvimi var mı?" -> scrape_announcements ile ara, ilgili duyuruyu bul, detaylarını al
"""

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[scrape_announcements, scrape_announcement, get_document_from_url],
    state_schema=State,
    context_schema=Context,
    checkpointer=checkpointer
)