from apps.school_web_site_agent.scrapper_agent import agent
from apps.school_web_site_agent.context import Context

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Guncel duyurular neler"}]},
    config=config,
    context=Context(url="https://eem.bakircay.edu.tr", school="Izmir Bakircay Universitesi", department="Elektrik Elektronik Mühendisliği")
)

print(response)

