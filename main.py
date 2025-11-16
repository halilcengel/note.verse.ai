import logging

from fastapi import FastAPI
from langchain_core.messages import ToolMessage

from apps.school_web_site_agent.orchestrator import orchestrator

app = FastAPI()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from apps.school_web_site_agent.context import Context
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    message: str
    thread_id: str
    url: str = "https://eem.bakircay.edu.tr"
    school: str = "Izmir Bakircay Universitesi"
    department: str = "Elektrik Elektronik Mühendisliği"


@app.post("/chat")
async def query_agent(request: QueryRequest):
    async def event_generator():
        config = {"configurable": {"thread_id": request.thread_id}}

        context = Context(
            url=request.url,
            school=request.school,
            department=request.department
        )
        tool_response = False
        try:
            for chunk in orchestrator.stream(
                    {"messages": [{"role": "user", "content": request.message}]},
                    stream_mode=["messages","custom"],
                    config=config,
                    context=context,
                    subgraphs=True
            ):

                namespace, stream_type, data = chunk

                print("namespace:", namespace)
                print("data:", data)

                if isinstance(data, dict) and 'agent' in data:
                    agent_name = data['agent']
                    if agent_name:
                        yield f"data: {json.dumps({'type': 'agent_decision', 'agent_name': agent_name})}\n\n"
                        print(f"Routed to agent: {agent_name}")
                    else:
                        print("No agent routing information found")

                if isinstance(data, tuple) and len(data) == 2:
                    message, metadata = data

                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            if tool_call.get('name') and tool_call.get('id'):
                                current_tool_name = tool_call['name']
                                tool_args = tool_call.get('args', {})

                                yield f"data: {json.dumps({
                                    "type": "tool_start",
                                    "data": {
                                        "name": current_tool_name,
                                        "args": tool_args,
                                        "id": tool_call['id']
                                    }
                                })}\n\n"
                                print(f"Tool started: {current_tool_name}")
                                tool_response = True

                    if hasattr(message, '__class__') and message.__class__.__name__ == 'ToolMessage':
                        tool_name = getattr(message, 'name', current_tool_name)
                        tool_content = getattr(message, 'content', '')
                        try:
                            tool_result = json.loads(tool_content)
                        except:
                            tool_result = tool_content

                        yield f"data: {json.dumps({
                            "type": "tool_response",
                            "data": {
                                "name": tool_name,
                                "result": tool_result
                            }
                        })}\n\n"
                        print(f"Tool response from: {tool_name}")
                        tool_response = False

                    if message.__class__.__name__ == "AIMessageChunk" and metadata.get("langgraph_node") != "router" and tool_response is False:
                        yield f"data: {json.dumps({'type': 'message', 'content': message.content})}\n\n"

        except Exception as e:
            error_data = {
                "type": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"

        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
