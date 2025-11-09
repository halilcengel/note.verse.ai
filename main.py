from fastapi import FastAPI

app = FastAPI()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from apps.school_web_site_agent.scrapper_agent import agent
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

        try:
            for token, metadata in agent.stream(
                    {"messages": [{"role": "user", "content": request.message}]},
                    stream_mode="messages",
                    config=config,
                    context=context
            ):
                # Determine the content type
                content_blocks = token.content_blocks

                # Check if this is a tool call
                is_tool_call = any(
                    block.get('type') in ['tool_call_chunk', 'tool_call']
                    for block in content_blocks
                )

                # Check if this is a tool result
                is_tool_result = metadata['langgraph_node'] == 'tools'

                # Determine message type
                if is_tool_result:
                    message_type = "tool_result"
                elif is_tool_call:
                    message_type = "tool_call"
                else:
                    message_type = "ai"

                event_data = {
                    "node": metadata['langgraph_node'],
                    "type": message_type,
                    "content": content_blocks
                }

                yield f"data: {json.dumps(event_data)}\n\n"

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
