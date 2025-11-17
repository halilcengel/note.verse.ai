import logging
import os
import tempfile
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType, TextIndexParams, TokenizerType

from apps.school_web_site_agent.orchestrator import orchestrator
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


@app.post("/embed")
async def embed_document(
    collection_name: str = Form(...),
    file: UploadFile = File(...),
    title: str = Form(...),
    uploaded_by: str = Form(...),
    course_id: str = Form(...),
    document_id: str = Form(...)
):
    """
    Endpoint to upload a file, create embeddings, and store in Qdrant vector store.

    Parameters:
    - collection_name: Name of the collection to store embeddings
    - file: PDF file to process
    - title: Title of the document
    - uploaded_by: User who uploaded the document
    - course_id: ID of the associated course
    - document_id: Unique document identifier

    Returns:
    - Success message with document count
    """
    temp_path = None

    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()

        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        collections = qdrant_client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)

        if not collection_exists:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
            )
            logging.info(f"Created new collection: {collection_name}")
        else:
            logging.info(f"Using existing collection: {collection_name}")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            temp_path = tmp_file.name
            content = await file.read()
            tmp_file.write(content)

        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)

        for doc in splits:
            doc.metadata['uploaded_at'] = datetime.now().isoformat()
            doc.metadata['title'] = title
            doc.metadata['uploaded_by'] = uploaded_by
            doc.metadata['course_id'] = course_id
            doc.metadata['document-id'] = document_id

        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name,
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=120,
        )

        vector_store.add_documents(splits)

        return {
            "status": "success",
            "message": f"Successfully processed and embedded {len(splits)} document chunks",
            "collection_name": collection_name,
            "document_count": len(splits),
            "filename": file.filename,
            "title": title,
            "uploaded_by": uploaded_by,
            "course_id": course_id,
            "document_id": document_id,
            "uploaded_at": datetime.now().isoformat(),
            "collection_existed": collection_exists
        }

    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
