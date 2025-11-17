from langchain.tools import tool, ToolRuntime
from apps.school_web_site_agent.context import Context
from typing import Optional
from core.vector_store import store

@tool
def query_school_regulations(
        runtime: ToolRuntime[Context],
        query: str,
        k: Optional[int] = 5
):
    """

    """

    k = max(1, min(k, 20))

    try:

        print(f"Searching for: '{query}' (returning top {k} results)")
        results = store.similarity_search_with_score(query, k=k)

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": float(score)
            })

        runtime.state["regulation_search_results"] = formatted_results
        runtime.state["last_regulation_query"] = query

        response = {
            "query": query,
            "num_results": len(formatted_results),
            "results": formatted_results
        }

        print(f"✓ Found {len(formatted_results)} relevant document chunks")

        return response

    except Exception as e:
        error_msg = f"Error querying vector store: {str(e)}"
        print(f"✗ {error_msg}")
        return {
            "error": error_msg,
            "query": query,
            "num_results": 0,
            "results": []
        }