from typing import Any, Dict
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from State.GraphState import GraphState
from dotenv import load_dotenv

load_dotenv()

web_search_tool = TavilySearch(max_results = 3)

def web_search(state: GraphState) -> Dict[Any, str]:
    print("==== Web Search using Tavily ====")
    question = state["question"]
    documents = state[documents] if "documents" in state else None
    
    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_results = "\n".join(tavily_result["content"] for tavily_result in tavily_results)
    
    web_results = Document(page_content= joined_results)
    
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
        
    return {"question": question, "documents": documents}