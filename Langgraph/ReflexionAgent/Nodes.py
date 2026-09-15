from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from Schema import AnswerQuestion, ReviseAnswer

load_dotenv()

tavily_tool = TavilySearch(max_results=5)


def search_queries(search_queries: list[str], **kwargs):
    """Run the generated queries"""
    return tavily_tool.batch([{"query": query} for query in search_queries])


execute_tools = ToolNode(
    [
        StructuredTool.from_function(
            search_queries, name=AnswerQuestion.__name__, args_schema=AnswerQuestion
        ),
        StructuredTool.from_function(
            search_queries, name=ReviseAnswer.__name__, args_schema=ReviseAnswer
        ),
    ]
)
