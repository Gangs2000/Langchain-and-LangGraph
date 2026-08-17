from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num: float) -> float:
    """
        Param num : a number to triple
        Returns : The triple of input number
    """
    return float(num) * 3
    
tools = [TavilySearch(max_results = 1), triple]

llm = ChatOpenAI(model= "gpt-4", temperature= 0).bind_tools(tools)