from langchain_core.tools import tool
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv
import AgentResponse

load_dotenv()

class SearchAgent:
    def __init__(self, name):
        self.name = name

@tool
def search(query: str) -> str:
    """
    Search for information about the given query.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    print("Searching for information about:", query)
    return TavilyClient().search(query=query)

def main():
    print("Search Agent using LangChain!!")
    query = input("Ask a question : ")

    llm = ChatOpenAI(temperature=0, model="gpt-4", streaming=True)
    agent = create_agent(model=llm, tools=[search], response_format=AgentResponse.AgentResponse)

    for token, metadata in agent.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="messages",
    ):
        if token.content:
            print(token.content, end="", flush=True)
    
if __name__ == "__main__":
    main()