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
    
    # LLM initialization
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    # Agent creation
    agent = create_agent(model= llm, tools=[search], response_format=AgentResponse.AgentResponse)
    # Invoking the agent with the user's query
    response = agent.invoke({"messages" : [HumanMessage(content=query)]})
    # Access structured response from the agent
    structured = response.get("structured_response", None)
    print(structured if structured is not None else response)
    
if __name__ == "__main__":
    main()