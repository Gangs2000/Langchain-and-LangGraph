from React import llm, tools
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

SYSYEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

tool_node = ToolNode(tools)

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
        Run the agent reasoning node
    """
    messages = [SystemMessage(content=SYSYEM_MESSAGE), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}