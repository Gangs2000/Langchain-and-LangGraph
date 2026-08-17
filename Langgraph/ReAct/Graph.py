from Nodes import run_agent_reasoning, tool_node
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import HumanMessage
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

@traceable(name= "Langgraph reasoning agent")
def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue , {
    ACT:ACT, 
    END:END
})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Agent reasoning using LangGraph!!")
    query = input("Ask a question : ")
    humanMessage = [HumanMessage(content=query)]
    response = app.invoke({"messages": humanMessage})
    print(response["messages"][LAST].content)
    