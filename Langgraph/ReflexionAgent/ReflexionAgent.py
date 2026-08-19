from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from Chains import reviser_responder, first_responder
from Nodes import execute_tools
from langgraph.graph import START, END, StateGraph, MessagesState

load_dotenv()

MAX_ITERATIONS = 2

DRAFT = "draft"
REVISE = "revise"
EXECUTE_TOOLS = "execute_tools"

def draft_node(state: MessagesState):
    """Draft the initial response"""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def revise_node(state: MessagesState):
    """Revise the answer based on the tool results"""
    response = reviser_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def event_loop(state: MessagesState):
    """Determine whether to continue executing tools or to finish based on iterations count"""
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    if count_tool_visits > MAX_ITERATIONS:
        return END
    return EXECUTE_TOOLS

flow = StateGraph(MessagesState)
flow.add_node(DRAFT, draft_node)
flow.add_node(EXECUTE_TOOLS, execute_tools)
flow.add_node(REVISE, revise_node)
flow.add_edge(START, DRAFT)
flow.add_edge(DRAFT, EXECUTE_TOOLS)
flow.add_edge(EXECUTE_TOOLS, REVISE)
flow.add_conditional_edges(REVISE, event_loop, path_map= {EXECUTE_TOOLS: EXECUTE_TOOLS, END: END})

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path= "reflexion.png")

human_message = HumanMessage(
    content= """
        Is DSA problem dead in MAANG interviews,
        List down top technologies to be learnt
    """)

response = app.invoke(input= {"messages": [human_message]})

# Extract last message
last_message = response["messages"][-1]

if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])
print(response)