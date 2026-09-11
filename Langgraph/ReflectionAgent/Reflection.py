from langchain_core.messages import HumanMessage, BaseMessage
from typing import TypedDict, Annotated
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from Chains import generation_chain, reflection_chain

from dotenv import load_dotenv

load_dotenv()

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    response = generation_chain.invoke({"messages": state["messages"]})
    return {"messages": response}

def reflection_node(state: MessageGraph):
    response = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content= response.content)]}

def should_continue(state: MessageGraph):
    # pause and surface the latest draft to a human for approve/reject
    decision = interrupt({"question": "Approve this draft?", "messages": state["messages"]})
    if decision == "approve":
        return END
    return REFLECT

flow = StateGraph(state_schema= MessageGraph)
flow.add_node(GENERATE, generation_node)
flow.add_node(REFLECT, reflection_node)
flow.set_entry_point(GENERATE)

flow.add_edge(REFLECT, GENERATE)

flow.add_conditional_edges(GENERATE, should_continue, path_map={
    END:END,
    REFLECT:REFLECT
})

# interrupt() requires a checkpointer to persist state across the pause/resume
checkpointer = MemorySaver()
app = flow.compile(checkpointer=checkpointer)
app.get_graph().draw_mermaid_png(output_file_path="reflect.png")

if __name__ == "__main__":
    print("==== Reflection Agent demonstration ====")
    inputs = {
        "messages": [
            HumanMessage(
                content=
                """
                    Make this LinkedIn better:"
                    About langchain human loop concept
                """
            )
        ]
    }
    config = {"configurable": {"thread_id": "reflection-agent-demo"}}

    response = app.invoke(inputs, config=config)
    while "__interrupt__" in response:
        payload = response["__interrupt__"][0].value
        print("\n--- Draft awaiting review ---")
        print(payload["messages"][-1].content)
        decision = input("Approve or reject this draft? [approve/reject]: ").strip().lower()
        response = app.invoke(Command(resume=decision), config=config)

    print("\n=== Final approved output ===")
    print(response["messages"][-1].content)