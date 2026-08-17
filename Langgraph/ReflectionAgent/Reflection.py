from langchain_core.messages import HumanMessage, BaseMessage
from typing import TypedDict, Annotated
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
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
    if len(state["messages"]) > 6:
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

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="reflect.png")

if __name__ == "__main__":
    print("==== Reflection Agent demonstration ====")
    inputs = {
        "messages": [
            HumanMessage(
                content=
                """
                    Make this LinkedIn better:"
                    @LangChainAI newly Tool Calling feature is seriously underrated.
                    After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.
                    Made a video covering their newest blog post
                """
            )
        ]
    }
    response = app.invoke(inputs)
    print(response)