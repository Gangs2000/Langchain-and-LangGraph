from typing import Annotated, TypedDict

from Chains import generation_chain, reflection_chain
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from langsmith import traceable

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


@traceable(name="generation_agent")
def generation_node(state: MessageGraph):
    response = generation_chain.invoke({"messages": state["messages"]})
    return {"messages": response}


@traceable(name="reflection_agent")
def reflection_node(state: MessageGraph):
    # Run the chain
    initial_post = state["messages"][0].content
    generated_post = state["messages"][-1].content
    critique = reflection_chain.invoke(
        {"initial_post": initial_post, "generated_post": generated_post}
    )
    return {"messages": [HumanMessage(content=critique.content)]}


def should_continue(state: MessageGraph):
    decision = interrupt(
        {"question": "Approve this draft?", "messages": state["messages"]}
    )
    if decision == "approve":
        return END
    return REFLECT


flow = StateGraph(state_schema=MessageGraph)
flow.add_node(GENERATE, generation_node)
flow.add_node(REFLECT, reflection_node)
flow.set_entry_point(GENERATE)

flow.add_edge(REFLECT, GENERATE)

flow.add_conditional_edges(
    GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT}
)

checkPointer = MemorySaver()

app = flow.compile(checkpointer=checkPointer)
app.get_graph().draw_mermaid_png(output_file_path="reflect.png")

if __name__ == "__main__":
    print("==== Reflection Agent for resume analyzer ====")
    inputs = {"messages": [HumanMessage(content="""
                    Make this resume portion better:"

                    Achievements
                    * Won 2nd prize in chess in Intramural Games organised by Sourashtra Co-Education Higher Secondary School, Feb,2014.
                    * Participated in State level Technical Symposium, IFEST-2K18, Software Debugging, Lady Doak
                      College, Feb,2018
                    * Won 2nd in State level Technical Symposium, CS INNOWIZ-2018, Tech Hunt, Alagappa
                      University, Oct,2018
                    * Recognized as Best employee of the year 2025-2026 in the organization for outstanding performance and dedication to work.

                """)]}
    config = {"configurable": {"thread_id": "analyze_resume_achievements_section"}}
    response = app.invoke(inputs, config=config)
    while "__interrupt__" in response:
        payload = response["__interrupt__"][0].value
        print("\n --- Drafting content for review ---")
        print(response["messages"][-1].content)
        decision = (
            input("Approve or reject this draft? [approve/reject] : ").strip().lower()
        )
        response = app.invoke(Command(resume=decision), config=config)
    print("\n === Final approved output ===")
    print(response["messages"][-1].content)
