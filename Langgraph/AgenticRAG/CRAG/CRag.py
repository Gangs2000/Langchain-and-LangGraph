import os
import sys

C_RAG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if C_RAG_ROOT not in sys.path:
    sys.path.insert(0, C_RAG_ROOT)

from Nodes.Retrieve import retrieve
from Nodes.GradeDocuments import grade_documents
from Nodes.WebSearch import web_search
from Nodes.Generation import generate_answer
from langgraph.graph import END, StateGraph
from State.GraphState import GraphState
from dotenv import load_dotenv

load_dotenv()

RETRIEVE = "retrieve"
GRADE_DOCS = "grade_docs"
WEB_SEARCH = "web_search"
GENERATE = "generate"

def decide_to_generate(state: GraphState) -> str:
    print("==== ASSESS graded docs ====")
    web_search = state["web_search"]
    if web_search is True:
        print ("==== NOT ALL DOCS ARE RELEVANT, INCLUDE WEB SEARCH DATA ====")
        return WEB_SEARCH
    else:
        print ("==== ALL DOCS ARE RELEVANT TO THE QUESTION ====")
        return GENERATE

flow = StateGraph(GraphState)
flow.add_node(RETRIEVE, retrieve)
flow.add_node(GRADE_DOCS, grade_documents)
flow.add_node(WEB_SEARCH, web_search)
flow.add_node(GENERATE, generate_answer)

flow.set_entry_point(RETRIEVE)

flow.add_edge(RETRIEVE, GRADE_DOCS)
flow.add_conditional_edges(GRADE_DOCS, decide_to_generate, {
    WEB_SEARCH: WEB_SEARCH,
    GENERATE: GENERATE
})

flow.add_edge(WEB_SEARCH, GENERATE)
flow.add_edge(GENERATE, END)

app = flow.compile()

app.get_graph().draw_mermaid_png(output_file_path= "CRAG.png")

if __name__ == "__main__":
    query = "agent memory"
    print(app.invoke(input = {"question": query}))

