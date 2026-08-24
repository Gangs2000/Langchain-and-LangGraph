from Nodes import Retrieve, GradeDocuments, WebSearch, Generation
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
flow.add_node(RETRIEVE, Retrieve)
flow.add_node(GRADE_DOCS, GradeDocuments)
flow.add_node(WEB_SEARCH, WebSearch)
flow.add_node(GENERATE, Generation)

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
    query = "About agent memory?"
    print(app.invoke(input = {"question": query}))

