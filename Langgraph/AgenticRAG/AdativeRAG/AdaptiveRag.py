from Nodes import Retrieve, GradeDocuments, WebSearch, Generation
from Chains import HallucinationGrader, AnswerGrader, RouterGrader
from langgraph.graph import END, StateGraph
from State.GraphState import GraphState
from Chains.RouterGrader import RouteQuery
from dotenv import load_dotenv

load_dotenv()

RETRIEVE = "retrieve"
GRADE_DOCS = "grade_docs"
WEB_SEARCH = "web_search"
GENERATE = "generate"

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    question = state["question"]
    documents = state["documents"]
    generated_answer = state["generation"]
    print("==== Check hallucination ====")
    if generated_answer is not None:
        score = HallucinationGrader.invoke({"documents": documents, "generation": generated_answer})
        if hallucination_grade := score.binary_score:
            print("==== DECISION: GENERATION IS GROUNDED IN DOCUMENTS ====")
            print("==== GRADE GENERATION vs QUESTION ====")
            score = AnswerGrader.invoke({"question": question, "generation": generated_answer})
            if answer_grade := score.binary_score:
                print("==== DECISION: GENERATION ADDRESSES QUESTION ====")
                return "useful"
            else:
                print("==== DECISION: GENERATION DOES NOT ADDRESS QUESTION ====")
                return "not useful"
        else:
            print("==== DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS ====")
            return "not supported"

def decide_to_generate(state: GraphState) -> str:
    print("==== ASSESS graded docs ====")
    web_search = state["web_search"]
    if web_search is True:
        print ("==== NOT ALL DOCS ARE RELEVANT, INCLUDE WEB SEARCH DATA ====")
        return WEB_SEARCH
    else:
        print ("==== ALL DOCS ARE RELEVANT TO THE QUESTION ====")
        return GENERATE
    
def route_question(state: GraphState) -> str:
    question = state["question"]
    source: RouteQuery = RouterGrader.invoke({"question": question})
    if source.dataSource == "vectore_db":
        print("=== ROUTER TO RAG ====")
        return RETRIEVE
    elif source.dataSource == "web_search":
        print("=== ROUTER TO WEB SEARCH ====")
        return WEB_SEARCH

flow = StateGraph(GraphState)
flow.add_node(RETRIEVE, Retrieve)
flow.add_node(GRADE_DOCS, GradeDocuments)
flow.add_node(WEB_SEARCH, WebSearch)
flow.add_node(GENERATE, Generation)

flow.set_conditional_entry_point(route_question, {
    WEB_SEARCH: WEB_SEARCH,
    RETRIEVE: RETRIEVE
})

flow.add_edge(RETRIEVE, GRADE_DOCS)
flow.add_conditional_edges(GRADE_DOCS, decide_to_generate, {
    WEB_SEARCH: WEB_SEARCH,
    GENERATE: GENERATE
})

flow.add_edge(WEB_SEARCH, GENERATE)

flow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question, {
    "not_supported": GENERATE,
    "useful": END,
    "not_useful": WEB_SEARCH
})

flow.add_edge(GENERATE, END)

app = flow.compile()

app.get_graph().draw_mermaid_png(output_file_path= "SelfRAG.png")

if __name__ == "__main__":
    query = "About agent memory?"
    print(app.invoke(input = {"question": query}))

