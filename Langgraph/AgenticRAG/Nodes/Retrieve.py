from State.GraphState import GraphState
from Ingestion import retriever
from typing import Any, Dict

def retrieve(state: GraphState) -> Dict[Any, str]:
    print("===== Retrieving documents =====")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"question": question, "documents": documents}
    