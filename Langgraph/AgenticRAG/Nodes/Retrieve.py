from typing import Any, Dict

from Ingestion import retriever
from State.GraphState import GraphState


def retrieve(state: GraphState) -> Dict[Any, str]:
    print("===== Retrieving documents =====")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"question": question, "documents": documents}
