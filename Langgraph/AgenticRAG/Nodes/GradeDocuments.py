from Chains.RetrievalGrader import retrieval_grader
from typing import Any, Dict
from State.GraphState import GraphState

def grade_documents(state: GraphState) -> Dict[Any, str]:
    """
        Tool to check if retrieved documents are relevent to the question.
        If any document is not relevent, we will make web_search
        
        Args:
        state (dict): The current graph state

        Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    
    web_search = False
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "documents": documents})
        grade = score.binary_score
        
        if grade.lower() == "yes":
            print("==== DOC IS RELEVANT ====")
            filtered_docs.append(doc)
        else:
            print("==== GRADE: DOCUMENT NOT RELEVANT ====")
            web_search = True
            continue
    return {"question": question, "documents": filtered_docs, "web_search": web_search}
            