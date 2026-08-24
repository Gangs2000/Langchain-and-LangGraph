from Chains.GenerationChain import generation_chain
from typing import Any, Dict
from State.GraphState import GraphState


def generate_answer(state: GraphState) -> Dict[Any, str]:
    print("==== Generation Answer ====")
    
    question = state["question"]
    documents = state["documents"]
    
    generation = generation_chain.invoke({"question": question, "context": documents})
    
    return {"question": question, "documents": documents, "generation": generation}
    