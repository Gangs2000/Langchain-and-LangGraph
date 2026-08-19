from pydantic import BaseModel, Field
from typing import List

class Reflection(BaseModel):
    missing: str = Field(description= "Critique of what is missing")
    superfluous: str = Field(description= "Critique of what is superfluous")
    
class AnswerQuestion(BaseModel):
    """Answer the question"""
    answer : str = Field(description= "~250 word answer to your question")
    reflection: Reflection = Field(description= "Your reflection on the initial answer")
    search_queries: List[str] = Field(
        description= "1-3 Search queries to prepare answer using critique for the current answer"
    )

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question"""
    references : List[str] = Field(description= "List of references to revise the answer to your question")