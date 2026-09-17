from typing import List, TypedDict


class ResumeState(TypedDict):
    """
    Represents the state of the graph

    Attributes:

    resume_content: Original resume content shared by user
    grammer_mistake_present: hold boolean flag whether resume has any grammatical error
    grammer_mistakes: this list hold all grammer errors if found during the analyze
    ats_resume_score: ATS resume score given by agent
    suggestions: Suggestions given by LLM to improve resume quality
    improved_content: In case ATS score is below 80%, this field holds improved content
    """

    resume_content: str
    grammer_mistake_present: bool
    grammer_mistakes: List[str]
    ats_resume_score: float
    suggestions: List[str]
    improved_content: str
