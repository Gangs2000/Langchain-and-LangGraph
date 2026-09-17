import os
import sys

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)

from typing import List, TypedDict

from Chains.grammer_validator_chain import GrammeticalErrors
from Chains.resume_improvement_chain import Improvements


class ResumeState(TypedDict):
    """
    Represents the state of the graph

    Attributes:

    resume_content: Original resume content shared by user
    grammer_mistake_present: hold boolean flag whether resume has any grammatical error
    grammer_mistakes: this list hold all grammer errors if found during the analyze
    ats_resume_score: ATS resume score given by agent
    improvements: Improvement suggestions given by LLM to improve resume quality
    improved_content: In case ATS score is below 80%, this field holds improved content
    """

    resume_content: str
    grammer_mistake_present: bool
    grammer_mistakes: List[GrammeticalErrors]
    ats_resume_score: float
    improvements: List[Improvements]
    improved_content: str
