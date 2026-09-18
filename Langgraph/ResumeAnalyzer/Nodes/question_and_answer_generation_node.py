import os
import sys

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)

from typing import Any, Dict

from langsmith import traceable

from Chains.question_and_answer_generation_chain import (
    ques_and_answer_generation_grader,
)
from State.ResumeState import ResumeState


@traceable(name="Techincal question generation agent")
def question_generation(state: ResumeState) -> Dict[Any, Any]:
    print("==== Generate technical questions agent ===")
    resume_content = (state.get("improved_content") or state["resume_content"]).strip()
    difficulty = (state.get("difficulty") or "Easy").strip()

    response = ques_and_answer_generation_grader.invoke(
        {"difficulty": difficulty, "resume_content": resume_content}
    )

    return {
        "resume_content": resume_content,
        "ats_resume_score": state.get("ats_resume_score", "Not available"),
        "improved_content": state.get("improved_content", "Not available"),
        "improvements": state.get("improvements", []),
        "difficulty": response.difficulty,
        "generated_questions": response.generated_questions,
    }
