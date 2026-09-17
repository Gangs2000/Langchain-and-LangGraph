import os
import sys

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)

from typing import Any, Dict

from langsmith import traceable

from Chains.ats_resume_validator_chain import ats_resume_validatoion_grader
from State.ResumeState import ResumeState


@traceable(name="ATS resume validator agent")
def ats_resume_score_validator(state: ResumeState) -> Dict[Any, Any]:
    print("=== ATS Resume validator ===")
    resume_content = state["resume_content"].strip()
    content_to_validate = (
        state.get("improved_content") or state["resume_content"]
    ).strip()
    response = ats_resume_validatoion_grader.invoke(
        {"resume_content": content_to_validate}
    )

    return {
        "resume_content": resume_content,
        "ats_resume_score": response.ats_score,
        "improved_content": state.get("improved_content") or None,
    }
