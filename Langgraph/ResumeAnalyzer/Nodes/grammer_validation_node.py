import os
import sys

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)

from typing import Any, Dict

from langsmith import traceable

from Chains.grammer_validator_chain import grammer_validation_grader
from State.ResumeState import ResumeState


@traceable(name="Grammer validation agent")
def grammer_validation(state: ResumeState) -> Dict[Any, Any]:
    print("==== Find grammatical error agent ===")
    resume_content = state["resume_content"].strip()

    response = grammer_validation_grader.invoke({"resume_content": resume_content})

    return {
        "grammer_mistake_present": response.grammer_mistake_present,
        "grammer_mistakes": response.grammer_mistakes,
    }
