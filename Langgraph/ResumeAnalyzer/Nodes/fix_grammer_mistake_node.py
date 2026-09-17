import os
import sys

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)

from typing import Any, Dict

from langsmith import traceable

from Chains.fix_grammer_mistake_chain import fix_grammer_grader
from Chains.grammer_validator_chain import GrammeticalErrors
from State.ResumeState import ResumeState


# To format the grammer mistakes
def format_grammer_mistakes_for_prompt(grammer_error_data: GrammeticalErrors) -> str:
    """Converts a GrammeticalErrors Pydantic object into a clean Markdown string."""
    formatted_list = []
    for idx, item in enumerate(grammer_error_data, 1):
        formatted_list.append(
            f"{idx}.- Mistake: {item.mistaken_word}\n"
            f"      - Reason: {item.reason}\n"
            f"      - Correct word: {item.correct_word}"
        )
    return "\n\n".join(formatted_list)


@traceable(name="Fix grammer mistake agent")
def fix_grammer_mistake(state: ResumeState) -> Dict[Any, str]:
    print("==== Fix grammer mistake agent ====")
    error_content = state["resume_content"].strip()
    grammer_mistakes = format_grammer_mistakes_for_prompt(state["grammer_mistakes"])

    response = fix_grammer_grader.invoke(
        {"error_content": error_content, "grammer_errors": grammer_mistakes}
    )

    return {
        "grammer_mistake_present": False,
        "resume_content": response.fixed_grammer_content,
    }
