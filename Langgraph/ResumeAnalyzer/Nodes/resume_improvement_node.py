import os
import sys

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)

from typing import Any, Dict

from langsmith import traceable

from Chains.resume_improvement_chain import (Improvements,
                                             improvements_applier_grader,
                                             improvements_extractor_grader)
from State.ResumeState import ResumeState


# To format the suggestions
def format_improvements_for_prompt(improvements_data: Improvements) -> str:
    """Converts a Improvements Pydantic object into a clean Markdown string."""
    formatted_list = []
    for idx, item in enumerate(improvements_data.improvements, 1):
        formatted_list.append(
            f"{idx}. [{item.category.upper()}]\n"
            f"   - Issue: {item.issue}\n"
            f"   - Recommendation: {item.recommendation}"
        )
    return "\n\n".join(formatted_list)


@traceable(name="Improvement extractor and applier agent")
def suggestion_extraction_and_apply(state: ResumeState) -> Dict[Any, Any]:
    print("=== Extracting improvement suggestions ===")
    content_to_validate = (
        state.get("improved_content") or state["resume_content"]
    ).strip()
    ats_resume_score = state["ats_resume_score"]

    # Extracting suggestions
    extracted_suggestion = improvements_extractor_grader.invoke(
        {"resume_content": content_to_validate}
    )
    improvements = format_improvements_for_prompt(extracted_suggestion)
    print("=== Improvement suggestions extracted ===")

    # Applying suggestions to resume
    print("=== Applying improvement suggestions ===")
    improved_content = improvements_applier_grader.invoke(
        {
            "resume_content": content_to_validate,
            "ats_resume_score": ats_resume_score,
            "improvements": improvements,
        }
    )
    print("=== Improvement suggestions applied ===")

    return {
        "resume_content": content_to_validate,
        "ats_resume_score": ats_resume_score,
        "improvements": improvements,
        "improved_content": improved_content,
    }
