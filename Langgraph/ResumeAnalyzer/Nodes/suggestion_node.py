import os
import sys

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)

from typing import Any, Dict

from langsmith import traceable

from Chains.suggestions_chain import (Suggestions, suggestion_applier_grader,
                                      suggestion_extractor_grader)
from State.ResumeState import ResumeState


# To format the suggestions
def format_suggestions_for_prompt(suggestions_data: Suggestions) -> str:
    """Converts a Suggestions Pydantic object into a clean Markdown string."""
    formatted_list = []
    for idx, item in enumerate(suggestions_data.suggestions, 1):
        formatted_list.append(
            f"{idx}. [{item.category.upper()}]\n"
            f"   - Issue: {item.issue}\n"
            f"   - Recommendation: {item.recommendation}"
        )
    return "\n\n".join(formatted_list)


@traceable(name="Suggestion extractor and applier agent")
def suggestion_extraction_and_apply(state: ResumeState) -> Dict[Any, Any]:
    print("=== Extracting suggestion ===")
    content_to_validate = (
        state.get("improved_content") or state["resume_content"]
    ).strip()
    ats_resume_score = state["ats_resume_score"]

    # Extracting suggestions
    extracted_suggestion = suggestion_extractor_grader.invoke(
        {"resume_content": content_to_validate}
    )
    suggestions = format_suggestions_for_prompt(extracted_suggestion)
    print("=== Suggestions extracted ===")

    # Applying suggestions to resume
    print("=== Applying suggestion ===")
    improved_content = suggestion_applier_grader.invoke(
        {
            "resume_content": content_to_validate,
            "ats_resume_score": ats_resume_score,
            "suggestions": suggestions,
        }
    )
    print("=== Suggestion applied ===")

    return {
        "resume_content": content_to_validate,
        "ats_resume_score": ats_resume_score,
        "suggestions": suggestions,
        "improved_content": improved_content,
    }
