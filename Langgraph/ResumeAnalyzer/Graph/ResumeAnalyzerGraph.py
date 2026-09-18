import os
import sys

RESUME_ANALYZER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RESUME_ANALYZER not in sys.path:
    sys.path.insert(0, RESUME_ANALYZER)

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from Nodes.ats_resume_validation_node import ats_resume_score_validator
from Nodes.fix_grammer_mistake_node import fix_grammer_mistake
from Nodes.grammer_validation_node import grammer_validation
from Nodes.question_and_answer_generation_node import question_generation
from Nodes.resume_improvement_node import suggestion_extraction_and_apply
from State.ResumeState import ResumeState

load_dotenv()

DECISION_ROUTING_NODE = "decision_routing_node"
GRAMMER_VALIDATOR_NODE = "grammer_validation"
FIX_GRAMMER_NODE = "fix_grammer_mistake"
ATS_RESUME_VALIDATOR_NODE = "ats_resume_validation"
ATS_METER_CHECK_NODE = "ats_resume_meter_check"
IMPROVEMENT_NODE = "resume_improvement"
TECHNICAL_QUESTION_GENERATION_NODE = "question_and_answer_generation"


def decision_routing_node(state: ResumeState):
    resume_dict = interrupt(
        {
            "question": "Enter 1 to generate only technical question, Enter 2 for full resume analyze flow"
        }
    )
    return {
        "user_choice": resume_dict["user_choice"],
        "difficulty": resume_dict.get("difficulty", "Easy"),
    }


def route_after_decision(state: ResumeState):
    if state.get("user_choice") == 1:
        return "generate_technical_question"
    return "analyze_full_resume"


def grammer_validation_check(state: ResumeState):
    grammer_mistake_present = state["grammer_mistake_present"]
    if grammer_mistake_present is True:
        return "grammer_error"
    return "no_grammer_error"


def ats_resume_meter_check(state: ResumeState):
    ats_score = state["ats_resume_score"]
    if ats_score >= 80.0:
        resume_dict = interrupt(
            {
                "question": "Resume ATS score is above 80%, Enter 1 to generate technical questions, Enter 2 to finish the flow"
            }
        )
        if resume_dict.get("user_choice") == 1:
            return {
                "difficulty": resume_dict.get("difficulty", "Easy"),
                "user_choice": 1,
            }
        return {"user_choice": 2}
    return {"user_choice": None}


def route_after_ats_meter_check(state: ResumeState):
    ats_score = state["ats_resume_score"]
    if ats_score < 80.0:
        return "score < 80%"
    if state.get("user_choice") == 1:
        return "generate_technical_question"
    return "score >= 80%"


flow = StateGraph(state_schema=ResumeState)

# Adding nodes in the flow
flow.add_node(DECISION_ROUTING_NODE, decision_routing_node)
flow.add_node(GRAMMER_VALIDATOR_NODE, grammer_validation)
flow.add_node(FIX_GRAMMER_NODE, fix_grammer_mistake)
flow.add_node(ATS_RESUME_VALIDATOR_NODE, ats_resume_score_validator)
flow.add_node(ATS_METER_CHECK_NODE, ats_resume_meter_check)
flow.add_node(IMPROVEMENT_NODE, suggestion_extraction_and_apply)
flow.add_node(TECHNICAL_QUESTION_GENERATION_NODE, question_generation)

flow.add_edge(START, DECISION_ROUTING_NODE)

flow.add_conditional_edges(
    DECISION_ROUTING_NODE,
    route_after_decision,
    path_map={
        "generate_technical_question": TECHNICAL_QUESTION_GENERATION_NODE,
        "analyze_full_resume": GRAMMER_VALIDATOR_NODE,
    },
)

# Edge from fix_grammer to grammer_validation
flow.add_edge(FIX_GRAMMER_NODE, GRAMMER_VALIDATOR_NODE)
# Conditional edge check on grammer_validation_check, if true go to fix_grammer else ats_resume_validator
flow.add_conditional_edges(
    GRAMMER_VALIDATOR_NODE,
    grammer_validation_check,
    path_map={
        "grammer_error": FIX_GRAMMER_NODE,
        "no_grammer_error": ATS_RESUME_VALIDATOR_NODE,
    },
)
# Edge from improvement_node to ats_resume_validator
flow.add_edge(IMPROVEMENT_NODE, ATS_RESUME_VALIDATOR_NODE)
flow.add_edge(ATS_RESUME_VALIDATOR_NODE, ATS_METER_CHECK_NODE)
# Conditional edge check on ats_resume_meter_check, if above 80.0% go to END, else improvement_node
# If wants to generate technical question, go to technical_question_generation_node
flow.add_conditional_edges(
    ATS_METER_CHECK_NODE,
    route_after_ats_meter_check,
    path_map={
        "generate_technical_question": TECHNICAL_QUESTION_GENERATION_NODE,
        "score < 80%": IMPROVEMENT_NODE,
        "score >= 80%": END,
    },
)

checkPointer = InMemorySaver()

app = flow.compile(checkpointer=checkPointer)

app.get_graph().draw_mermaid_png(output_file_path="resume_analyzer_graph.png")

if __name__ == "__main__":
    print("=== Begin resume analyzer ===")
    resume_content = """
        Paste your resume content here for analysis. 
        The resume analyzer will evaluate the content, check for grammar mistakes, assess the ATS score, provide improvement suggestions, 
        and generate technical interview questions based on the resume.    
    """
    config = {"configurable": {"thread_id": "ollama_resume_analyzer"}}
    response = app.invoke(input={"resume_content": resume_content}, config=config)

    while "__interrupt__" in response:
        payload = response["__interrupt__"][0].value
        resume_dict = {}
        print("\n --- Please choose one of below option ---")
        print(payload.get("question", "Make a choice : "))
        choice = int(input("Enter your choice either 1 or 2 : ").strip().lower())
        resume_dict["user_choice"] = choice
        if choice == 1:
            difficulty = (
                input("Enter the difficulty level (Easy, Moderate, Hard) : ")
                .strip()
                .capitalize()
            )
            resume_dict["difficulty"] = difficulty
        response = app.invoke(Command(resume=resume_dict), config=config)

    # print("\n *** Original resume content ***")
    # print("+" * 50)
    # print(response.get("resume_content"))
    print("+" * 50)
    print("\n *** Resume ATS Score ***")
    print(response.get("ats_resume_score", "Not available"))
    print("+" * 50)
    # print("\n *** Improved resume content ***")
    # print(response.get("improved_content", "Not available"))
    # print("+" * 50)

    generated_questions = response.get("generated_questions") or []
    difficulty = response.get("difficulty", "Not specified")

    if generated_questions:
        print(f"\n=== Technical Interview Questions (Difficulty: {difficulty}) ===")
        print("=" * 60)

        for idx, item in enumerate(generated_questions, 1):
            q_text = getattr(item, "question", None) or item.get("question")
            options = getattr(item, "options", None) or item.get("options", [])
            answer = getattr(item, "answer", None) or item.get("answer")

            print(f"\nQuestion {idx}: {q_text}")
            print("-" * 40)
            print("Options:")

            option_labels = ["A", "B", "C", "D"]
            for label, opt in zip(option_labels, options):
                print(f"  [{label}] {opt}")

            print(f"\n✓ Correct Answer: {answer}")
            print("=" * 60)
    else:
        print("No questions were generated.")

    print("=== End resume analyzer ===")
