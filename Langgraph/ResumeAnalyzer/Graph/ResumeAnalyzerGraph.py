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
        Career Objective :

        I is seekings a dynamicly environment where I can leverages my skill in write, test, and debugs code to driving website and AI applications performings and reliable. 
        Exciting to contributing for innovative project and took on new challenge in the Software and AI engineerings domains.

        --------------------------------------------------

        Work Experience :

        IT ANALYST
        Tata Consultancy Services
        Aug 2020 - Apr 2022

        -> Spearheaded Java Support Development, delivering features based on customer requests and maintaining codebase quality through regular clean-up efforts.
        -> Leveraged expertise in Spring Framework to develop robust solutions, solidifying skills in Java and Spring environments.

        SOFTWARE ENGINEER
        Globallogic Services
        May 2022 - Dec 2023

        -> Contributed as a Software Engineer on a networking product, utilizing Core Java, Microservices, PostgreSQL, MongoDB, and Docker management skills.
        -> Demonstrated expertise in designing and developing scalable solutions, ensuring seamless integration across multiple components.

        SOFTWARE DEVELOPER ENGINEER
        Centrico Software India Pvt Ltd
        Dec 2023 - Present

        -> Currently driving innovation as a Software Developer in the banking domain, focusing on writing clean, efficient, and reusable code using Microservices architecture patterns.
        -> Expanded skillset through Agentic AI projects, gaining expertise in LLM, RAG, Deep Agent skills, and contributing to cutting-edge AI solutions.

        -----------------------------------------------------

        Projects :

        1. Gen Re Insurance Project - TCS
                -> Duration: 1 year and 8 months
                -> Tech Stack: Core Java, Java 8, Spring Microservices, RabbitMQ
                -> Description: As a developer, I contributed to the development of an end-user insurance policy application that validated policies and sent automatic expiration warnings. 
        					   My responsibilities included implementing features on the application side using Core Java.

        2. IoT DMDC - Ericsson
                -> Duration: 1 year and 6 months
                -> Tech Stack: Core Java, Java 8, Spring Microservices, Spring Hibernate, Spring MVC, Restful Web Service, Spring Security, RabbitMQ
                -> Description: I was work on a virtual-based application that onboard virtual devices which supports various protocols and functionality. 
                         The app monitored device behaviors, generate CSV files and plotted graph in the GUI component.

        3. Covered Bond, Pricing - Centrico Software India Pvt Ltd, Agentic AI Chat Bot
                -> Duration: December 2023 - Present
                -> Tech Stack: Core Java, Java 17, JUnit, Mockito, Spring Microservices, Spring Hibernate, Spring MVC, Restful Web-Service, Spring Security, React JS, SQL, Agentic AI, Langchain, Langgraph
                -> Description: I have start with the Covered Bond project, focus on backend development and delivers a successful project in a short timeframe. 
                         Subsequently, I work on Platform Pricing, contributed to both fontend and backend development. 
                         Additionally, I expand my skills in the Agentic AI domain by implement a chatbot that process tasks through chat command.
        
        ------------------------------------------------------

        Competencies :

        -> Programming Languages: Proficient on Core java, ReactJS, and Python.
        -> Collaborative Development: Proven abilities for working effectively with cross functional teams to delivers high-quality solution
        -> Planning Analytics: Strong analytical skill, with experience in data driven decision makings and process optimization
        -> Troubleshooting: Skilled to identify and resolving complex technical issues, ensured minimal downtime and maximum efficiency
        -> Teamwork: A natural team player which have excellent communication and interpersonal skills, able for building strong relationships with colleagues, and stakeholders
        -> Self-Motivation: Drived by a passion for innovations and continuous learning, with a strong work ethics and commitment to delivering of results
        
        ------------------------------------------------------

        Additional information :

        Certifications: Computer Architecture NPTEL, Programming in Java, Langchain and LangGraph
        course Udemy Certificate
        Awards/Activities: Recognized as Best Employee of the year award - 2025, Won 2nd prize at CS
        INNOWIZ-2018, Tech Hunt, Secured 2nd prize at the Intramural Games organized by Sourashtra Co-
        Education Higher Secondary School, February 2014.

        ------------------------------------------------------

        Declaration :

        I hereby solemnly declare that the information mentioned above are true and to the best of knowledge
        and belief.

        ------------------------------------------------------
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
