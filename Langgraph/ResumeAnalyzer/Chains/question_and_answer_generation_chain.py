from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOllama(model="llama3", num_ctx=16384, num_predict=4096, temperature=0)


class QueAndAnsGenerationLayout(BaseModel):
    question: str = Field(
        description="Technical question directly derived from candidate's experience or skills"
    )
    options: list[str] = Field(description="List of 4 distinct choices (A, B, C, D)")
    answer: str = Field(
        description="The exact string match of the correct option from the options list"
    )


class QuestionAndAnswers(BaseModel):
    """Schema to generate technical questions, options, and answers directly from resume content."""

    difficulty: str = Field(
        description="Difficulty level of the generated questions: Easy, Moderate, or Hard",
        default="Easy",
    )
    generated_questions: list[QueAndAnsGenerationLayout] = Field(
        description="List of generated multiple-choice technical questions"
    )


ques_and_answer_structured_output = llm.with_structured_output(QuestionAndAnswers)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a technical interviewer creating technical multiple-choice questions strictly from a candidate's resume skillsets.

                STRICT GROUNDING & QUESTION GENERATION RULES:

                1. SKILL IDENTIFICATION:
                First, identify and extract the core technical skills, programming languages, frameworks, databases, cloud platforms, and tools explicitly listed in the candidate's resume 
                (e.g., Core Java, Spring Microservices, PostgreSQL, Docker, LangChain).

                2. TECHNICAL ASSESSMENT FOCUS:
                Generate technical interview questions designed to test the candidate's actual mastery and theoretical/practical knowledge of those identified technologies.
                - DO NOT ask personal or resume-specific experience questions (e.g., do NOT ask "What project did the candidate work on at TCS?").
                - DO test conceptual understanding, architecture, best practices, debugging, and framework behaviors of the skills found on the resume 
                (e.g., "In Spring Microservices, how does circuit breaking work using Resilience4j?").

                3. DIFFICULTY LEVEL:
                Strictly align question complexity with the requested difficulty level:
                - Easy: Basic concepts, definitions, core syntax, and basic tool usage.
                - Moderate: Architectural patterns, performance trade-offs, common design choices, and intermediate features.
                - Hard: Deep framework internals, edge cases, distributed system failure modes, and advanced optimization.

                4. NO UNLISTED TECHNOLOGIES:
                You are strictly forbidden from asking questions about technologies, languages, or tools that do NOT appear on the candidate's resume.

                5. VALID OPTIONS & ANSWERS:
                - Always provide exactly 4 distinct options (A, B, C, D).
                - Exactly ONE option must be the correct answer.
                - The correct answer string must match one of the items in the options list verbatim.
            """,
        ),
        (
            "human",
            """
                Target Difficulty: {difficulty}

                Candidate Resume:
                ---
                {resume_content}
                ---
                Generate technical multiple-choice questions matching the requested schema and strictly following the system rules above.
                Extract the candidate's technical skills, frameworks, and tools from the resume, and generate conceptual domain questions testing theoretical and practical mastery of those technologies
            """,
        ),
    ]
)

ques_and_answer_generation_grader: RunnableSequence = (
    prompt | ques_and_answer_structured_output
)
