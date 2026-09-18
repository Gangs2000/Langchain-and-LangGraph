from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOllama(model="llama3", num_ctx=16384, num_predict=4096, temperature=0)


from pydantic import BaseModel, Field


class ATSResumeScore(BaseModel):
    ats_score: float = Field(
        description="ATS Score for the given resume, scale range varies from 1.0 to 100.0",
        ge=1.0,
        le=100.0,
    )


ats_structured_output = llm.with_structured_output(ATSResumeScore)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an expert Applicant Tracking System (ATS) Evaluator. Your task is to analyze the provided resume, calculate a deterministic ATS compatibility score between 1.0 and 100.0, and provide clear actionable feedback.

                EVALUATION CRITERIA (100 Points Total):
                1. Section Organization (Max 25 pts): Use of standard headings (e.g., Experience, Education, Skills, Projects).
                2. Date Formatting (Max 15 pts): Clear and consistent month/year structures (e.g., "Jan 2022 - Present").
                3. Quantified Achievements (Max 35 pts): Strong action verbs paired with metrics/numbers showing impact rather than basic job duties.
                4. Readability & Structure (Max 25 pts): Clean bulleting, standard English, and absence of non-parseable layouts.

                OUTPUT REQUIREMENTS:
                - Calculate `ats_score` as a float between 1.0 and 100.0 based strictly on the criteria above.
            """,
        ),
        (
            "human",
            """
                Please evaluate the following resume content: 
                --- RESUME TEXT ---
                {resume_content}
            """,
        ),
    ]
)

ats_resume_validatoion_grader: RunnableSequence = prompt | ats_structured_output
