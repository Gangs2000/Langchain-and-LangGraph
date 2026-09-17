from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOllama(
    model="llama3", num_thread=4, num_ctx=4096, num_predict=2048, temperature=0
)


class SuggestionAdvice(BaseModel):
    category: str = Field(description="The category of the suggestion")
    issue: str = Field(description="The issue or opportunity identified in the resume")
    recommendation: str = Field(
        description="The exact rewrite, added keyword, or structural adjustment to make"
    )


class Suggestions(BaseModel):
    suggestions: list[SuggestionAdvice] = Field(
        description="List of specific suggestions to improve the resume"
    )


suggestions_system_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ 
                You are an expert ATS (Applicant Tracking System) Resume Optimization Specialist. Your sole purpose is to provide direct, 
                high-impact suggestions to maximize resume parse rates and keyword alignment.

                ## Core Rules
                1. Quantity: Generate strictly 4 to 6 actionable improvement suggestions.
                2. Focus: Base every suggestion explicitly on ATS optimization (e.g., keyword density, formatting standardisation, measurable metrics, 
                    section headers, removing parsing obstacles).
                3. Precision: Avoid generic advice (e.g., "make it better"). Pinpoint specific lines, phrases, or structural elements from the provided resume.
                4. Tone: Concise, professional, and directly actionable.

                ## Format Requirements
                Output each suggestion as a single item with two clear parts:
                `category`: "Keywords | Formatting | Action Verbs | Metrics | Structure",
                `issue`: "Brief description of the problem in the text",
                `recommendation`: "Exact fix or rewrite to apply" 
            """,
        ),
        (
            "human",
            "Analyze this resume and provide ATS optimization suggestions:\n\n{resume_content}",
        ),
    ]
)

suggestion_extractor = llm.with_structured_output(Suggestions)

suggestion_extractor_grader: RunnableSequence = (
    suggestions_system_prompt | suggestion_extractor
)

suggestion_applier_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an expert Resume Writer specializing in ATS (Applicant Tracking System) optimization. Your goal is to rewrite and enhance a candidate's resume 
                by strictly implementing provided feedback and suggestions.

                CRITICAL REWRITING RULES:
                1. FIX ONLY PRESCRIBED SUGGESTIONS: Apply suggestions strictly based on the provided list of identified suggestions.
                2. PRESERVE ORIGINAL FORMATTING: Maintain the original sentence structure, tone, dynamic action verbs, and resume styling (e.g., bullet points, past/present tense choices for roles).
                3. DO NOT OVER-EDIT: Do not rephrase, rewrite, or attempt to "improve" text that was not flagged in the suggestion list.
                4. CLEAN OUTPUT: Return ONLY the corrected version of the text. Do not include commentary, intro, or wrap-up notes.
            """,
        ),
        (
            "human",
            """
                Please update the resume based on the current evaluation metrics below.
                --- CURRENT ATS SCORE ---
                {ats_resume_score} / 100.0
                --- IMPROVEMENT SUGGESTIONS TO APPLY ---
                {suggestions}
                --- ORIGINAL RESUME TEXT ---
                {resume_content}
            """,
        ),
    ]
)

suggestion_applier_grader = suggestion_applier_prompt | llm | StrOutputParser()
