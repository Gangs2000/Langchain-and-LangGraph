from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOllama(model="llama3", num_ctx=16384, num_predict=4096, temperature=0)


class GrammeticalErrors(BaseModel):
    """Schema to map the mistakes along with sentences, and correct words to be replaced"""

    mistaken_word: str = Field(description="Mistaken word found in content")
    reason: str = Field(description="Provide the detailed reason of Why it is wrong")
    correct_word: str = Field(
        description="Correct word equivalent to the mistaken word"
    )


class GrammerValidator(BaseModel):
    """Schema for grammatical agent to map the results with the pydantic field"""

    grammer_mistake_present: bool = Field(
        description="If any grammatical mistake present set True, else set False"
    )
    grammer_mistakes: list[GrammeticalErrors] = Field(
        description="Provide grammatical mistakes if any present, must be empty when no grammer errors found"
    )


grammertical_strucuted_output = llm.with_structured_output(GrammerValidator)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an expert Grammar Validator Agent specializing in resume evaluations.

                YOUR TASK:
                Analyze the provided resume text exclusively for objective English grammar, spelling, dynamic tense errors, and severe typos.

                RULES FOR EVALUATION:
                1. Ignore standard resume conventions like action-verb fragments (e.g., "Led a team of 5" is VALID).
                2. Do NOT suggest subjective style preferences or rephrasings if the original text is grammatically correct.
                3. If no grammatical or spelling errors exist, you MUST set `grammer_mistake_present` to False and return an empty list for `grammer_mistakes`.
                4. In case if any grammatical errors present, please provide the `mistaken_word`, details reason `reason` and correct equivalent word `correct_word`
                ( For example you can follow this patten `(mistaken_word - reason - correct word)`)
                5. Don't forget to include all found errors into the list, which is mandetory in order to keep tracking what are fixed.
            """,
        ),
        (
            "human",
            """
                Here is my resume content.
    
                Resume Content :
                {resume_content}
            """,
        ),
    ]
)

grammer_validation_grader: RunnableSequence = prompt | grammertical_strucuted_output
