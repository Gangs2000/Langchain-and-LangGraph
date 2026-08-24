from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature= 0)

class GraderAnswer(BaseModel):
    """Binary score to check generated answer relevant to the question"""
    
    binary_score: bool = Field(description= "Answer addresses the question 'yes' or 'no'")
    
answer_strucuted_output = llm.with_structured_output(GraderAnswer)

system_message = """
                You are a grader assessing whether an LLM generation is addressed the question \n 
                Give a binary score 'yes' or 'no'. 'Yes' means that the answer is relevant
            """

messages = ChatPromptTemplate.from_messages([
    SystemMessage(content= system_message),
    HumanMessage(content= """User Question \n\n {question} \n\n LLM generation: {generation}""")
])

answer_grader: RunnableSequence = messages | answer_strucuted_output