from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature= 0)

class GradeHallucination(BaseModel):
    """Binary score to check generated answer is hallucinated"""
    
    binary_score: bool = Field(description= "Answer is grounded in the facts 'yes' or 'no'")
    
hallucinated_strucuted_output = llm.with_structured_output(GradeHallucination)

system_message = """
                You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
                Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
            """

messages = ChatPromptTemplate.from_messages([
    SystemMessage(content= system_message),
    HumanMessage(content= """Set of facts: \n\n {documents} \n\n LLM generation: {generation}""")
])

hallucination_grader: RunnableSequence = messages | hallucinated_strucuted_output