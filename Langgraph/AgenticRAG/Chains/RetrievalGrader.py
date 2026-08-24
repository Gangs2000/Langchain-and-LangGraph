from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature= 0)

class GradeDocuments(BaseModel):
    """Binary score to check retrieved documents whether they are relevant or not"""
    binary_score: str = Field(description= "Documents are relevant to the question 'yes' or 'no'")
    
grade_structured_output = llm.with_structured_output(GradeDocuments)

system_message = """
        You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    """
    
messages = ChatPromptTemplate.from_messages([
    SystemMessage(content= system_message),
    HumanMessage(content = """
                    "Retrived documents :\n\n {documents} \n\n
                    "User Question : {question}
                 """)
])

retrieval_grader = messages | grade_structured_output