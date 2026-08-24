from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature= 0)

class RouteQuery(BaseModel):
    dataSource: Literal["vectore_db", "web_search"] = Field(
        description= "Given a user question choose to route it to web search or vectore store db "
    ) 
    
route_structured_output = llm.with_structured_output(RouteQuery)

system_message = """
                    You are an agent designed for routing purpose based on user question.
                    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
                    Use the vectorstore for questions on these topics. For all else, use web-search.
                """
                
messages = ChatPromptTemplate.from_messages([
    SystemMessage(content= system_message),
    HumanMessage(content= "Question : {question}")
])

question_router = messages | route_structured_output