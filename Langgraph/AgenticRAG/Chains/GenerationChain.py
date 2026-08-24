from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessage(
            content = 
                """
                    You are an assistant for question-answering tasks.
                    Use the following pieces of retrieved context to answer the question.
                    If you don't know the answer, just say that you don't know.
                    Use three sentences maximum and keep the answer concise.
                    Question: {question} \nContext: {context} \nAnswer:
                """),
    ]
)

generation_chain = prompt | llm | StrOutputParser()