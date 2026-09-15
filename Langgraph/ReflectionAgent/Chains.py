from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a resume analyzer. Generate a detailed critique and proper recommendations
                for the user's content based on the latest AI generation. Analyze content length, style, and impact.
            """,
        ),
        (
            "human",
            "Original Request: {initial_post}\n\nGenerated Content to Critique: {generated_post}",
        ),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="""
                    You are a resume analyzer assistant tasked with analyzing and improving given title and content.
                    Generate the best possible content based on the user's request. 
                    Ensure that the generated content is clear, concise, and impactful.
                    If the user provides critique, respond with a revised version of your previous attempts.
                  """),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOllama(
    model="llama3",
    num_thread=4,
    num_ctx=2048,
    temperature=0,
)
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
