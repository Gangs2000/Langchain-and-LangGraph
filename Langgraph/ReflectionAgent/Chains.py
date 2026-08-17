from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
                    You are a LinkedIn post advisor, Generate critique and proper recommendations for user's post.
                    Please detailed recommendations, including content length, style, virality etc..
                  """
                  ),
    MessagesPlaceholder(variable_name="messages")
])

generation_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
                    You are a LinkedIn techie influencer assistant tasked with writing excellent posts.
                    Generate the best LinkedIn post possible for the user's 
                    If the user provides critique, respond with a revised version of your previous attempts.
                  """),
    MessagesPlaceholder(variable_name="messages")
])

llm = ChatOpenAI(model="gpt-4", temperature=0)
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
