from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

def main():
    print("Welcome to LangChain course!! ")
    repeat = True
    while (repeat):
        query = input("Ask a question : ")
        
        template = """
            Could you please help user to get information about query {query}.
            I want you to give only short answers, don't give large response.
        """
        prompt_template = PromptTemplate(input_variables=["query"], template=template)
        
        llm = ChatOpenAI(temperature=0, model="gpt-4")
        
        chain = prompt_template | llm
        
        response = chain.invoke(input={"query": query})
        
        print(response.content)

        choice = input("Would like to exit from chat further enter YES/NO : ")
        
        if (choice.upper() == "YES"):
            repeat = False


if __name__ == "__main__":
    main()
