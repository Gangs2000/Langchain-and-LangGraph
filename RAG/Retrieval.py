from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorStore = PineconeVectorStore(index_name= os.environ.get("INDEX_NAME"), embedding=embeddings)

retrievar = vectorStore.as_retriever(kwargs={"k": 2})

promptTemplate = ChatPromptTemplate.from_template(
    """
        Please answer the question based on the given context only.
        
        Context : {context}
        
        Question : {question}
        
        Provide a detailed answer:
    """
)

class Retrieval:
    
    def __init__(self):
        pass
    
    def format_docs(self, docs):
        """Format all retrieved docs in one single core content."""
        print(">>> Formatting documents")
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Retrieval without langchain expression language
    def retrieve_response_without_lcel(self, query:str):
        """
            Retriving the result based on given context for the query
            without langchain expression language.
        """
        print(">>> Formatting messages")
        documents = retrievar.invoke(query)
        context = self.format_docs(documents)
        messages = promptTemplate.format_messages(question = query, context = context)
        print(">>> LLM Invocation")
        response = llm.invoke(messages)
        print(">>> Response generated")
        return response.content
    
    # Retrieval with langchain expression language
    def retrieve_response_with_lcel(self, dict):
        """
            Retrieving the result using langchain expression language.
            This helps to keep tracking all layers in one single trace file
            making debug much easier compare to naive flow.
        """
        question = dict["query"]
        retrieval_chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question") | retrievar | self.format_docs
            ) 
            | promptTemplate 
            | llm 
            | StrOutputParser()
        )
        return retrieval_chain.invoke({"question": question})
        
    
if __name__ == "__main__":
    query = "Please explain about NASDAQ exchange"
    print(">>> Retrieving content without langchain expression language")
    retrieval = Retrieval()
    response = retrieval.retrieve_response_without_lcel(query)
    print(f"Response using without lcel : {response}")
    print(">>> Retrieving content with langchain expression language")
    response = retrieval.retrieve_response_with_lcel({"query": query})
    print(f"Response using with lcel : {response}")
    
    
