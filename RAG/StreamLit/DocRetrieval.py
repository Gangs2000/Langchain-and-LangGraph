from typing import Dict, Any
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.messages import ToolMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Pinecone vector store config
vectorStore = PineconeVectorStore(index_name= "langchain-docs-index", embedding= embeddings)

# init chat model
model = init_chat_model("gpt-4", model_provider= "openai")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str) :
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    
    # Retrieve docs from vector store
    retrived_docs = vectorStore.as_retriever().invoke(query, k=2)
    
    # Serializing the docs
    serialized_content = "\n\n".join(
        (f"Source : {doc.metadata.get("source", "Unknown")} \n\n Content : {doc.page_content}")
        for doc in retrived_docs
    )
    
    # Return both serialized and JSON content
    return serialized_content, retrived_docs

def execute_query(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Message creation
    messages = [
        SystemMessage(content="""
                            You are a helpful AI assistant that answers questions about LangChain documentation.
                            You have access to a tool that retrieves relevant documentation.
                            Use the tool to find relevant information before answering questions.
                            Always cite the sources you use in your answers.
                            If you cannot find the answer in the retrieved documentation, say so.
                      """),
        HumanMessage(content=query)
    ]
    
    # Agent creation and invocation
    agent = create_agent(model, tools=[retrieve_context])
    response = agent.invoke({"messages": messages})
    
    # Retrieve last message from LLM response
    answer = response["messages"][-1].content
    
    context_docs = []

    # Extract docs from ToolMessage if artifact is present
    for message in response["messages"]:
        # Check if message is ToolMessage and it has attribute artifact
        if isinstance (message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance (message.artifact, list):
                context_docs.extend(message.artifact)
                
    return {
        "answer" : answer,
        "context": context_docs
    }

if __name__ == "__main__":
    query = "Share me insightful information about LangChain and LangGraph"
    result = execute_query(query)
    print(result)