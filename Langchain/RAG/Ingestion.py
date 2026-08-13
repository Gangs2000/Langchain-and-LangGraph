from langchain_unstructured import UnstructuredLoader
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os

load_dotenv()

class Ingestion:
    
    def __init__(self):
        self.name = "Ingestion"
    
    def ingest_data(self, data_source):
        """
        Ingest data from the specified data source.
        """
        # Implementation for data ingestion
        print(f"Ingesting data from {data_source}")
        print("Begin ingesting process..")
        loader = UnstructuredLoader(file_path= data_source, chunking_strategy = "basic", max_characters=1000000)
        document = loader.load()
        print("Splitting text files")
        textSplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = textSplitter.split_documents(document)
        print(f"created {len(texts)} chunks")
        
        embeddings = OpenAIEmbeddings(api_key = os.environ.get("OPENAI_API_KEY"))
        
        print("Ingesting the documents into vector store")
        
        PineconeVectorStore.from_documents(texts, embeddings, index_name = os.environ.get("INDEX_NAME"))
        
        print("Finish")
        
if __name__ == "__main__":
    source = "/Users/GBS09645/OneDrive - Sella/Desktop/LangChain and LangGraph/langchain-course/Sources/Sample.txt"
    Ingestion().ingest_data(source)