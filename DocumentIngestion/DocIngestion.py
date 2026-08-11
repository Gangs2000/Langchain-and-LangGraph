import os, ssl, asyncio, certifi
from typing import List, Any, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyMap, TavilyExtract
from langsmith import traceable

load_dotenv()

# SSL certification configuration
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Embedding config
embeddings = OpenAIEmbeddings(model= "text-embedding-3-small", chunk_size= 50, retry_min_seconds= 10)
# Pinecone vector store config
vectorStore = PineconeVectorStore(index_name= "langchain-docs-index", embedding= embeddings)

#Tavily config
tavily_crawl = TavilyCrawl()
tavily_map = TavilyMap(max_depth = 1, max_breath = 15, max_pages = 1000)
tavily_extract = TavilyExtract()

async def index_document_async(documents: List[Document], batch_size: int = 50):
    """Process batch for the given list of documents"""
    
    # Creating batches
    batches = [documents[i : i + batch_size] for i in range (0, len(documents), batch_size)]
    
    # Process all batches concurrently
    async def add_batch(batches: List[Document], batch_num: int):
        try:
            await vectorStore.aadd_documents(batches)
            print(f">>> Vector store index -> Successfully added batch {batch_num}/{len(batches)} ({len(batches)} documents)")
        except Exception as e:
            return False
        return True
    
    # Process each batch concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions = True)
    
    successful = sum(1 for result in results if result is True)
    
    if successful == len(batches):
        print(f"All batches processed successfully in total : {successful}/{len(batches)}")
    else :
        print(f" Processed {successful}/{len(batches)} batches successfully")

@traceable(name="Document ingestion process")
async def main():
    """Main async function to orchastrate document ingestion by crawling tavily"""
    print(">>> Executing document ingestion process")
    crawledResponse = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 2,
        "extract_depth": "advanced",
        "instructions": "Content on langchain and langgraph"
    })
    
    all_docs = []
    # Crawl and extract content from the crawled response
    if len(crawledResponse["results"]) == 0:
        print("No response found in crawl")
    for result in crawledResponse["results"]:
        all_docs.append(Document(
            page_content = str(result["raw_content"]),
            metadata = {"source": str(result["url"])}
        ))
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap = 200)
    splitted_docs = text_splitter.split_documents(all_docs)
    await index_document_async(splitted_docs, batch_size = 500)
    
if __name__ == "__main__":
    asyncio.run(main())