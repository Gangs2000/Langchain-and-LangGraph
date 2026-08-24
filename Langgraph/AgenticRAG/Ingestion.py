from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

load_dotenv()

print("======== Begin Document Ingestion =========")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

documents = [UnstructuredLoader(web_url=url, chunking_strategy= "basic", max_characters=1000000).load() for url in urls]
document_list = [item for sublist in documents for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

document_split = text_splitter.split_documents(document_list)

embeddings = OpenAIEmbeddings()

PineconeVectorStore.from_documents(document_split, embeddings, index_name = "advanced-rag")

retriever = PineconeVectorStore(embedding= embeddings, index_name = "advanced-rag").as_retriever()

print("======== Finish Document Ingestion =========")
