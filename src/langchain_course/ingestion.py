import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
load_dotenv()

from langchain_unstructured import UnstructuredLoader
ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = ROOT / "mediumblog1.txt"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
EMBEDDING_MODEL = "text-embedding-3-small"
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    os.getenv("AZURE_TOKEN_PROVIDER")
)
embeddings = OpenAIEmbeddings(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model=EMBEDDING_MODEL,
    api_key=token_provider,
)

from langchain_pinecone import PineconeVectorStore

def main() -> None:
    print("Hello from langchain-course!")

    loader = UnstructuredLoader(file_path=FILE_PATH, chunking_strategy="basic", max_characters=1000000)
    document = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))

if __name__ == "__main__":
    main()