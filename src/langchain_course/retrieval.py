import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

load_dotenv()

from langchain_unstructured import UnstructuredLoader

ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = ROOT / "mediumblog1.txt"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

EMBEDDING_MODEL = "text-embedding-3-small"
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), os.getenv("AZURE_TOKEN_PROVIDER")
)
embeddings = OpenAIEmbeddings(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model=EMBEDDING_MODEL,
    api_key=token_provider,
)

llm = ChatOpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=token_provider,
    model="gpt-5-nano",
    temperature=0,
)

from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"), embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


def retrieval_chain_without_lcel(query: str) -> str:
    docs = retriever.invoke(query)
    context = format_docs(docs)
    prompt = prompt_template.invoke({"context": context, "question": query})
    response = llm.invoke(prompt)
    return response.content


def create_retrieval_chain_with_lcel():
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


def main() -> None:
    print("Hello from langchain-course!")

    query = "What is Pinecone in machine learning?"

    # result_without_lcel = retrieval_chain_without_lcel(query)

    result_with_lcel = create_retrieval_chain_with_lcel().invoke({"question": query})

    print(result_with_lcel)


if __name__ == "__main__":
    main()
