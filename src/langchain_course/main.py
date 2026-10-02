import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

# llm = ChatOpenAI(
#     model="gpt-5-nano",  # your Azure deployment name
#     base_url=AZURE_OPENAI_ENDPOINT,
#     api_key=token_provider,  # callable that handles token refresh
# )

# Ollama LLM
llm = ChatOllama(model="gpt-oss:20b")

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches the web for information.
#     Args:
#         query: The query to search for.
#     Returns:
#         The search results.
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query)



tools = [TavilySearch()]

agent = create_agent(model=llm, tools=tools)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job openings in the field of full stack development in Hong Kong.")})
    print(result)


if __name__ == "__main__":
    main()