from dotenv import load_dotenv
from langchain.agents import create_agent

from langchain.tools import tool

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama

load_dotenv()

tavily = TavilyClient()


@tool
def search(query: str):
    """Search the web for information about a given query.
    Args:
        query (str): The query to search the web for.
    Returns:
        str: The search results.
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)    

llm = ChatOpenRouter(
    model="openai/gpt-5",
)
llm_1=ChatOllama(
    model="gemma4",
    temperature=0.2,
    max_tokens=1024,
    max_retries=2,
)

# agent = create_agent(
#     model=llm,
#     tools=[search],
#     system_prompt="You are a helpful assistant.",
# )

# res=agent.invoke({"messages": [HumanMessage(content="What is the weather in London?")]})
# print(res)
    
#  other option is to use inbuilt tavily function as a tool

agent = create_agent(
    model=llm_1,
    tools=[TavilySearch()],
    system_prompt="You are a helpful assistant.",
)
res=agent.invoke({"messages": [HumanMessage(content="What is the weather in London, UK?")]})
print(res)


