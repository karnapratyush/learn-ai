from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()


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


llm_1 = ChatOllama(
    model="gemma4",
    temperature=0.2,
    max_tokens=1024,
    max_retries=2,
)


class Source(BaseModel):
    """Source of the information."""

    url: str = Field(description="URL of the source.")


class AgentResponse(BaseModel):
    """Response of the agent."""

    answer: str = Field(description="The agent's anwer to the query.")
    sources: list[Source] = Field(
        default_factory=list,
        description="List of sources used in generating the answer",
    )


agent = create_agent(
    model=llm_1,
    tools=[TavilySearch()],
    system_prompt="You are a helpful assistant.",
    response_format=AgentResponse,
)
res = agent.invoke(
    {"messages": [HumanMessage(content="Give me 5 latest job posting for GenAI roles")]}
)
print(res)
