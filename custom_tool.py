import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tavily import TavilyClient
load_dotenv()

# tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) #unecessary parameter
tavily= TavilyClient()
@tool
def  Test_Tool(query: str) -> str:
    # MAKE THIS DESCRIPTION AS DETAILED AS POSSIBLE IT IS IMPORTANT
    # SO THE LLM CAN KNOW WHETHER TO RUN THIS FUNCTION OR NOT
    """
    Tool that searches for information on the web.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    print(f"Searching for: {query}")
    output = tavily.search(query=query)
    return output
    

llm = ChatOllama(model="qwen2.5:3b")
# llm = ChatOpenAI(model="gpt-5")
tools = [Test_Tool]
agent = create_agent(llm, tools=tools)
response = agent.invoke({"messages": [HumanMessage(content="Who is the current president of Egypt")]} )
print(response)
