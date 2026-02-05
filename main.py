


import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from langchain_tavily import TavilySearch
load_dotenv()

    

llm = ChatOllama(model="qwen2.5:3b")
# llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(llm, tools=tools)
response = agent.invoke({"messages": [HumanMessage(content="Who is the current president of Egypt")]} )
print(response)
