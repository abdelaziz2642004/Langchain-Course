from dotenv import load_dotenv
load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import create_react_agent
from langchain_classic.agents import AgentExecutor
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

llm = ChatOllama(model="qwen2.5:3b")
tools = [TavilySearch()]

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

response = agent_executor.invoke({
    "input": "Hello,could u tell me who is the president of Egypt right now?"
})

print(response["output"])
