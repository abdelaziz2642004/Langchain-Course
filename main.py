from dotenv import load_dotenv

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import create_react_agent
from langchain_classic.agents import AgentExecutor
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda


llm = ChatOllama(model="qwen2.5:3b")
tools = [TavilySearch()]
#NEW
structured_llm = llm.with_structured_output(AgentResponse)
#NEW

# prompt = hub.pull("hwchase17/react")
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad","tools","tool_names","format_instructions"],
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
).partial(format_instructions="") # DOESN'T matter any more

agent = create_react_agent(structured_llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#NEW
extract_output = RunnableLambda(lambda x: x["output"])
chain= agent_executor | extract_output | structured_llm
#NEW


response = chain.invoke(
    {"input": "Hello,could u tell me who is the president of Egypt in 2004?"},
    handle_parsing_errors=True
)

print(response)

