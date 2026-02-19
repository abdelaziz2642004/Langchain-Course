from dotenv import load_dotenv
load_dotenv()

from langchain_classic.agents import create_react_agent
from langchain_classic.agents import AgentExecutor
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from schemas import AgentResponse
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda


# ---------------------------
# LLM + Tools
# ---------------------------
llm = ChatOllama(model="qwen2.5:3b")
tools = [TavilySearch()]

# ---------------------------
# Output Parser
# ---------------------------
output_parser = PydanticOutputParser(
    pydantic_object=AgentResponse
)

# ---------------------------
# Custom PromptTemplate
# ---------------------------
prompt = PromptTemplate(
    input_variables=[
        "input",
        "agent_scratchpad",
        "tools",
        "tool_names",
        "format_instructions"
    ],
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    output_parser=output_parser,
).partial(
    format_instructions=output_parser.get_format_instructions()
)

# ---------------------------
# Create Agent
# ---------------------------
agent = create_react_agent(
    llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# ---------------------------
# NORMAL FUNCTIONS (instead of lambda)
# ---------------------------

def extract_output(result: dict) -> str:
    return result["output"]

def parse_output(text: str):
    return output_parser.parse(text)

# Wrap them in RunnableLambda
extract_output_runnable = RunnableLambda(extract_output)
parse_output_runnable = RunnableLambda(parse_output)

# ---------------------------
# Build Chain
# ---------------------------
chain = agent_executor | extract_output_runnable | parse_output_runnable

# ---------------------------
# Run
# ---------------------------
response = chain.invoke(
    {"input": "Hello, could you tell me who was the president of Egypt in 2004?"},
    handle_parsing_errors=True
)

print(response)
