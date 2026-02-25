from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

from dotenv import load_dotenv
load_dotenv() # this is to load the environment variables from the .env file

MAX_ITERATIONS = 10
MODEL = "qwen2.5:3b"
MODEL_PROVIDER="ollama"

# --- Tools (LangChain @tool decorator) ---

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f">> Executing get_product_price(product='{product}')")
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50
    }
    return prices.get(product.lower(), 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold.
    """
    print(f">> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")

    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }

    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)
    # ---  Agent loop -----




# this is to view this on langsmith
@traceable(name="test for langsmith")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools} #unnecessary
    llm = init_chat_model(model=MODEL,model_provider=MODEL_PROVIDER)
    llm_with_tools = llm.bind_tools(tools)
    messages = [
    SystemMessage(
        content=(
            "You are a helpful shopping assistant. "
            "You have access to a product catalog tool "
            "and a discount tool.\n\n"

            "STRICT RULES - you must follow these exactly:\n"

            "1. NEVER guess or assume any product price. "
            "You MUST call get_product_price first to get the real price.\n"

            "2. Only call apply_discount AFTER you have received "
            "a price from get_product_price. Pass the exact price "
            "returned by get_product_price - do NOT pass a made-up number.\n"

            "3. NEVER calculate discounts yourself using math. "
            "Always use the apply_discount tool.\n"

            "4. If the user does not specify a discount tier, "
            "ask them which tier to use - do NOT assume one."
        )
    ),
    HumanMessage(content=question)
]
    for i in range(1,MAX_ITERATIONS+1):
        print(f"--- Iteration {i} ---")
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        tool_calls = response.tool_calls
    
        if not tool_calls:
            # print(f"Final Answer: {response.content}")
            return response.content
        
        #Let's process only the first tool call for SIMPLICITY
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        print(f">> Calling tool: {tool_name} with args {tool_args}")
        
        tool_to_use = tools_dict.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"Tool {tool_name} not found")
        observation = tool_to_use.invoke(tool_args)
        print(f">> Tool Result: {observation}")

################# here ########################################
        messages.append(response) # append the response to the messages
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"])) # append the tool result to the messages
 ################# here ########################################
       
    
    return "Maximum iterations reached"

if __name__ == "__main__":
    print("Hello langchain agent (.bind_tools)")
    print()
    result= run_agent("What is the price of headphones with a silver discount?")
    print(f"Final result: {result}")