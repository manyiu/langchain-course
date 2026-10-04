from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3.8:27b-mlx"


@tool
def get_product_price(product_name: str) -> float:
    """Look up the price of a product in the catelog"""
    print(f"Looking up the price of {product_name}")
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
        "mouse": 49.99,
        "monitor": 299.99,
        "printer": 199.99,
        "scanner": 149.99,
        "speaker": 199.99,
        "tablet": 299.99,
        "smartphone": 999.99,
        "smartwatch": 299.99,
        "smart home hub": 199.99,
        "smart TV": 999.99,
    }

    print(f"Price of {product_name}: {prices.get(product_name.lower(), 0.0)}")

    return prices.get(product_name.lower(), 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price"""
    print(f"Applying discount tier {discount_tier} to price {price}")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}

    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(model=f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant that can help with product pricing and discounts."
                "You have access to the following tools: {tools_dict}"
                "Use the tools to answer the question."
                "You must first call get_product_price to get the price of the product."
                "Then you must call apply_discount to apply the discount to the price."
                "You must then return the exact final price."
                "If the product or discount is not found, tell the user that the product or discount is not found."
            )
        ),
        HumanMessage(content=question),
    ]

    for i in range(MAX_ITERATIONS):
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        print(f"Tool calls: {tool_calls}")

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_call_id = tool_call.get("id")

        tool_to_use = tools_dict.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"Tool {tool_name} not found")

        tool_result = tool_to_use.invoke(tool_args)
        print(f"Tool result: {tool_result}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call_id))

    raise ValueError("Max iterations reached")

if __name__ == "__main__":
    print("Starting the agent...")

    run_agent("What is the price of a laptop with a gold discount?")
