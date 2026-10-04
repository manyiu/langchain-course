from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3.8:27b-mlx"


@traceable(name="get_product_price", run_type="tool")
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


@traceable(name="apply_discount", run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price"""
    print(f"Applying discount tier {discount_tier} to price {price}")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}

    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catelog",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "The name of the product to look up"},
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The price to apply the discount to"},
                    "discount_tier": {"type": "string", "description": "The discount tier to apply"},
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

@traceable(name="Ollama LLM", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, messages=messages, tools=tools_for_llm)

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that can help with product pricing and discounts."
                "You have access to the following tools: {tools_dict}"
                "Use the tools to answer the question."
                "You must first call get_product_price to get the price of the product."
                "Then you must call apply_discount to apply the discount to the price."
                "You must then return the exact final price."
                "If the product or discount is not found, tell the user that the product or discount is not found."
            )
        },
        {
            "role": "user",
            "content": question,
        }
    ]

    for i in range(MAX_ITERATIONS):
        response = ollama_chat_traced(messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        print(f"Tool calls: {tool_calls}")

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        tool_to_use = tools_dict.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"Tool {tool_name} not found")

        tool_result = tool_to_use(**tool_args)
        print(f"Tool result: {tool_result}")

        messages.append(ai_message)
        messages.append({
            "role": "tool",
            "tool_name": tool_name,
            "content": str(tool_result),
        })

    raise ValueError("Max iterations reached")

if __name__ == "__main__":
    print("Starting the agent...")

    run_agent("What is the price of a laptop with a gold discount?")
