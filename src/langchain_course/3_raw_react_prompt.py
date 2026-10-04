import re
import inspect

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
    price = float(price)
    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

def get_tool_description(tools_dict: dict) -> str:
    descriptions = []

    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function)
        descriptions.append(f"{tool_name}{signature}: {docstring}")
    return "\n".join(descriptions)


tool_names = ", ".join(tools.keys())
tool_descriptions = get_tool_description(tools)

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


@traceable(name="Ollama LLM", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    print(f"Question: {question}")

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for i in range(MAX_ITERATIONS):
        print(f"Iteration {i+1}")
    
        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={
                "temperature": 0.0,
                "stop": ["\nObservation"],
            },
        )
        output = response.message.content
        print(f"Output: {output}")

        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"Final answer: {final_answer}")
            return final_answer

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "[Parsing] Error: Could not parse action or action input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"[Tool selection] Using tool: {tool_name} with input: {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"Tool executing: {tool_name}({args})")

        if tool_name not in tools:
            observation = f"Error: Tool {tool_name} not found. Available tools: {list[str(tools.keys())]}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"[Tool result] Observation: {observation}")

        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    raise ValueError("Max iterations reached")


if __name__ == "__main__":
    print("Starting the agent...")

    run_agent("What is the price of a laptop with a gold discount?")
