from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gemma4"


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""

    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@tool
def get_product_discount(product: str) -> float:
    """Look up the discount percentage for a product."""
    print(f"    >> Executing get_product_discount(product='{product}')")
    discounts = {"laptop": 0.10, "headphones": 0.05, "keyboard": 0.15}
    return discounts.get(product, 0)


@tool
def apply_discount(price: float = 0, discount: float = 0.0) -> float:
    """Apply a discount to the product price."""
    print(f"    >> Executing apply_discount(price={price}, discount={discount})")
    return price * (1 - discount)


llm = init_chat_model(f"ollama:{MODEL}", temperature=0)


@traceable(name="LangChain Agent Loop")
def run_agent(prompt: str):
    llm_with_tools = llm.bind_tools(
        [get_product_price, get_product_discount, apply_discount]
    )
    print(f"Question: {prompt}")
    print("=" * 60)
    tools_map = {
        "get_product_discount": get_product_discount,
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price.\n"
                "write an example for each tool when and how to use item"
                """Example 1: get_product_price
                User: What is the price of a laptop?
                Agent: {"name": "get_product_price", "args": {"product": "laptop"}}
                returns price of product in float

                Example 2: get_product_discount
                User: What is the discount percentage for a laptop?
                Agent: {"name": "get_product_discount", "args": {"product": "laptop"}}
                returns discount percentage in float

                Example 3: apply_discount
                User: Apply a discount of 10% to a laptop
                Agent: {"name": "apply_discount", "args": {"price": "1000", "discount": 0.10}}
                returns final price after discount in float
                """
            )
        ),
        HumanMessage(content=prompt),
    ]
    iterations = MAX_ITERATIONS
    while iterations > 0:
        print(f"\n--- Iteration {iterations} ---")
        ai_message = llm_with_tools.invoke(messages)
        print("ai_message", ai_message)
        messages.append(ai_message)  # Must append the AI's message to history!
        if ai_message.tool_calls:
            print("==" * 100)
            print("all_tools", ai_message.tool_calls)
            print("==" * 100)
            for tool_call in ai_message.tool_calls:
                print("tool_call", tool_call)
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                if tool_name not in tools_map:
                    print(f"Tool not found: {tool_name}")
                    continue
                tool_result = tools_map[tool_name].invoke(tool_args)
                print("tool_result", tool_result)
                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )
        else:
            print("No tool calls")
            print(ai_message.content)
            break
        iterations -= 1


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the final discounted price of a laptop?")
