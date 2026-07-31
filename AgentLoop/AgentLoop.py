from langchain.tools import tool
from langsmith import traceable
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

MAX_ITERATIONS = 3
MODEL = "gpt-4"

class AgentLoop:
    @tool
    def get_product_price(product:str) -> float:
        """Fetch the price/amount for the given product type"""
        products = {"laptop": 1000.50, "mobile": 500.90, "tablet": 300.60}
        price = products.get(product, 0.0)
        print(f" >> Fetching price for {product} and price is '{price}'")
        return price
    
    @tool
    def apply_discount(price:float, category:str) -> float:
        """
            Apply discount for the given price.
            Available discount categories are Bronze, Silver and Gold
        """
        discounts = {"bronze": 10.0, "silver":20.0, "gold": 25.0}
        discount = discounts.get(category.lower(), 0.0)
        appliedDiscount = round(price * (1 - discount/100), 2)
        print(f" >> Applied {category} tier discount and the price post discount is '{appliedDiscount}'")
        return appliedDiscount
    
    # Agent loop begins here
    @traceable(name="Langchain agent loop")
    def runAgent(self, query:str):
        tools = [AgentLoop.get_product_price, AgentLoop.apply_discount]
        tool_dict = {t.name: t for t in tools}
        llm = init_chat_model(f"openai:{MODEL}", temperature = 0)
        llm_tools = llm.bind_tools(tools=tools)
        
        print(f" Question {query}")
        print("*" * 60)
        
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful shopping assistant. "
                    "You have access to a product catalog tool "
                    "and a discount tool.\n\n"
                    "STRICT RULES — you must follow these exactly:\n"
                    "1. NEVER guess or assume any product price. "
                    "You MUST call get_product_price first to get the real price.\n"
                    "2. Only call apply_discount AFTER you have received "
                    "a price from get_product_price. Pass the exact price "
                    "returned by get_product_price — do NOT pass a made-up number.\n"
                    "3. NEVER calculate discounts yourself using math. "
                    "Always use the apply_discount tool.\n"
                    "4. If the user does not specify a discount tier, "
                    "ask them which tier to use — do NOT assume one."
                )
            ),
            HumanMessage(content=query)
        ]
        
        for iteration in range(1, MAX_ITERATIONS + 1):
            print(f"\n -- Iteration {iteration} -- ")
            ai_message = llm_tools.invoke(messages)
            tool_calls = ai_message.tool_calls
            
            if not tool_calls:
                print(f"\n Final answer : {ai_message.content}")
                return ai_message.content
            
            # Obtain tool call meta data such as id, name and args
            tool_call = tool_calls[0]
            tool_id = tool_call.get("id")
            tool_args = tool_call.get("args", {})
            tool_name = tool_call.get("name")
            
            print(f"\n Tool exected {tool_name} with args {tool_args}")
            
            tool_to_use = tool_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Tool '{tool_name}' not found")
            
            observation = tool_to_use.invoke(tool_args)
            
            print(f" Tool Result : {observation}")
            
            messages.append(ai_message)
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_id))
            
        print("Max Iteration completed.. LLM couldn't find solution!!")
        return None

def main():
    print("Agent loop demonstration without using langchain framework!!")
    query = input("Enter product name to get discounted price : ")
    response = AgentLoop().runAgent(query)
    print(f" >> Final response -> Applied discount price is '{response}'")

if __name__ == "__main__":
    main()