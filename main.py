import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

llm = ChatOpenAI(temperature= 0)

MCP_SERVER_ABSOLUTE_PATH = str(Path(__file__).parent / "servers" / "Math.py")

stdio_server_params = StdioServerParameters(
    command= "python",
    args = [MCP_SERVER_ABSOLUTE_PATH]
)

async def main():
    async with stdio_client(stdio_server_params) as (read, write):
        async with ClientSession(read_stream= read, write_stream= write) as session:
            await session.initialize()
            print("=== Session Initialized ===")
            tools = await load_mcp_tools(session= session)
            
            agent = create_agent(llm, tools= tools)
            
            message = HumanMessage(content= "What is 90-904?")
            
            result = await agent.ainvoke({"messages" : [message]})
            
            print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
