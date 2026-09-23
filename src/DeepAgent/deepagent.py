import asyncio
import os
import sys

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


async def initialize_mcp_connection():
    mcp_tools_path = r"C:\\Users\\GBS09645\\OneDrive - Sella\\Desktop\\LangChain and LangGraph\\mcp-servers\\servers\\mcp_server.py"
    mcp_server_connection = StdioServerParameters(
        command="python", args=[mcp_tools_path], env=os.environ.copy()
    )
    return mcp_server_connection


async def agent_preparation(tools, backend):
    return create_deep_agent(
        model=llm,
        tools=tools,
        backend=backend,
        skills=["/Skills/"],
        checkpointer=MemorySaver(),
        interrupt_on={
            "search_weather": {"allowed_decisions": ["approve", "reject", "edit"]}
        },
    )


async def human_in_the_loop(agent, result, config):
    while "__interrupt__" in result:
        interrupt = result["__interrupt__"][0]
        print(interrupt.value)
        decision = (
            input("Select one of the action a -> Approve, r -> Reject, e -> Edit:")
            .strip()
            .lower()
        )
        if decision == "a":
            # Approve case
            print("Tool execution approved")
            result = await agent.ainvoke(
                Command(resume={"decisions": [{"type": "approve"}]}),
                config=config,
            )
        elif decision == "r":
            # Rejection case
            print("Tool execution rejected by user.")
            return
        elif decision == "e":
            # Edition query case
            print("Editing argument with new input")
            new_query = input(
                "Please provide input to resume search weather operation : "
            ).strip()
            action = interrupt.value["action_requests"][0]
            result = await agent.ainvoke(
                Command(
                    resume={
                        "decisions": [
                            {
                                "type": "edit",
                                "message": f"User edited the action, now your goal is to execute tool and obtain final answer with the new input '{new_query}'.",
                                "edited_action": {
                                    "name": action["name"],
                                    "args": {"query": new_query},
                                },
                            }
                        ]
                    }
                ),
                config=config,
            )

    print(result["messages"][-1].content)


async def normal_deep_agent():
    server_connection = await initialize_mcp_connection()
    async with stdio_client(server_connection) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("=== Session Initialized with MCP Server ===")

            backend = FilesystemBackend(
                root_dir=PATH_CORRECTOR,
                virtual_mode=True,
            )

            tools = await load_mcp_tools(session)

            agent = await agent_preparation(tools, backend)

            print("=== Deep Agent Created ===")

            config = {"configurable": {"thread_id": "weather-deep-agents-demo"}}

            result = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(content="What is current weather in madurai"),
                    ]
                },
                config=config,
            )

            await human_in_the_loop(agent=agent, result=result, config=config)


async def streamable_deep_agent():
    server_connection = await initialize_mcp_connection()
    async with stdio_client(server_connection) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("=== Session Initialized with MCP Server ===")

            backend = FilesystemBackend(
                root_dir=PATH_CORRECTOR,
                virtual_mode=True,
            )

            tools = await load_mcp_tools(session)
            agent = await agent_preparation(tools, backend)

            config = {"configurable": {"thread_id": "weather-deep-agents-demo"}}

            stream = await agent.astream_events(
                {
                    "messages": [
                        HumanMessage(content="What is current weather in madurai")
                    ]
                },
                config=config,
                version="v3",
            )

            async def consume_messages(target_stream):
                async for message in target_stream.messages:
                    async for token in message.text:
                        print(token, end="", flush=True)

            async def consume_tool_calls(target_stream):
                async for call in target_stream.tool_calls:
                    print("\nTool:", call.tool_name, call.input)
                    async for delta in call.output_deltas:
                        print(delta, end="", flush=True)
                    if call.completed and call.error is None:
                        print("\nResult:", call.output)
                    elif call.error is not None:
                        print("\nError:", call.error)

            await asyncio.gather(consume_messages(stream), consume_tool_calls(stream))

            while stream._interrupted:
                print("\n\n[Interrupt]:", stream._interrupts)
                decision = input("Approve tool execution? [y/N]: ").strip().lower()
                resume_decision = (
                    {"type": "approve"} if decision == "y" else {"type": "reject"}
                )

                stream = await agent.astream_events(
                    Command(resume={"decisions": [resume_decision]}),
                    config=config,
                    version="v3",
                )

                await asyncio.gather(
                    consume_messages(stream), consume_tool_calls(stream)
                )


if __name__ == "__main__":
    print("\nTwo types of agents available ( Deep agent and Streamable Deep agent ) :")
    agent_type = int(
        input(
            "\n Please choose the type of agent you want to use. 1. Deep agent 2. Streamable Deep agent : [1/2]? "
        ).strip()
    )
    if agent_type == 1:
        asyncio.run(normal_deep_agent())
    elif agent_type == 2:
        asyncio.run(streamable_deep_agent())
