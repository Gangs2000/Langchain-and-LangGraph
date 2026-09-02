# math_server.py
import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()
from tavily import TavilyClient

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b


@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    return a / b


@mcp.tool()
def search_weather(query: str) -> str:
    """Search weather using Tavily search engine"""
    search_results = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")).search(
        query=query
    )["results"]
    return "\n".join(result["content"] for result in search_results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
