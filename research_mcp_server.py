# Import required libraries and wrappers
from fastmcp import FastMCP  # MCP server library to expose tools
from pydantic import BaseModel  # For structured inputs (not used explicitly here)
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper  # Tool wrappers
from langchain_tavily import TavilySearch  # Real-time web search wrapper
import os, asyncio
from dotenv import load_dotenv  # To load environment variables from .env file

# Load environment variables (e.g., TAVILY_API_KEY)
load_dotenv()

# Instantiate the MCP server with a name
mcp = FastMCP("ResearchAgent")

# ----------------------------------------
# Define and register tools (functions)
# ----------------------------------------

# Setup the Arxiv wrapper to fetch top 2 results and limit content to 1500 characters
arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1500)

# Setup the Wikipedia wrapper similarly
wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1500)

# Setup Tavily wrapper for real-time web search
tavily = TavilySearch(tavily_api_key=os.getenv("TAVILY_API_KEY"))


# Register arxiv_search as a callable MCP tool
@mcp.tool()
def arxiv_search(query: str) -> str:
    """Search academic papers on arXiv using a query string."""
    return arxiv.run(query)


# Register wikipedia_search as a callable MCP tool
@mcp.tool()
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for a topic using a query string."""
    return wiki.run(query)


# Register tavily_web_search as a callable MCP tool
@mcp.tool()
def tavily_web_search(query: str) -> str:
    """Perform a real‑time web search via Tavily using a query string."""
    result = tavily.invoke(query)  # Synchronous call to Tavily
    if result.get("answer"):  # If Tavily provides a direct answer
        return result["answer"]
    if result.get("results"):  # If not, format top 3 search results
        return "\n".join(
            f"{i+1}. {item['title']} - {item['url']}\n{item['content']}"
            for i, item in enumerate(result["results"][:3])
        )
    return "No relevant information found."  # Fallback if no results


# ----------------------------------------
# Start the MCP server
# ----------------------------------------
if __name__ == "__main__":
    # Start the MCP server on localhost at port 8000 using streamable HTTP transport
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
