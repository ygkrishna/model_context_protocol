# Import required libraries
import asyncio
import os
from dotenv import load_dotenv  # For loading environment variables from .env file
from langchain_mcp_adapters.client import MultiServerMCPClient  # Used to connect to multiple MCP tool servers
from langgraph.prebuilt import create_react_agent  # A prebuilt LangGraph agent framework using ReAct pattern
from langchain_groq import ChatGroq  # Groq LLM wrapper for LangChain
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage  # For interacting with LLM agent outputs

# Load environment variables (like GROQ_API_KEY)
load_dotenv()

# ----------------------------------------
# Configure the Groq LLM model
# ----------------------------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),  # Load Groq API key
    model='meta-llama/llama-4-scout-17b-16e-instruct'  # Model selection
).with_config(
    system=(  # System prompt to guide agent behavior
        "Always call **exactly one** of the provided tools before answering, "
        "even if you think you know the answer."
    )
)

# ----------------------------------------
# Async Main Function
# ----------------------------------------
async def main():
    # Connect to a running MCP server
    client = MultiServerMCPClient(
        {
            "research": {  # Server name alias
                "transport": "streamable_http",  # Protocol type (must match server)
                "url": "http://127.0.0.1:8000/mcp"  # Endpoint for MCP server
            }
        }
    )

    # Dynamically discover all available tools from the MCP server
    tools = await client.get_tools()
    print("tools discovered:", [tool.name for tool in tools])

    # Create a ReAct agent (LLM + tool use reasoning)
    agent = create_react_agent(
        llm,      # LLM with the system message
        tools     # Tools retrieved from the MCP server
    )

    # Ask a question — this should trigger the use of one tool
    question = "Explain machine learning?"
    final = await agent.ainvoke({"messages": [HumanMessage(content=question)]})

    # ----------------------------------------
    # Optional: Inspect Tool Usage Trace
    # ----------------------------------------
    def inspect_tool_usage(trace: dict) -> None:
        """Print tool name, parameters, result, and final answer from trace."""
        if not isinstance(trace, dict) or "messages" not in trace:
            print("Unexpected trace format")
            return

        for msg in trace["messages"]:
            # Detect tool call(s) inside AI message
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    print(f"🔧 Tool requested: {call['name']}")
                    print(f"   ↳ parameters: {call['args']}")
            # Print result returned by the tool
            if isinstance(msg, ToolMessage):
                print("Tool result:")
                print(msg.content.strip())
            # Print the final LLM answer
            if isinstance(msg, AIMessage) and msg.content:
                print("Final answer:")
                print(msg.content)

    # Inspect what happened during execution
    inspect_tool_usage(final)

# Run the async main function
asyncio.run(main())
