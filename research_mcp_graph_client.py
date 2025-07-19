# research_client_stategraph.py

# Standard library imports
import asyncio, os

# For loading environment variables
from dotenv import load_dotenv

# For defining strongly typed state structures
from typing_extensions import TypedDict
from typing import Annotated

# LangChain message types
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage

# FastMCP client to connect to MCP servers
from langchain_mcp_adapters.client import MultiServerMCPClient

# Groq LLM (you could also plug in OpenAI, Anthropic, etc.)
from langchain_groq import ChatGroq

# LangGraph core imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# ------------------------------------------------------------------
# 1) Load environment and set up the LLM
# ------------------------------------------------------------------
load_dotenv()

# Initialize Groq LLM and enforce use of tools
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="meta-llama/llama-4-scout-17b-16e-instruct",
).with_config(
    system="If you are not absolutely sure, always call exactly one tool to assist your answer."
)

# ------------------------------------------------------------------
# 2) Connect to MCP server and fetch tools
# ------------------------------------------------------------------
async def build_graph_and_run(question: str):
    client = MultiServerMCPClient(
        {
            "research": {
                "transport": "streamable_http",  # Use streamable-http protocol
                "url": "http://127.0.0.1:8000/mcp",  # MCP server running locally
            }
        }
    )

    # Fetch the list of tools exposed by MCP
    tools = await client.get_tools()
    print("Tools discovered:", [t.name for t in tools])

    # Bind tools to the LLM so it can use them during reasoning
    llm_with_tools = llm.bind_tools(tools)

    # ------------------------------------------------------------------
    # 3) Define LangGraph state and behavior
    # ------------------------------------------------------------------

    # Define the state type; state carries message history
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]  # Tracks LLM, tool messages, etc.

    # Node: Calls LLM (which may or may not invoke a tool)
    def tool_calling_llm(state: State) -> State:
        result = llm_with_tools.invoke(state["messages"])  # Call LLM with current message history
        print("LLM Response:", result.content or "[tool call]")  # Print LLM output
        return {"messages": [result]}  # Return updated message state

    # ------------------------------------------------------------------
    # 4) Define the LangGraph workflow
    # ------------------------------------------------------------------
    sg = StateGraph(State)

    # Add LLM reasoning node
    sg.add_node("tool_calling_llm", tool_calling_llm)

    # Add tool execution node (executes whatever tool LLM chooses)
    sg.add_node("tools", ToolNode(tools))

    # Define edges between nodes
    sg.add_edge(START, "tool_calling_llm")  # Start → LLM
    sg.add_conditional_edges("tool_calling_llm", tools_condition)  # If LLM calls tool → tools
    sg.add_edge("tools", "tool_calling_llm")  # After tool is run → go back to LLM
    sg.add_edge("tool_calling_llm", END)  # If no tool called → end

    # Compile the graph into an executable workflow
    graph = sg.compile()

    # ------------------------------------------------------------------
    # 5) (Optional) Visualize LangGraph state flow as a Mermaid diagram
    # ------------------------------------------------------------------
    try:
        from IPython.display import Image, display
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 6) Run the graph by sending the question as a human message
    # ------------------------------------------------------------------
    trace = await graph.ainvoke({"messages": [HumanMessage(content=question)]})

    # ------------------------------------------------------------------
    # 7) Inspect the full trace of execution (tool calls and responses)
    # ------------------------------------------------------------------
    tool_seen = False
    for msg in trace["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_seen = True
            for call in msg.tool_calls:
                print(f"\n Tool requested: {call['name']}")
                print(f"   ↳ parameters: {call['args']}")
        elif isinstance(msg, ToolMessage):
            print("\n Tool result:")
            print(msg.content.strip())
        elif isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content

    if not tool_seen:
        print("\nNo tool was used.")
    print("\nFinal answer:\n", final_answer)

# ------------------------------------------------------------------
# 8) Run the async function with the research question
# ------------------------------------------------------------------
if __name__ == "__main__":
    QUESTION = "Use any tool available to Write a research summary on quantum computing trends in 2024"
    asyncio.run(build_graph_and_run(QUESTION))
