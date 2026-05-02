import os
import logging
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage

# Local imports
from state import ChatState
from chatbot_tools import tools

load_dotenv()

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangSmith environment variables (automatically picked up by LangChain)
LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "false")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT")

async def chat_node(state: ChatState) -> dict:
    """
    The main chat node that invokes the LLM.
    Asynchronous implementation for better performance.
    """
    try:
        # Initialize the model with tools binding
        model = ChatGroq(model="llama-3.1-8b-instant")
        model_with_tools = model.bind_tools(tools)
        
        # Asynchronously invoke the model
        response = await model_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in chat_node: {e}")
        # Fallback to prevent graph failure
        return {"messages": [AIMessage(content="I'm sorry, I encountered an error processing your request.")]}

def build_uncompiled_graph():
    """
    Build and return the uncompiled state graph.
    """
    try:
        # Create the graph
        graph = StateGraph(ChatState)
        
        # Add nodes
        graph.add_node("chat", chat_node)
        graph.add_node("tools", ToolNode(tools))
        
        # Define edges
        graph.add_edge(START, "chat")
        
        # Route based on whether a tool needs to be called
        graph.add_conditional_edges("chat", tools_condition)
        
        # After tool execution, synthesize answer
        graph.add_edge("tools", "chat")
        graph.add_edge("chat", END)
        
        return graph
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        raise

# Export the uncompiled graph for dynamic compilation
uncompiled_graph = build_uncompiled_graph()
