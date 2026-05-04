import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage

# Local imports
from state import ChatState
from chatbot_tools import base_tools
from logger_setup import get_logger

load_dotenv()

# Setup centralized logging
logger = get_logger(__name__)

# LangSmith environment variables (automatically picked up by LangChain)
LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "false")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT")

def build_dynamic_graph(mcp_tools=None):
    """
    Build and return the uncompiled state graph, integrating standard tools and dynamic MCP tools.
    """
    all_tools = list(base_tools) + (mcp_tools or [])
    
    async def chat_node(state: ChatState) -> dict:
        """
        The main chat node that invokes the LLM with all available tools.
        """
        try:
            # Initialize the model with tools binding
            model = ChatGroq(model="llama-3.3-70b-versatile")
            model_with_tools = model.bind_tools(all_tools)
            
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an advanced AI assistant. You have access to external tools via MCP. "
                           "CRITICAL: When calling tools (especially for appending/writing to files), you MUST perfectly format your tool calls as valid JSON. "
                           "Ensure all newlines are properly escaped as \\n, and double quotes are escaped as \\\" inside strings. "
                           "Never output raw multiline strings inside JSON values."),
                MessagesPlaceholder(variable_name="messages"),
            ])
            
            chain = prompt | model_with_tools
            
            # Asynchronously invoke the model
            response = await chain.ainvoke({"messages": state["messages"]})
            return {"messages": [response]}
        except Exception as e:
            error_details = str(e)
            if hasattr(e, "failed_generation"):
                error_details += f"\nFailed generation text: {e.failed_generation}"
            logger.error(f"Error in chat_node: {error_details}")
            # Fallback to prevent graph failure
            return {"messages": [AIMessage(content=f"I'm sorry, I encountered an error processing your request: {e}")]}

    try:
        # Create the graph
        graph = StateGraph(ChatState)
        
        # Add nodes
        graph.add_node("chat", chat_node)
        graph.add_node("tools", ToolNode(all_tools))
        
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

