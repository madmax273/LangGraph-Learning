import asyncio
import streamlit as st
import os
from uuid import uuid4
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# MCP Imports
from MCP_client import MCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# Import from the refactored bot module
from agent import build_dynamic_graph
from bot1 import retrieve_All_threads
from database import DB_PATH, get_sync_checkpointer
from logger_setup import get_logger

load_dotenv()

# Setup centralized logging
logger = get_logger(__name__)

LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT")

st.title("🤖 Chatbot")

### UTILITY FUNCTIONS ###

def get_thread_id():
    return str(uuid4())

def new_chat():
    st.session_state["message_history"] = []
    st.session_state["thread_id"] = get_thread_id()
    st.session_state["threads"].append(st.session_state["thread_id"])

def load_chat(thread_id):
    try:
        # Load messages synchronously using SqliteSaver. Empty tools list is fine for reading state.
        checkpointer, conn = get_sync_checkpointer()
        chatbot_sync = build_dynamic_graph([]).compile(checkpointer=checkpointer)
        state = chatbot_sync.get_state(config={"configurable": {"thread_id": thread_id}})
        mess = state.values.get("messages", []) if hasattr(state, 'values') else []
        conn.close()
        return mess
    except Exception as e:
        logger.error(f"Error loading chat {thread_id}: {e}")
        return []

### SESSION SETUP ###

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "threads" not in st.session_state:
    try:
        st.session_state["threads"] = retrieve_All_threads()
    except Exception as e:
        logger.error(f"Error retrieving threads: {e}")
        st.session_state["threads"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = get_thread_id()
    if st.session_state["thread_id"] not in st.session_state["threads"]:
        st.session_state["threads"].append(st.session_state["thread_id"])

### SIDEBAR ###
with st.sidebar:
    st.title("Chat History")
    st.divider()
    
    if st.button("New Chat"):
        new_chat()
        st.rerun()
    
    st.divider()
    for thread_id in st.session_state["threads"]:
        if st.button(f"Chat {thread_id[:15]}...", key=f"chat_{thread_id}"):
            st.session_state["thread_id"] = thread_id
            loaded_messages = load_chat(thread_id)
            temp_messages = []
            for message in loaded_messages:
                # Differentiate between HumanMessage and AIMessage/ToolMessage
                role = 'user' if isinstance(message, HumanMessage) else 'assistant'
                # Only append if there's content to show
                if hasattr(message, 'content') and message.content:
                    temp_messages.append({"role": role, "content": message.content})
            st.session_state["message_history"] = temp_messages
            st.rerun()

### CHAT DISPLAY ###
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

### USER INPUT ###
user_input = st.chat_input("TYPE HERE...", key="chat_input")

if user_input == "exit":
    st.write("Goodbye!")
    st.stop()    

if user_input and user_input.strip():
    # Add user message to session state
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Setup configuration
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    
    # Display bot response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        async def process_stream():
            response_chunks = []
            bot_response = ""
            client = MCPClient()
            try:
                # Initialize MCP client and fetch tools dynamically
                await client.load_from_config("server_config.json")
                mcp_tools = []
                for session in client.sessions.values():
                    # Load Langchain tools from the MCP session
                    mcp_tools.extend(await load_mcp_tools(session))
                    
                # Build the dynamic graph including MCP tools
                uncompiled_graph = build_dynamic_graph(mcp_tools)
                
                # Use AsyncSqliteSaver for the async execution
                async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
                    chatbot_async = uncompiled_graph.compile(checkpointer=checkpointer)
                    
                    from langchain_core.messages import AIMessageChunk
                    # Use astream for asynchronous streaming
                    async for message_chunk, metadata in chatbot_async.astream(
                        {"messages": [HumanMessage(content=user_input)]}, 
                        config=config,
                        stream_mode="messages"
                    ):
                        if isinstance(message_chunk, AIMessageChunk):
                            if hasattr(message_chunk, 'content') and message_chunk.content:
                                if isinstance(message_chunk.content, list):
                                    for block in message_chunk.content:
                                        if isinstance(block, dict) and "text" in block:
                                            bot_response += block["text"]
                                        elif isinstance(block, str):
                                            bot_response += block
                                else:
                                    bot_response += str(message_chunk.content)
                                
                                placeholder.markdown(bot_response)
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                error_msg = f"An error occurred: {e}"
                placeholder.markdown(f"**Error:** {error_msg}")
                bot_response = error_msg
            finally:
                # Always safely close MCP connections
                await client.close()
                
            return bot_response
        
        # Run the async stream processing block
        final_bot_response = asyncio.run(process_stream())
        
    # Add bot response to session state
    if final_bot_response:
        st.session_state["message_history"].append({"role": "assistant", "content": final_bot_response})
