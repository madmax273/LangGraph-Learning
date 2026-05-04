import asyncio
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# MCP Imports
from MCP_client import MCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# Local modular imports
from agent import build_dynamic_graph
from database import retrieve_all_threads_sync, DB_PATH
from logger_setup import get_logger

# Alias for backward compatibility with streamlit.py
retrieve_All_threads = retrieve_all_threads_sync

logger = get_logger(__name__)

async def run_cli():
    """
    Run the asynchronous chatbot CLI.
    """
    print("Welcome to the AI Chatbot! Type 'exit' to quit.")
    config = {"configurable": {"thread_id": "2"}}

    client = MCPClient()
    try:
        # Load tools dynamically from MCP servers
        print("Connecting to MCP Servers...")
        await client.load_from_config("server_config.json")
        mcp_tools = []
        for session in client.sessions.values():
            mcp_tools.extend(await load_mcp_tools(session))
            
        print(f"Loaded {len(mcp_tools)} external tools.")
        
        # Build the dynamic graph
        uncompiled_graph = build_dynamic_graph(mcp_tools)

        async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
            chatbot = uncompiled_graph.compile(checkpointer=checkpointer)

            while True:
                try:
                    user_input = input("You: ")
                    if user_input.lower() == "exit":
                        print("Goodbye!")
                        break
                    
                    if not user_input.strip():
                        continue

                    # Asynchronously invoke the chatbot
                    response = await chatbot.ainvoke(
                        {"messages": [HumanMessage(content=user_input)]}, 
                        config=config
                    )

                    # The response contains the state; get the last message content
                    bot_response_raw = response["messages"][-1].content
                    if isinstance(bot_response_raw, list):
                        bot_response = ""
                        for block in bot_response_raw:
                            if isinstance(block, dict) and "text" in block:
                                bot_response += block["text"]
                            elif isinstance(block, str):
                                bot_response += block
                    else:
                        bot_response = str(bot_response_raw)
                        
                    print(f"Bot: {bot_response}")
                    
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    logger.error(f"An error occurred during interaction: {e}")
                    print("Bot: Sorry, something went wrong. Please try again.")
    except Exception as e:
        logger.error(f"Failed to start CLI: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    # Run the async CLI loop
    try:
        asyncio.run(run_cli())
    except Exception as e:
        logger.critical(f"Fatal error running CLI: {e}")


