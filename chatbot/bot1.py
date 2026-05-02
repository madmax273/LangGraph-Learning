import asyncio
import logging
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Local modular imports
from agent import uncompiled_graph
from database import retrieve_all_threads_sync, DB_PATH

# Alias for backward compatibility with streamlit.py
retrieve_All_threads = retrieve_all_threads_sync

logger = logging.getLogger(__name__)

async def run_cli():
    """
    Run the asynchronous chatbot CLI.
    """
    print("Welcome to the AI Chatbot! Type 'exit' to quit.")
    config = {"configurable": {"thread_id": "2"}}

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
                bot_response = response["messages"][-1].content
                print(f"Bot: {bot_response}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"An error occurred during interaction: {e}")
                print("Bot: Sorry, something went wrong. Please try again.")

if __name__ == "__main__":
    # Run the async CLI loop
    try:
        asyncio.run(run_cli())
    except Exception as e:
        logger.critical(f"Fatal error running CLI: {e}")

