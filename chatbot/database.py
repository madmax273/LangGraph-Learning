import sqlite3
import logging
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = logging.getLogger(__name__)

DB_PATH = "chatbot.db"

def get_sync_checkpointer():
    """
    Establish a synchronous connection and return a SqliteSaver checkpointer.
    Returns (checkpointer, connection) so the connection can be closed after use.
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        return SqliteSaver(conn=conn), conn
    except sqlite3.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise

def retrieve_all_threads_sync() -> list[str]:
    """
    Retrieve all unique thread IDs using the synchronous checkpointer.
    """
    try:
        checkpointer, conn = get_sync_checkpointer()
        threads = set()
        for thread in checkpointer.list(None):
            threads.add(thread.config["configurable"]["thread_id"])
        conn.close()
        return list(threads)
    except Exception as e:
        logger.error(f"Failed to retrieve threads: {e}")
        return []

