from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    """
    Represents the state of the chatbot.
    'messages' holds the conversation history and is updated via 'add_messages'.
    """
    messages: Annotated[list[BaseMessage], add_messages]
