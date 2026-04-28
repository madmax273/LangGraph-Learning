from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated   
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
import sqlite3
import os
from chatbot_tools import tools
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()  

LANGSMITH_TRACING = "true"
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT")

class chatstate(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat(state: chatstate) -> chatstate:
    model = ChatGroq(model="llama-3.1-8b-instant")
    model_with_tools = model.bind_tools(tools)
    prompt = PromptTemplate.from_template("You are a helpful assistant. Answer the following question: {question}")
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

graph = StateGraph(chatstate)
graph.add_node("chat", chat)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")
graph.add_edge("chat", END)

sqlite_conn = sqlite3.connect("chatbot.db",check_same_thread=False)

checkpointer = SqliteSaver(conn=sqlite_conn)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_All_threads():
    threads=set()
    for thread in checkpointer.list(None):
        threads.add(thread.config["configurable"]["thread_id"])
    return list(threads)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "2"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        print("Bot:", response["messages"][-1].content)


