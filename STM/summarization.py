from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.messages import trim_messages
from langgraph.graph import StateGraph,END,START
from langgraph.graph.message import add_messages,RemoveMessage   
from typing import Annotated,TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
import os 
from langgraph.checkpoint.sqlite import SqliteSaver    
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

import sqlite3
conn = sqlite3.connect("chat.db", check_same_thread=False)
sqlite_saver=SqliteSaver(conn)


class GraphState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    summary:str


#node definition
def chat_node(state:GraphState):
    llm=ChatGroq(api_key=GROQ_API_KEY,model="llama-3.1-8b-instant")
    
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    print("\ncurrent message count", len(state["messages"]), "\n")
    response=llm.invoke(messages)

    

    if state.get("summary"):
        print("\nSummary: ", state["summary"], "\n")
    for message in messages:
        if isinstance(message,HumanMessage):
            print("--> User: ",message.content) 
        elif isinstance(message,AIMessage):
            print("--> AI: ",message.content)
        elif isinstance(message,SystemMessage):
            print("--> System: ", message.content)

    return {"messages":[response]}


def summary_node(state:GraphState):
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of conversatiyes you can on to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    
    llm=ChatGroq(api_key=GROQ_API_KEY,model="llama-3.1-8b-instant")
    response = llm.invoke(messages)
    
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-4]]
    
    return {"summary": response.content, "messages": delete_messages}


def should_summarize(state: GraphState):
    messages = state["messages"]
    print("Length of messages is:",len(messages),"\n")
    if len(messages) > 6:
        return "summarize"
    return END


checkpointer=sqlite_saver
graph_builder=StateGraph(GraphState)
graph_builder.add_node("chat",chat_node)
graph_builder.add_node("summarize", summary_node)

graph_builder.add_edge(START, "chat")
graph_builder.add_conditional_edges("chat", should_summarize)
graph_builder.add_edge("summarize", END)

app=graph_builder.compile(checkpointer=checkpointer)

config={"configurable":{"thread_id":"thread_1"}}

if __name__== "__main__":
    print("\n-------------------------")
    print("Starting the chatbot...\n")
    while True:
        user_input=input("User: ")
        if user_input.lower()=="exit":
            break
        res=app.invoke({"messages":[HumanMessage(content=user_input)]},config=config)
        print("AI: ",res["messages"][-1].content)