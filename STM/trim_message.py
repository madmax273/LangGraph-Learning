from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.messages import trim_messages
from langgraph.graph import StateGraph,END,START
from langgraph.graph.message import add_messages    
from typing import Annotated,TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
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
    
    
    messages=trim_messages(
        state["messages"],
        max_tokens=100,
        strategy="last",
        token_counter=count_tokens_approximately,
        )

    print("messages",messages,"\n")     

    
    print("\ncurrent token count",count_tokens_approximately(messages=messages),"\n")
    response=llm.invoke(messages)
    return {"messages":[response]}




checkpointer=sqlite_saver
graph_builder=StateGraph(GraphState)
graph_builder.add_node("chat",chat_node)
graph_builder.add_edge(START, "chat")
graph_builder.add_edge("chat", END)
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