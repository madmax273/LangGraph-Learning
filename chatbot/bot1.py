from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated   
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages



load_dotenv()  

class chatstate(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat(state: chatstate) -> chatstate:
    model = ChatGroq(model="llama-3.1-8b-instant")
    prompt = PromptTemplate.from_template("You are a helpful assistant. Answer the following question: {question}")
    response = model.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(chatstate)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        print("Bot:", response["messages"][-1].content)



