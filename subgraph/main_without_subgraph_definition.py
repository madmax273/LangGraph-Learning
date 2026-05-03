from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main_graph")
from sub import Sub_graph

load_dotenv()

class MainState(TypedDict):
    msg:str
    english_str:str
    hindi_str:str
    

GROK_API_KEY = os.environ.get("GROK_API_KEY")

def node1(state:MainState):
    logger.info(f"--- [MainGraph: node1] Entering node with state: {state}")
    llm = ChatGroq(api_key=GROK_API_KEY, model="llama-3.1-8b-instant")
    prompt = """you are a assistant who help user write content on any topic in english
    """
    response = llm.invoke([HumanMessage(content=f"{prompt}\n\n{state['msg']}")])
    output = {"english_str": response.content}
    logger.info(f"--- [MainGraph: node1] Exiting node with updates: {output}")
    return output
    
def sub_node(state:MainState):
    logger.info(f"--- [MainGraph: sub_node] Entering node with state: {state}")
    res = Sub_graph.invoke({"english_str":state["english_str"]})
    output = {"hindi_str": res["hindi_str"]}
    logger.info(f"--- [MainGraph: sub_node] Exiting node with updates: {output}")
    return output
    
graph = StateGraph(MainState)
graph.add_node("node1", node1)
graph.add_node("sub_node", sub_node)
graph.add_edge(START, "node1")
graph.add_edge("node1", "sub_node")
graph.add_edge("sub_node", END)
graph = graph.compile()

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input == "exit":
            break
        res = graph.invoke({"msg": user_input})
        print("AI: " + res["hindi_str"])
