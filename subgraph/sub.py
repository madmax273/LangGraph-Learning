from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("subgraph")

load_dotenv()

class SubState(TypedDict):
    english_str:str
    hindi_str:str
    

GROK_API_KEY = os.environ.get("GROK_API_KEY")

def converter(state:SubState):
    logger.info(f"--- [Subgraph: converter] Entering node with state: {state}")
    llm = ChatGroq(api_key=GROK_API_KEY, model="llama-3.1-8b-instant")
    prompt = """Convert the following text to hindi
    """
    response = llm.invoke([HumanMessage(content=f"{prompt}\n\n{state['english_str']}")])
    output = {"hindi_str": response.content}
    logger.info(f"--- [Subgraph: converter] Exiting node with updates: {output}")
    return output
    

Sub_graph = StateGraph(SubState)
Sub_graph.add_node("converter", converter)
Sub_graph.add_edge(START, "converter")
Sub_graph.add_edge("converter", END)

Sub_graph = Sub_graph.compile()