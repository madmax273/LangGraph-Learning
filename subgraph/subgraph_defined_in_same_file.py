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

load_dotenv()

class MainState(TypedDict):
    msg:str
    english_str:str
    hindi_str:str
    

GROK_API_KEY = os.environ.get("GROK_API_KEY")

def translator_node(state:MainState):
    logger.info(f"--- [MainGraph: translator_node] Entering node with state: {state}")
    prompt = """you are a assistant who help converts the content from english to hindi
    """
    response = sub_llm.invoke([HumanMessage(content=f"{prompt}\n\n{state['english_str']}")])
    return {"hindi_str": response.content}

sub_llm=ChatGroq(api_key=GROK_API_KEY, model="llama-3.1-8b-instant")
sub_graph=StateGraph(MainState)
sub_graph.add_node("translator_node", translator_node)
sub_graph.add_edge(START, "translator_node")
sub_graph.add_edge("translator_node", END)
sub_graph = sub_graph.compile()



def main_node(state:MainState):
    logger.info(f"--- [MainGraph: main_node] Entering node with state: {state}")
    llm=ChatGroq(api_key=GROK_API_KEY, model="llama-3.1-8b-instant")
    prompt = """you are a assistant who help user write content on any topic in english
    """
    response = llm.invoke([HumanMessage(content=f"{prompt}\n\n{state['msg']}")])
    output = {"english_str": response.content}
    logger.info(f"--- [MainGraph: main_node] Exiting node with updates: {output}")
    return output
    
main_graph=StateGraph(MainState)
main_graph.add_node("main_node", main_node)
main_graph.add_node("translate",sub_graph)
main_graph.add_edge(START, "main_node")
main_graph.add_edge("main_node", "translate")
main_graph.add_edge("translate", END)
main_graph = main_graph.compile()
    

    


if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input == "exit":
            break
        res = main_graph.invoke({"msg": user_input})
        print("AI: " + res["hindi_str"])
