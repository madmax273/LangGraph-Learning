from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()  

class jokestate(TypedDict):
    topic: str
    joke: str
    explanation: str

model = ChatGroq(model="llama-3.1-8b-instant")

def create_joke(state: jokestate) -> jokestate:
    prompt=f"Create a joke about {state['topic']}"
    response = model.invoke(prompt)
    return {"joke": response.content}

def explain_joke(state: jokestate) -> jokestate:
    prompt=f"Explain this joke: {state['joke']}"
    response = model.invoke(prompt)
    return {"explanation": response.content}

graph=StateGraph(jokestate) 

graph.add_node("create_joke", create_joke)
graph.add_node("explain_joke", explain_joke)

graph.add_edge(START, "create_joke")
graph.add_edge("create_joke", "explain_joke")
graph.add_edge("explain_joke", END)

checkpointer=InMemorySaver()

chatbot= graph.compile(checkpointer=checkpointer)


config1={"configurable": {"thread_id": "1"}}
config2={"configurable": {"thread_id": "2"}}
# response=chatbot.invoke({"topic": "cat"}, config=config2)

# print(response)
print(chatbot.get_state(config2))
print(list(chatbot.get_state_history(config2)))