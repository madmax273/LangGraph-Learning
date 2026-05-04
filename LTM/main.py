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
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from typing import List
import uuid

load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

import sqlite3
conn = sqlite3.connect("chat.db", check_same_thread=False)
sqlite_saver=SqliteSaver(conn)

class MemoryItem(BaseModel):
    text:str
    is_new:bool

class MemoryDecision(BaseModel):
    should_write:bool
    memories: List[MemoryItem] = Field(default_factory=list)

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    summary:str



# ----------------------------
# 2) System prompt template (your prompt)
# ----------------------------
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.
If user-specific memory is available, use it to personalize 
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailored 
assistance that reflects the user’s preferences, context, and past interactions.

If the user’s name or relevant personal context is available, always personalize your responses by:
    – Always Address the user by name (e.g., "Sure, Punit...") when appropriate
    – Referencing known projects, tools, or preferences (e.g., "your MCP  server python based project")
    – Adjusting the tone to feel friendly, natural, and directly aimed at the user

Avoid generic phrasing when personalization is possible. For example, instead of "In TypeScript apps..." 
say "Since your project is built with TypeScript..."

Use personalization especially in:
    – Greetings and transitions
    – Help or guidance tailored to tools and frameworks the user uses
    – Follow-up messages that continue from past context

Always ensure that personalization is based only on known user details and not assumed.

In the end suggest 3 relevant further questions based on the current response and user profile

The user’s memory (which may be empty) is provided as: {user_details_content}
The user's summary is provided as: {summary_content}
"""






MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term (identity, stable preferences, ongoing projects/goals).
- For each extracted item, set is_new=true ONLY if it adds NEW information compared to CURRENT USER DETAILS.
- If it is basically the same meaning as something already present, set is_new=false.
- Keep each memory as a short atomic sentence.
- No speculation; only facts stated by the user.
- If there is nothing memory-worthy, return should_write=false and an empty list.
"""



llm=ChatGroq(api_key=GROQ_API_KEY,model="llama-3.3-70b-versatile")
structured_llm=llm.with_structured_output(MemoryDecision)


#node definition
def chat_node(state:GraphState, config,store:BaseStore):
    llm=ChatGroq(api_key=GROQ_API_KEY,model="llama-3.3-70b-versatile")
    
    user_id=config["configurable"]["thread_id"]
    name_space=("user",user_id,"details")
    
    items=store.search(name_space)
    user_details = "\n".join(it.value["data"] for it in items) if items else ""

    summary = state.get("summary", "")
    if summary:
        summary_message = f"Summary of conversation earlier: {summary}"

    if user_details:
        ltm_message =(SystemMessage(content=f"User details: {user_details}"))

    system_message = SYSTEM_PROMPT_TEMPLATE.format(user_details_content=user_details, summary_content=summary)
    
    messages=[SystemMessage(content=system_message)] + state["messages"]


    response=llm.invoke(messages)

    return {"messages":[response]}

def remember_node(state:GraphState, config, store:BaseStore):
    user_id=config["configurable"]["thread_id"]
    name_space=("user",user_id,"details")
    
    items=store.search(name_space)
    user_details = "\n".join(it.value["data"] for it in items) if items else ""
    
    last_message = state["messages"][-1]

    decision: MemoryDecision = structured_llm.invoke([
        SystemMessage(content=MEMORY_PROMPT.format(user_details_content=user_details or "empty")),
        HumanMessage(content=last_message.content)
    ])
    
    if decision.should_write:
        for item in decision.memories:
            if item.is_new:
                store.put(name_space, str(uuid.uuid4()), {"data": item.text})

    return {}            
    


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

    if len(messages) > 6:
        return "summarize"
    return END


checkpointer=sqlite_saver
graph_builder=StateGraph(GraphState)

graph_builder.add_node("chat",chat_node)
graph_builder.add_node("summarize", summary_node)
graph_builder.add_node("remember", remember_node)

graph_builder.add_edge(START, "remember")
graph_builder.add_edge("remember", "chat")
graph_builder.add_conditional_edges("chat", should_summarize)
graph_builder.add_edge("summarize", END)

# Create store before compiling the graph
store = InMemoryStore()

# Compile with store
app=graph_builder.compile(checkpointer=checkpointer, store=store)

user_id="user_1"
config={"configurable":{"thread_id":user_id}}

if __name__== "__main__":
    print("\n-------------------------")
    print("Starting the chatbot...\n")

    while True:
        user_input=input("User: ")
        if user_input.lower()=="exit":
            break
        res=app.invoke({"messages":[HumanMessage(content=user_input)]},config=config, store=store)
        print("AI: ",res["messages"][-1].content)