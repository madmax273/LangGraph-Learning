import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv


from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()

GROK_API_KEY = os.environ.get("GROK_API_KEY")
# Define the State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: State) -> State:
    """
    This node is responsible for the chatbot's response generation.
    """
    messages = state["messages"]
    
    # Initialize your LLM here. Update the API key/model as needed.
    # Note: Using your preferred model (e.g., grok-4, gpt-4o, etc.)
    llm = ChatGroq(api_key=GROK_API_KEY, model="llama-3.1-8b-instant") 
    
    response = llm.invoke(messages)
    return {"messages": [response]}

def hitl_node(state: State) -> State:
    """
    This node pauses the graph execution to ask the human for feedback.
    """
    # The interrupt function halts the graph and surfaces this prompt to the user
    user_feedback = interrupt("Please review the AI's response. Provide feedback or type 'approve' to finish.")
    
    # If the user provides feedback (instead of just approving), 
    # we add it as a new HumanMessage to the state.
    if user_feedback.lower().strip() != "approve":
        return {"messages": [HumanMessage(content=user_feedback)]}
    
    # If approved, we don't add anything to the state.
    return {"messages": []}

def route_after_hitl(state: State) -> str:
    """
    Decide whether to loop back to the chat_node or end the execution.
    If the last message is from a human (meaning they provided feedback), we go back to 'chat'.
    If it's from the AI (meaning they typed 'approve' and no human message was added), we end.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        return "chat"
    return END

def build_graph():
    """
    Builds and returns the LangGraph application.
    """
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("chat", chat_node)
    workflow.add_node("hitl", hitl_node)
    
    # Add edges
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", "hitl")
    
    # Add conditional edge after HITL
    workflow.add_conditional_edges("hitl", route_after_hitl)
    
    # A checkpointer is REQUIRED for Human-in-the-Loop to save state
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    return app

def main():
    app = build_graph()
    
    # We must provide a thread ID so LangGraph can keep track of the state memory
    config = {"configurable": {"thread_id": "thread-1"}}
    
    # 1. Start the conversation
    input_text=input("You: ")
    initial_input = {"messages": [HumanMessage(content=input_text)]}
    print("--- User Request ---")
    print(initial_input["messages"][0].content)
    print("\n--- Running Graph ---")
    
    res = app.invoke(initial_input, config)

    print("============================res==================================:", res)
    
    # We must look at the state from the graph rather than just the final output of invoke, 
    # because invoke returns the state at the point of interruption.
    print("AI: ", res["messages"][-1].content)
    
    # 2. Loop as long as the graph is interrupted
    while True:
        state = app.get_state(config)
        
        # Check if there are pending tasks (which means we hit an interrupt)
        if not state.next:
            break
            
        print("state:", state)
        print("state.next", state.next)
        print("==================================================================")
        # The value passed to `interrupt()` is available in the state tasks
        interrupt_value = state.tasks[0].interrupts[0].value
        print(f"--- Graph Paused ---")
        
        # 3. Get human input
        user_input = input(f"{interrupt_value}\nYour feedback: ")
        
        # 4. Resume the graph by passing the human input using Command
        print("\n--- Resuming Graph ---")
        # We use stream here so we can see the chat node output instantly
        for event in app.stream(Command(resume=user_input), config, stream_mode="updates"):
             for node_name, node_state in event.items():
                if node_name == "chat":
                    print(f"AI: {node_state['messages'][-1].content}\n")
    
    print("--- Graph Execution Complete ---")

if __name__ == "__main__":
    main()
