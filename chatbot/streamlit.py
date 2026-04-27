import streamlit as st
from bot1 import chatbot
from langchain_core.messages import HumanMessage
from uuid import uuid4

st.title("🤖 Chatbot")

### UTILITY FUNCTIONS ###

def get_thread_id():
    return uuid4()

def new_chat():
    st.session_state["message_history"] = []
    st.session_state["thread_id"] = get_thread_id()
    st.session_state["threads"].append(st.session_state["thread_id"])

def load_chat(thread_id):
    try:
        mess = chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values.get("messages", [])
        return mess
    except:
        return []
    

### SESSION SETUP ###

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "threads" not in st.session_state:
    st.session_state["threads"] = []    

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = get_thread_id()
    st.session_state["threads"].append(st.session_state["thread_id"])

###   SIDEBAR   ###
with st.sidebar:
    st.title("Chat History")
    st.divider()
    
    if st.button("New Chat"):
        new_chat()
        st.rerun()
    
    st.divider()
    for thread_id in st.session_state["threads"]:
        if st.button(f"Chat {thread_id}", key=f"chat_{thread_id}"):
            st.session_state["thread_id"] = thread_id
            loaded_messages = load_chat(thread_id)
            temp_messages = []
            for message in loaded_messages:
                if isinstance(message, HumanMessage):
                    role='user'
                else:
                    role='assistant'
                temp_message = {"role": role, "content": message.content}
                temp_messages.append(temp_message)
            st.session_state["message_history"] = temp_messages
            st.rerun()

###   CHAT DISPLAY   ###
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

###   USER INPUT   ###
user_input = st.chat_input("TYPE HERE...", key="chat_input")

if user_input=="exit":
    st.write("Goodbye!")
    exit()    

if user_input and user_input.strip():  # Check if input exists and is not empty
    # Add user message to session state
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.text(user_input)
    
    # Get bot response
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    
    # Display bot response
    with st.chat_message("assistant"):
        # Collect streamed response
        response_chunks = []
        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]}, 
            config=config,
            stream_mode="messages"
        ):
            if hasattr(message_chunk, 'content') and message_chunk.content:
                response_chunks.append(message_chunk.content)
        
        bot_response = ''.join(response_chunks)
        st.write(bot_response)
        
    # Add bot response to session state
    st.session_state["message_history"].append({"role": "assistant", "content": bot_response})


        
