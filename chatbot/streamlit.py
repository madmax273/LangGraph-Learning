import streamlit as st
from bot1 import chatbot
from langchain_core.messages import HumanMessage

st.title("🤖 Chatbot")

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# Get user input with unique key
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
    config = {"configurable": {"thread_id": "1"}}
    result = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    
    # # Add bot response to session state
    # bot_response = result["messages"][-1].content
    # st.session_state["message_history"].append({"role": "assistant", "content": bot_response})
    
    # Display bot response
    with st.chat_message("assistant"):

        bot_response = st.write_stream(
            message_chunk.content 
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                config=config,
                stream_mode="messages"
            )
        )
        
    # Add bot response to session state
    st.session_state["message_history"].append({"role": "assistant", "content": bot_response})


        
