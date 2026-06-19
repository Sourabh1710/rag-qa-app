"""
STEP 4: Conversation Memory (Modern implementation)
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

def init_memory():
    """
    Initialize a fresh conversation memory instance in Streamlit session state.
    It ensures the chat history persists correctly across Streamlit reruns.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def add_to_memory(user_query: str, ai_response: str):
    """
    Appends a new conversation turn to the running chat history.
    """
    if "chat_history" in st.session_state:
        st.session_state.chat_history.extend([
            HumanMessage(content=user_query),
            AIMessage(content=ai_response)
        ])

def clear_memory():
    """
    Clears the chat history (useful for a 'Clear Chat' UI button).
    """
    st.session_state.chat_history = []
