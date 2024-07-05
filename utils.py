import os
import cohere
import random
import streamlit as st
from datetime import datetime

#decorator
def enable_chat_history(func):
    if os.environ.get("COHERE_API_KEY"):

        # to clear chat history after swtching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        # to show chat history on ui
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def configure_cohere():
    cohere_api_key = st.sidebar.text_input(
        label="Cohere API Key",
        type="password",
        value=st.session_state['COHERE_API_KEY'] if 'COHERE_API_KEY' in st.session_state else '',
        placeholder="sk-..."
        )
    if cohere_api_key:
        st.session_state['COHERE_API_KEY'] = cohere_api_key
        os.environ['COHERE_API_KEY'] = cohere_api_key
    else:
        st.error("Please add your Cohere API key to continue.")
        st.info("Obtain your key from this link: https://dashboard.cohere.com/api-keys")
        st.stop()

    # model = "gpt-3.5-turbo"
