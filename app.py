import streamlit as st
import time
from llm.llm_factory import get_llm
from rag.rag_service import RAGService

st.set_page_config(
    page_title="Hacky Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 My Chatbot")
llm = get_llm()
@st.cache_resource
def get_rag():
    return RAGService("data/Optimize_your_console_and_PC_game_performance_Unity_2022_LTS.pdf")

rag = get_rag()
# Sidebar
with st.sidebar:
    st.header("Options")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask something...")

if prompt:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    response = rag.ask(prompt)
    
    # Display typing effect
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for word in response.split():
            full_response += word + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })