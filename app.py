import streamlit as st
import time
import os
import hashlib

from rag.rag_service import RAGService

st.set_page_config(
    page_title="Hacky Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 My Chatbot")

# ----------------------------
# Session State
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------
# Sidebar
# ----------------------------

with st.sidebar:

    st.header("Knowledge Source")

    uploaded_pdf = st.file_uploader(
        "Browse PDF",
        type=["pdf"]
    )

    folder_path = st.text_input("Or enter folder path)",
        placeholder="example: data/docs/"
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ----------------------------
# Determine Active Source
# ----------------------------

active_source = None
source_hash = None

# ---------------- PDF ----------------

if uploaded_pdf:

    os.makedirs("uploads", exist_ok=True)

    file_bytes = uploaded_pdf.getbuffer()

    source_hash = hashlib.md5(file_bytes).hexdigest()

    active_source = f"uploads/{source_hash}.pdf"

    if not os.path.exists(active_source):

        with open(active_source, "wb") as f:
            f.write(file_bytes)

# ---------------- Folder (HTML docs) ----------------

elif folder_path:

    if os.path.exists(folder_path):

        active_source = folder_path

        hash_builder = hashlib.md5()

        html_count = 0

        for root, _, files in os.walk(folder_path):

            for file in files:

                if file.endswith((".html", ".htm")):

                    html_count += 1

                    full_path = os.path.join(root, file)

                    hash_builder.update(full_path.encode())

        source_hash = hash_builder.hexdigest()

        st.sidebar.caption(f"HTML files detected: {html_count}")

    else:
        st.sidebar.error("Folder does not exist")

# ---------------- Default ----------------

else:

    active_source = "data/chm_doc/PowerCollections"
    source_hash = "default"


# ----------------------------
# Load RAG
# ----------------------------

@st.cache_resource(show_spinner=False)
def load_rag(source, source_hash):
    return RAGService(source, source_hash)

with st.spinner(f"📚 Indexing knowledge source: {active_source}"):
    rag = load_rag(active_source, source_hash)

# ----------------------------
# Display Knowledge Source
# ----------------------------

st.caption(f"📚 Knowledge Source: `{active_source}`")


# ----------------------------
# Chat History
# ----------------------------

for message in st.session_state.messages:
    avatar = "assets/user.png" if message["role"] == "user" else "assets/assistant.png"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# ----------------------------
# Chat Input
# ----------------------------

prompt = st.chat_input("Ask something...")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user", avatar="assets/user.png"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="assets/assistant.png"):
        with st.spinner("Thinking..."):
            response, sources = rag.ask(prompt)

        placeholder = st.empty()
        full_response = ""

        for word in response.split():

            full_response += word + " "
            time.sleep(0.04)

            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    if sources:

        st.markdown("---")
        st.markdown("**📚 Sources**")

        for src in set(sources):
            st.markdown(f"- `{src}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })