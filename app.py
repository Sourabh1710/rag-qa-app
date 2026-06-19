"""
STEP 5: Streamlit Chat Interface
"""

import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag.document_processor import process_uploaded_files
from rag.llm_chain import create_llm, create_rag_chain
from rag.vector_store import create_vector_store


# PAGE CONFIGURATION
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# SESSION STATE INITIALIZATION
def init_session_state():
    defaults = {
        "messages": [],           # list of {role, content, sources} for UI display
        "chat_history": [],       # list of HumanMessage/AIMessage objects for LangChain
        "rag_chain": None,        # the active modern retrieval chain
        "docs_loaded": False,     # True after successful document processing
        "doc_stats": {},          # {files, chunks, filenames} for display
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Warm up the embedding model at app start
with st.spinner("⚙️ Loading embedding model…"):
    from rag.vector_store import get_embedding_model
    get_embedding_model()   # populates the cache; subsequent calls are instant



# SIDEBAR - Configuration & Document Upload
with st.sidebar:
    st.title("📚 RAG Document Q&A")
    st.caption("LangChain · FAISS · HuggingFace")
    st.divider()

    # LLM Provider
    st.subheader("🤖 LLM Provider")

    provider_display = st.selectbox(
        "Choose provider",
        ["Gemini (Google) — recommended", "Groq (Meta Llama 3)"],
        help="Both are FREE. Gemini: aistudio.google.com | Groq: console.groq.com",
    )
    provider = "Gemini" if "Gemini" in provider_display else "Groq"

    # API key input - password type hides the key
    api_key = st.text_input(
        f"{provider} API Key",
        type="password",
        help="Your key is never stored — it lives only in this session.",
    )

    # Fall back to environment variable if set (useful for local dev with .env)
    if not api_key:
        env_var = "GOOGLE_API_KEY" if provider == "Gemini" else "GROQ_API_KEY"
        api_key = os.getenv(env_var, "")

    if not api_key:
        st.info(
            f"Get a free {'Gemini' if provider == 'Gemini' else 'Groq'} API key:\n\n"
            f"{'-> https://aistudio.google.com/' if provider == 'Gemini' else '-> https://console.groq.com/'}"
        )

    st.divider()

    # File Upload
    st.subheader("📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Financial reports, research papers, legal docs all work great.",
    )

    # Advanced chunking settings (collapsed by default)
    with st.expander("⚙️ Advanced settings"):
        chunk_size = st.slider(
            "Chunk size (chars)", 200, 1500, 800, 50,
            help="Larger = more context per chunk, less retrieval precision.",
        )
        chunk_overlap = st.slider(
            "Chunk overlap (chars)", 0, 300, 100, 25,
            help="Overlap prevents context loss at chunk boundaries.",
        )
        top_k = st.slider(
            "Chunks to retrieve (K)", 1, 8, 3,
            help="How many chunks to send to the LLM per question.",
        )

    # Process Button
    can_process = bool(uploaded_files and api_key)
    process_clicked = st.button(
        "🚀 Process Documents",
        disabled=not can_process,
        use_container_width=True,
        type="primary",
    )

    if not can_process and not st.session_state.docs_loaded:
        if not uploaded_files:
            st.caption("↑ Upload at least one file")
        elif not api_key:
            st.caption("↑ Enter your API key above")

    # Document Processing Logic
    if process_clicked:
        with st.spinner(""):
            try:
                # Progress bar gives visual feedback for the slow embedding step
                progress = st.progress(0, "Extracting text from documents...")

                # STEP 1: Extract + chunk
                chunks = process_uploaded_files(
                    uploaded_files,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                if not chunks:
                    st.error("No text extracted. Check that your files contain readable text.")
                    st.stop()
                progress.progress(30, f" {len(chunks)} chunks extracted")

                # STEP 2: Embed + build FAISS index
                progress.progress(40, "Generating embeddings - downloading model if needed (~90MB)...")
                vector_store = create_vector_store(chunks)
                progress.progress(80, " Vector index built")

                # STEP 3: Initialize LLM + Modern RAG chain (REMOVED legacy memory)
                progress.progress(88, f"Connecting to {provider}...")
                llm = create_llm(api_key, provider)
                st.session_state.rag_chain = create_rag_chain(
                    vector_store, llm, k=top_k
                )
                progress.progress(100, " Ready!")
                time.sleep(0.4)
                progress.empty()

                # Save stats and reset chat history for the new document set
                st.session_state.docs_loaded = True
                st.session_state.doc_stats = {
                    "files": len(uploaded_files),
                    "chunks": len(chunks),
                    "filenames": [f.name for f in uploaded_files],
                    "provider": provider,
                    "k": top_k,
                }
                st.session_state.messages = []      # Clear UI conversation
                st.session_state.chat_history = []  # Clear LangChain conversation context
                st.success(f" Ready! {len(chunks)} chunks indexed from {len(uploaded_files)} file(s).")

            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                st.info("Common fixes:\n- Double-check your API key\n- Make sure PDFs contain real text (not scanned images)")

    # Loaded Documents Display
    if st.session_state.docs_loaded:
        st.divider()
        stats = st.session_state.doc_stats

        st.subheader("📊 Loaded")
        col1, col2 = st.columns(2)
        col1.metric("Files", stats["files"])
        col2.metric("Chunks", stats["chunks"])

        for fname in stats.get("filenames", []):
            st.markdown(f"📄 `{fname}`")

        st.caption(f"LLM: {stats['provider']} · Retrieving top-{stats['k']} chunks")

        if st.button("🗑️ Clear & start over", use_container_width=True):
            for key in ["rag_chain", "messages", "chat_history", "doc_stats"]:
                if key == "rag_chain":
                    st.session_state[key] = None
                elif key in ["messages", "chat_history"]:
                    st.session_state[key] = []
                else:
                    st.session_state[key] = {}
            st.session_state.docs_loaded = False
            st.rerun()


# MAIN CHAT AREA
st.title("💬 Chat with Your Documents")

# Status banner
if st.session_state.docs_loaded:
    st.success(
        f" {st.session_state.doc_stats['files']} document(s) loaded · "
        f"{st.session_state.doc_stats['chunks']} chunks indexed · "
        f"Ask anything below"
    )
else:
    # Landing state — show value props when no documents are loaded
    st.info(" Upload documents and enter your API key in the sidebar to get started.")

    st.markdown("### What this app does")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📊 Financial Reports**\nAsk about revenue, risks, forecasts from annual reports or 10-Ks")
    with c2:
        st.markdown("**🔬 Research Papers**\nExtract methodology, findings, and conclusions from PDFs")
    with c3:
        st.markdown("**⚖️ Legal Documents**\nUnderstand contracts, clauses, and compliance requirements")

    st.markdown("---")
    st.markdown(
        "**How it works:** Documents are chunked -> embedded with HuggingFace -> stored in FAISS. "
        "Your questions are embedded and matched to the most similar chunks. "
        "Only those chunks - not the whole document - are sent to the LLM, which answers with citations."
    )

st.divider()
# Render Chat History
# On each rerun, it replays the full message history so the UI looks continuous.
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        # Show source citations for assistant messages (collapsed by default)
        sources = message.get("sources", [])
        if sources and message["role"] == "assistant":
            with st.expander(f"📖 {len(sources)} source chunk(s) used"):
                for i, src in enumerate(sources, 1):
                    fname = src["metadata"].get("source", "unknown file")
                    page = src["metadata"].get("page", "?")
                    content = src["content"]
                    st.markdown(f"**Chunk {i}** — `{fname}` · page {page}")
                    # Show a preview (first 400 chars) with blockquote styling
                    preview = content[:400] + ("…" if len(content) > 400 else "")
                    st.markdown(f"> {preview}")
                    if i < len(sources):
                        st.markdown("---")


# Chat Input & Response Generation
# st.chat_input stays disabled until documents are loaded
user_question = st.chat_input(
    "Ask a question about your documents…",
    disabled=not st.session_state.docs_loaded,
)

if user_question:
    # 1. Show the user's message immediately (don't wait for LLM)
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_question)

    # 2. Generate the assistant's response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching documents and generating answer…"):
            try:
                # MODERN CORE RAG CALL:
                # chain.invoke() runs our LCEL history-aware retrieval chain pipeline.
                # It expects the user input in the "input" key and raw history list in "chat_history".
                result = st.session_state.rag_chain.invoke({
                    "input": user_question,
                    "chat_history": st.session_state.chat_history
                })

                answer = result.get("answer", "I couldn't generate a response.")
                
                # In modern create_retrieval_chain, retrieved chunks are returned in the "context" key
                raw_sources = result.get("context", [])

                # Normalize source documents into plain dicts for storage in session_state
                # (LangChain Document objects aren't directly JSON-serializable)
                sources = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in raw_sources
                ]

            except Exception as e:
                answer = f" Error: {str(e)}\n\nTry re-uploading your documents or checking your API key."
                sources = []

        # Display the answer
        st.markdown(answer)

        # Display sources inline (same logic as history replay above)
        if sources:
            with st.expander(f"📖 {len(sources)} source chunk(s) used"):
                for i, src in enumerate(sources, 1):
                    fname = src["metadata"].get("source", "unknown file")
                    page = src["metadata"].get("page", "?")
                    content = src["content"]
                    st.markdown(f"**Chunk {i}** — `{fname}` · page {page}")
                    preview = content[:400] + ("…" if len(content) > 400 else "")
                    st.markdown(f"> {preview}")
                    if i < len(sources):
                        st.markdown("---")

    # 3. Import chat message classes for tracking background context
    from langchain_core.messages import HumanMessage, AIMessage

    # Persist assistant message to UI history tracking list
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

    # Persist structured objects to LangChain's running context tracker
    st.session_state.chat_history.extend([
        HumanMessage(content=user_question),
        AIMessage(content=answer)
    ])

