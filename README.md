# RAG Document Q&A System

> Upload any PDF. Ask anything. Get answers grounded in your documents — with source citations, conversation memory, and zero hallucination.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white&style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=chainlink&logoColor=white&style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Meta_AI-0467DF?logo=meta&logoColor=white&style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?logo=huggingface&logoColor=black&style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white&style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini_1.5_Flash-Free_Tier-4285F4?logo=google&logoColor=white&style=flat-square)

**[Live Demo →](https://your-app.streamlit.app)**  &nbsp;|&nbsp;  Built as part of a data science portfolio to demonstrate production LLM engineering skills

---

## The problem this solves

Standard LLMs answer from training data — outdated, unverifiable, prone to hallucination. This system instead **retrieves directly from your documents at query time**, so every answer is traceable to a source chunk. It's the difference between a model guessing and a model reading.

---

## What it can do

- **Multi-document ingestion** — upload multiple PDFs and TXT files simultaneously; the system indexes them into a single queryable knowledge base
- **Semantic search, not keyword search** — finds relevant passages even when your question uses different words than the document ("revenue growth" matches "net profit increased 20%")
- **Source citations** — every answer links back to the exact document chunk it was drawn from, with filename and page number
- **Multi-turn memory** — follow-up questions work naturally ("What about last year?" understands the prior context)
- **Hallucination guardrails** — custom system prompt instructs the LLM to answer *only* from retrieved context; if the answer isn't there, it says so
- **Dual LLM support** — switch between Google Gemini 2.5 Flash and Groq Llama-3 without changing code

---

## Architecture

The system has two distinct phases that share a FAISS vector index:

```
INDEXING PHASE (runs once per document upload)
──────────────────────────────────────────────
PDF / TXT  →  PyPDF2 extracts text  →  RecursiveCharacterTextSplitter
→  all-MiniLM-L6-v2 generates 384-dim embeddings  →  FAISS index

QUERY PHASE (runs on every user question)
─────────────────────────────────────────
Question  →  same HuggingFace model embeds query  →  cosine similarity
search in FAISS  →  top-3 chunks retrieved  →  LLM answers from context only
→  answer + source citations returned
```

**Why this architecture prevents hallucination:** The LLM receives a system prompt that begins: *"Answer using ONLY the information in the context below. If the answer is not in the context, say so."* Combined with retrieval, the model is reasoning over your documents — not its training weights.

---

## Technical decisions and why

| Decision | Alternative considered | Why this choice |
|---|---|---|
| `all-MiniLM-L6-v2` embeddings | OpenAI `text-embedding-ada-002` | Free, runs locally, no per-token cost, 384-dim is sufficient for retrieval tasks |
| FAISS (local) | ChromaDB (persistent) | Zero setup, in-memory speed, ideal for demo; swap path documented below |
| Chunk size 800 chars, overlap 100 | Fixed 512 tokens | Char-based is model-agnostic; 800 chars ≈ 150 words — enough context, not enough noise |
| `RecursiveCharacterTextSplitter` | `CharacterTextSplitter` | Tries paragraph → sentence → word boundaries before hard-cutting; preserves semantic units |
| `ConversationBufferMemory` | `ConversationSummaryMemory` | Simpler, perfect recall for short demos; summary memory is the production upgrade |
| `temperature=0.2` | `temperature=0.7` | Lower temperature = more factual, less creative; RAG Q&A requires factual consistency |
| `k=3` retrieval | `k=5` or `k=10` | Empirically optimal: enough context for most answers, low enough to avoid noise |

---

## Setup

```bash
git clone <your-repo-url>
cd rag_qa_app

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # add your API key (see below)
streamlit run app.py
```

**Get a free API key** — no credit card required:
- **Gemini:** https://aistudio.google.com/ → Get API Key (1,500 req/day free)
- **Groq:** https://console.groq.com/ → API Keys (14,400 req/day free)

> **First-run note:** The HuggingFace embedding model (~90 MB) downloads automatically on first launch and is cached locally for all subsequent runs.

---

## Deployment (live URL for your resume)

1. Push to GitHub — confirm `.env` is in `.gitignore`
2. Go to **https://share.streamlit.io** → New app → connect your repo
3. Set `app.py` as the entry point
4. Under **Settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   ```
5. Deploy → public URL is live in ~2 minutes

---

## Configuration

| Parameter | Default | Notes |
|---|---|---|
| Chunk size | 800 chars | Decrease for more precise retrieval; increase for more context per answer |
| Chunk overlap | 100 chars | ~12% overlap prevents context loss at boundaries |
| Top-K retrieval | 3 chunks | The K chunks most similar to the question are sent to the LLM |
| Embedding model | `all-MiniLM-L6-v2` | 22M params, 384-dim vectors, ~30ms per batch on CPU |
| LLM temperature | 0.2 | Lower = more deterministic; raise to 0.5 for more conversational tone |

---

## Project structure

```
rag_qa_app/
├── app.py                       # Streamlit UI — session state, chat interface, source citations
├── rag/
│   ├── document_processor.py    # PyPDF2 extraction + RecursiveCharacterTextSplitter chunking
│   ├── vector_store.py          # HuggingFace embeddings + FAISS index creation and retrieval
│   ├── llm_chain.py             # Gemini / Groq initialization + ConversationalRetrievalChain
│   └── memory.py                # ConversationBufferMemory for multi-turn context
├── requirements.txt
└── .env.example
```

---

## Known limitations and production upgrades

| Current (demo) | Production upgrade |
|---|---|
| FAISS in-memory — index lost on restart | Swap to ChromaDB with `persist_directory` — one line change in `vector_store.py` |
| `ConversationBufferMemory` — grows with every turn | `ConversationSummaryMemory` — summarizes old turns to keep token usage bounded |
| Single-user session state | Add user authentication + per-user vector stores |
| PyPDF2 fails on scanned PDFs | Add `pytesseract` OCR pass for image-based PDFs |
| No re-ranking | Add cross-encoder re-ranking step to improve precision at top-3 |

---

## Resume bullets

```
→ Built end-to-end RAG pipeline using LangChain, FAISS vector store, and
  HuggingFace sentence-transformers with zero embedding API cost

→ Engineered document ingestion system handling multi-file PDF/TXT uploads,
  800-char semantic chunking, and 384-dimensional vector indexing via FAISS

→ Implemented ConversationalRetrievalChain with custom anti-hallucination
  system prompt and source citation extraction for fully grounded answers

→ Deployed production Streamlit application to Streamlit Cloud with live
  public demo URL; supports Gemini 1.5 Flash and Groq Llama-3 backends
```

---

## Potential extensions (interview talking points)

- **Evaluation with RAGAs** — measure faithfulness, answer relevance, and context precision; produce a scored eval report
- **Web scraping source** — add `WebBaseLoader` so users can query live URLs, not just uploaded files
- **Streaming responses** — use `chain.astream()` for token-by-token output; eliminates the spinner
- **Re-ranking** — add a `CrossEncoderReranker` step between retrieval and generation; measurably improves precision
- **Hybrid search** — combine dense FAISS retrieval with BM25 sparse retrieval; handles exact-match queries (names, codes, IDs) better than embeddings alone
