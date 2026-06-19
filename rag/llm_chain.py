"""
STEP 3: LLM Integration & RAG Chain
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

# SYSTEM PROMPT — THE MOST IMPORTANT PART

# Modern LangChain splits prompts into "Contextualization" and "QA Answering"
# 1. Prompt to reformulate the user's question if they refer to past messages
CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question."""

CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 2. Main System Prompt for answering your documents strictly
QA_SYSTEM_PROMPT = """You are a precise document assistant. Your job is to answer questions based strictly on the provided context from the user's documents.

Context from retrieved document chunks:
{context}

Rules:
- Answer using ONLY the information in the context above.
- If the answer is not in the context, say exactly: "I don't have enough information in the uploaded documents to answer that question."
- Be concise and factual.
- When relevant, mention which part of the documents supports your answer.
- Do not use your training knowledge to fill gaps - only use the provided context."""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# LLM INITIALIZATION

def create_llm(api_key: str, provider: str = "Gemini"):
    """
    Initialize the language model.
    """
    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.2,
        )

    elif provider == "Groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=api_key,
            temperature=0.2,
        )

    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose 'Gemini' or 'Groq'.")


# RAG CHAIN ASSEMBLY

# REMOVED memory parameter as history is now passed directly during invocation
def create_rag_chain(vector_store: FAISS, llm, k: int = 3):
    """
    Assemble the full modern Retrieval Chain.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # Rephrases the question using chat history before sending it to FAISS
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALIZE_Q_PROMPT
    )

    # Combines the retrieved context documents with the prompt layout
    question_answer_chain = create_stuff_documents_chain(llm, QA_PROMPT)

    # Final end-to-end chain pipeline
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
