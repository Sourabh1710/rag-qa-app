"""
STEP 2: Vector Store & Embeddings

"""

from typing import List

import streamlit as st
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# EMBEDDING MODEL

@st.cache_resource  
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace sentence-transformer embedding model.

    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# VECTOR STORE CREATION

def create_vector_store(chunks: List[Document]) -> FAISS:
    """
    Embed all document chunks and store them in a FAISS index.

    """
    print(f"  Generating embeddings for {len(chunks)} chunks...")
    print("   (First run downloads ~90MB model -- subsequent runs are instant)")

    embeddings = get_embedding_model()

    # FAISS.from_documents() does two things:
    #   1. Calls embeddings.embed_documents([chunk.page_content for chunk in chunks])
    #   2. Builds the FAISS index from those vectors
    vector_store = FAISS.from_documents(chunks, embeddings)

    print(f" FAISS index built with {len(chunks)} vectors (384 dimensions each)")
    return vector_store


# RETRIEVER

def get_retriever(vector_store: FAISS, k: int = 3):
    """
    Create a retriever from the FAISS vector store.
    
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
