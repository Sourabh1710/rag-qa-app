"""
STEP 1: Document Ingestion Pipeline

"""

import io
from typing import List

import PyPDF2
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# TEXT EXTRACTION

def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Extract text from a PDF file, page by page.

    """
    documents = []
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))

    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()

        # Skip pages that are blank or contain only whitespace
        if not text or not text.strip():
            continue

        documents.append(Document(
            page_content=text,
            metadata={
                "source": filename,
                "page": page_num + 1,                    # 1-indexed for humans
                "total_pages": len(pdf_reader.pages),
                "file_type": "pdf",
            }
        ))

    return documents


def extract_text_from_txt(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Extract text from a plain text file.

    Text files are simpler - one Document for the whole file.
    It will UTF-8 first; if the file has odd encoding, It will ignore errors
    rather than crashing.

    """
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [Document(
        page_content=text,
        metadata={
            "source": filename,
            "page": 1,
            "total_pages": 1,
            "file_type": "txt",
        }
    )]


# TEXT CHUNKING

def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Split documents into smaller, overlapping chunks.

    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,  # Count characters (not tokens - simple and fast)
    )

    chunks = splitter.split_documents(documents)
    return chunks


# MAIN ENTRY POINT

def process_uploaded_files(
    uploaded_files,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Main entry point: takes Streamlit UploadedFile objects, returns Document chunks.

    This is what gets called from app.py when the user clicks "Process Documents".
    It handles multiple files and returns a flat list of all chunks, ready for embedding.

    The flow: files -> raw text -> LangChain Documents -> chunks -> ready for FAISS

    """
    all_chunks = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name

        # Route to the correct extractor based on file extension
        if filename.lower().endswith(".pdf"):
            raw_docs = extract_text_from_pdf(file_bytes, filename)
        elif filename.lower().endswith(".txt"):
            raw_docs = extract_text_from_txt(file_bytes, filename)
        else:
            print(f"  Skipping unsupported file type: {filename}")
            continue

        if not raw_docs:
            print(f"  No text extracted from {filename} -- skipping.")
            continue

        # Chunk the extracted text
        chunks = chunk_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

        print(f" {filename}: {len(raw_docs)} pages -> {len(chunks)} chunks")

    print(f"\n Total: {len(all_chunks)} chunks from {len(uploaded_files)} file(s)")
    return all_chunks
