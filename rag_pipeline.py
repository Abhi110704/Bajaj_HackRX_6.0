import os
from pathlib import Path
import requests
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# Set up Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Temp directory for downloads
TEMP_DIR = Path("./temp_docs")
TEMP_DIR.mkdir(exist_ok=True)

def download_and_extract(url: str) -> str:
    """Download a PDF from a URL, extract its text, then delete it."""
    response = requests.get(url)
    response.raise_for_status()
    
    pdf_path = TEMP_DIR / "eval_doc.pdf"
    with open(pdf_path, "wb") as f:
        f.write(response.content)

    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # Clean up
    pdf_path.unlink()
    return text

def generate_answer(question: str, full_text: str) -> str:
    """Generate an answer using RAG pipeline from provided full text and question."""
    # Split text into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.create_documents([full_text])

    # Embed using Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    index = FAISS.from_documents(chunks, embeddings)

    # Retrieve similar documents
    relevant_docs = index.similarity_search(question, k=7)

    # Build context
    context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

    prompt = (
        "You are an AI assistant for an insurance company. "
        "Answer based strictly on the provided policy document.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    response = model.generate_content(prompt)

    return (
        response.text if response.parts else
        "This information is not found in the provided document sections."
    )
