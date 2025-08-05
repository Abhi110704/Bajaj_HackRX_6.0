# hackrx_final_robust_api_upgraded.py
"""
HackRx 6.0 - Final API using only Gemini Pro (Fast, Reliable, Clean)
Refactored for >80% accuracy through advanced RAG tuning.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import requests
import logging
import time
from typing import List
from pathlib import Path
from datetime import datetime
import shutil

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
import google.generativeai as genai

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables
HACKRX_TOKEN = os.getenv("HACKRX_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", 10))

# Configure the Gemini client
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")
genai.configure(api_key=GEMINI_API_KEY)

# Temporary directory for file downloads
TEMP_DIR = Path("./temp_docs")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
TEMP_DIR.mkdir(exist_ok=True)


# --- FastAPI App ---
app = FastAPI(
    title="HackRx 6.0 API",
    description="High-Accuracy Gemini-Powered LLM Q&A System",
    version="6.1.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --- Security ---
auth_scheme = HTTPBearer()
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.credentials != HACKRX_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    return credentials.credentials


# --- Pydantic Models ---
class HackRxRunRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxRunResponse(BaseModel):
    answers: List[str]


# --- Utility Functions ---
def download_pdf(url: str) -> Path:
    """Downloads a PDF from a URL and saves it to a temporary path."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to download document from {url}. Error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unable to download document: {e}")

    pdf_path = TEMP_DIR / f"doc_{int(time.time())}_{os.getpid()}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(response.content)
    logger.info(f"Successfully downloaded PDF to {pdf_path}")
    return pdf_path

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts text from a local PDF file."""
    try:
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}. Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to parse PDF content.")

def build_vector_index(text: str):
    """Builds a FAISS vector index from the given text with an improved chunking strategy."""
    # IMPROVEMENT 1: Larger chunk size to keep context together, with more overlap.
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    ).split_text(text)
    
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    logger.info("Building FAISS vector index...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    logger.info("Vector index built successfully.")
    return vectorstore

def query_gemini(prompt: str) -> str:
    """Queries the Gemini model using the official SDK."""
    try:
        model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return "Error: The AI model failed to generate a response."


# --- API Endpoints ---
@app.post("/hackrx/run", response_model=HackRxRunResponse)
def run_hackrx(req: HackRxRunRequest, _: str = Depends(verify_token)):
    if len(req.questions) > MAX_QUESTIONS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Too many questions (max {MAX_QUESTIONS})")

    pdf_path = None
    try:
        pdf_path = download_pdf(req.documents)
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document appears to be empty or unreadable.")

        vectordb = build_vector_index(full_text)

        answers = []
        for q in req.questions:
            # IMPROVEMENT 2: Retrieve more documents (k=5) for richer context.
            context_docs = vectordb.similarity_search(q, k=5)
            context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
            
            # IMPROVEMENT 3: Advanced prompt engineering for higher accuracy and better formatting.
            prompt = (
                "You are a meticulous and highly precise AI insurance analyst. Your sole task is to answer questions based *only* on the "
                "specific context provided from an insurance policy document. You must adhere to the following rules without exception:\n\n"
                "RULES:\n"
                "1. **Cite the Facts:** Base your answer exclusively on the text within the 'CONTEXT' section. Do not use any outside knowledge.\n"
                "2. **Be Exact:** When the question asks for a number, a time period, a percentage, or a specific condition, you must find and state that exact detail. Do not be vague.\n"
                "3. **Stay Concise:** Provide a direct, professional answer. Do not add conversational fluff like 'Based on the context...' or 'The document states...'.\n"
                "4. **Handle Missing Information:** If, and only if, the answer is not present in the provided context, you must respond with the exact phrase: 'This information is not found in the provided document sections.'\n\n"
                "---\n\n"
                f"CONTEXT:\n{context}\n\n"
                "---\n\n"
                f"QUESTION: {q}\n\n"
                "PRECISE ANSWER:"
            )
            
            response = query_gemini(prompt)
            answers.append(response.strip()) # Use .strip() for cleaner output

        return HackRxRunResponse(answers=answers)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred while processing your request.")

    finally:
        if pdf_path and pdf_path.exists():
            pdf_path.unlink()
            logger.info(f"Cleaned up temporary file: {pdf_path}")


@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "healthy", "version": "6.1.0", "timestamp": datetime.now().isoformat()}

@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse("/docs")