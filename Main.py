# Main.py
"""
HackRx 6.0 Submission - High-Accuracy, High-Speed RAG System
Designed for >90% accuracy and <30s response time on a free tier.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import requests
import logging
import time
from typing import List, Set
from pathlib import Path
from datetime import datetime
import shutil
import re

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

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
    description="High-Speed, High-Accuracy Gemini Q&A System (Free Tier)",
    version="7.0.0-final"
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
    try:
        response = requests.get(url, timeout=20) # Lowered timeout
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
    try:
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}. Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to parse PDF content.")

def get_keywords(question: str) -> Set[str]:
    """Extracts simple, meaningful keywords from a question."""
    question = re.sub(r'[^\w\s-]', '', question.lower())
    stop_words = {"what", "is", "the", "and", "are", "a", "an", "for", "to", "of", "in", "does", "this", "policy", "cover", "under"}
    words = {word for word in question.split() if word not in stop_words and len(word) > 3}
    return words

def find_relevant_chunks_fast(keywords: Set[str], all_chunks: List[Document]) -> List[Document]:
    """Performs a fast keyword search to find relevant chunks."""
    relevant_chunks = [] # <-- FIX 1: Initialize as a list
    seen_chunks = set() # Helper to track duplicates
    for chunk in all_chunks:
        # Create a unique identifier for the chunk to avoid duplicates
        chunk_id = chunk.page_content
        if chunk_id in seen_chunks:
            continue

        chunk_text_lower = chunk.page_content.lower()
        if any(keyword in chunk_text_lower for keyword in keywords):
            relevant_chunks.append(chunk) # <-- FIX 2: Append to the list
            seen_chunks.add(chunk_id)
    return relevant_chunks

def build_targeted_vector_index(docs: List[Document]):
    """Builds a FAISS index from a SMALL list of pre-selected documents."""
    if not docs:
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY)
    logger.info(f"Building targeted FAISS index for {len(docs)} chunks...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    logger.info("Targeted vector index built successfully.")
    return vectorstore

def query_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        # Add a check for empty or blocked responses
        if not response.parts:
            logger.warning("Gemini response was blocked or empty.")
            return "The model could not generate a response for this query."
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
        # --- Stage 1: Fast initial processing (Done once per request) ---
        pdf_path = download_pdf(req.documents)
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document appears to be empty or unreadable.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        all_chunks = text_splitter.create_documents([full_text])

        answers = []
        # --- Stage 2: Process each question individually and quickly ---
        for q in req.questions:
            logger.info(f"Processing question: '{q}'")
            
            keywords = get_keywords(q)
            candidate_chunks = find_relevant_chunks_fast(keywords, all_chunks)

            if not candidate_chunks:
                logger.warning(f"No relevant chunks found via keyword search for: {keywords}")
                # As a fallback, use the first few chunks of the document
                candidate_chunks = all_chunks[:10] # Fallback to first 10 chunks

            try:
                vectordb = build_targeted_vector_index(candidate_chunks)
                if not vectordb:
                    answers.append("Could not build a searchable index for this question.")
                    continue
            except Exception as e:
                logger.error(f"Failed to build vector index for question '{q}': {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"An error occurred during embedding: {e}")

            context_docs = vectordb.similarity_search(q, k=5)
            context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
            
            prompt = (
                "You are an AI assistant for an insurance company. Your task is to answer questions based strictly on the provided text from a policy document. "
                "Provide direct, factual answers in the same format as the examples. Do not add any conversational phrases. "
                "If the information is not in the provided text, you must state exactly: 'This information is not found in the provided document sections.'\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {q}\n\n"
                "ANSWER:"
            )
            
            response = query_gemini(prompt)
            answers.append(response.strip())

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
    return {"status": "healthy", "version": "7.0.0-final", "timestamp": datetime.now().isoformat()}

@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse("/docs")
