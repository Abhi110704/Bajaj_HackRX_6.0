# Main.py (optimized and cleaned)
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

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HACKRX_TOKEN = os.getenv("HACKRX_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", 10))

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set.")
genai.configure(api_key=GEMINI_API_KEY)

TEMP_DIR = Path("./temp_docs")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
TEMP_DIR.mkdir(exist_ok=True)

# --- FastAPI ---
app = FastAPI(
    title="HackRx 6.0 API",
    description="Enhanced Gemini RAG System",
    version="7.1.1-optimized"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Auth ---
auth_scheme = HTTPBearer()
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.credentials != HACKRX_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return credentials.credentials

# --- Models ---
class HackRxRunRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxRunResponse(BaseModel):
    answers: List[str]

# --- Helpers ---
def download_pdf(url: str) -> Path:
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        pdf_path = TEMP_DIR / f"doc_{int(time.time())}_{os.getpid()}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Downloaded PDF: {pdf_path}")
        return pdf_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")

def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to parse PDF")

def get_keywords(question: str) -> Set[str]:
    question = re.sub(r'[^\w\s-]', '', question.lower())
    stop_words = {"what", "is", "the", "and", "are", "a", "an", "for", "to", "of", "in", "does", "this", "policy", "cover", "under"}
    return {word for word in question.split() if word not in stop_words and len(word) > 3}

def find_relevant_chunks_fast(keywords: Set[str], all_chunks: List[Document]) -> List[Document]:
    relevant = []
    seen = set()
    for chunk in all_chunks:
        text = chunk.page_content.lower()
        if chunk.page_content in seen:
            continue
        if any(kw in text for kw in keywords):
            relevant.append(chunk)
            seen.add(chunk.page_content)
    return relevant

def query_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text if response.parts else "This information is not found in the provided document sections."
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return "Error: The AI model failed to generate a response."

# --- Main Endpoint ---
@app.post("/hackrx/run", response_model=HackRxRunResponse)
def run_hackrx(req: HackRxRunRequest, _: str = Depends(verify_token)):
    start_time = time.time()
    pdf_path = None
    try:
        if len(req.questions) > MAX_QUESTIONS:
            raise HTTPException(status_code=400, detail=f"Too many questions (max {MAX_QUESTIONS})")

        pdf_path = download_pdf(req.documents)
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Empty or unreadable document")

        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        all_chunks = splitter.create_documents([full_text])
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY)
        index = FAISS.from_documents(all_chunks, embeddings)

        answers = []
        for q in req.questions:
            logger.info(f"Processing: {q}")
            docs = index.similarity_search(q, k=7)
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)
            prompt = (
                "You are an AI assistant for an insurance company. Answer based strictly on the provided policy document.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {q}\n\nAnswer:"
            )
            ans = query_gemini(prompt).strip()
            answers.append(ans)

        total_time = time.time() - start_time
        logger.info(f"/hackrx/run completed in {total_time:.2f} seconds.")
        print(f"\n🎯 Total time to respond: {total_time:.2f} seconds.\n")
        return HackRxRunResponse(answers=answers)

    finally:
        if pdf_path and pdf_path.exists():
            pdf_path.unlink()
            logger.info(f"Cleaned up temp file: {pdf_path}")

@app.get("/health")
def health():
    return {"status": "healthy", "version": "7.1.1-optimized", "timestamp": datetime.now().isoformat()}

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")
