
# 🛡️ Policy Aware Assistant – HackRx 6.0

An intelligent FastAPI-based backend system that parses insurance documents (PDFs), understands user queries, and responds with accurate, policy-specific answers using rule-based NLP and optional Perplexity API fallback.

---

## 👥 Team Details

**Team Name:** HackerXHacker  
**Team Members:**
- Abhiyanshu Anand  
- Sanskar Singh  
- Abhishek Singh  
- Siddharth Tripathi  
- Harsh Katiyar

---

## 🚀 Features

✅ PDF Policy Parsing  
✅ Multiple Question Answering  
✅ Intelligent Semantic Matching  
✅ Optional Perplexity LLM Integration  
✅ Docker-Ready Deployment  
✅ Works With Any Policy Document (Health, General, etc.)

---

## 🏗️ Folder Structure

```
HackRx_Complete_Submission/
├── Main.py               # Main FastAPI backend
├── requirements.txt      # Python dependencies
```

---

## 🧪 How to Run Locally

1. **Clone Repo & Setup Virtual Env**
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
pip install -r requirements.txt
```

2. **Create `.env` File**
```env
HACKRX_TOKEN=71bb650c7118766ab68c3df4923475c4cddb449b7332aa8cefb2d48aa3554e4b
GEMINI_API_KEY=AIzaSy...your_gemini_api_key_here
```

3. **Run the API**
```bash
uvicorn Main:app --reload
```

4. **Test at Swagger UI**
Visit 👉 https://hackerxhacker.onrender.com/docs
Use `POST /hackrx/run` endpoint to ask queries on any insurance PDF via URL.

---

## 🔁 Sample Request (JSON)

```json
{
  "documents": "https://example.com/sample-policy.pdf",
  "questions": [
    "What is the waiting period for cataract surgery?",
    "Does this policy cover AYUSH treatment?",
    "Is ICU stay covered?"
  ]
}
```

⚠️ Add the Bearer Token in `Authorize`:
```
Bearer your_token_here
```

---

## 🧠 How It Works

- Extracts text from insurance PDFs (remote URLs)
- Uses keyword rules & pattern matching for policy-specific answers
- Falls back to Perplexity API for generic or fuzzy queries (optional)
- Detects mismatched or unrelated policy references and avoids hallucination

---

## ✅ HackRx Testing Checklist

| Feature                            | Status |
|-----------------------------------|--------|
| PDF Document Ingestion (via URL)  | ✅     |
| Ask Multiple Questions            | ✅     |
| RAG - based Answer Extraction     | ✅     |
| Token-based API Auth              | ✅     |
| Google Gemini Integration         | ✅     |
---

## 📚 Example PDFs to Try

- Arogya Sanjeevani Policy (Govt Health Policy)
- National Health Insurance (Private)
- Any PDF with standard terms like coverage, exclusions, waiting period

---

## 📬 Contact

Built with 💡 for **HackRx 6.0** – Policy Aware AI Challenge  
Team: HackerXHacker  
