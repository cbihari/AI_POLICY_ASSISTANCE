import os
import json
import uuid
import time
import hashlib
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Load Environment
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_SECRET = os.getenv("API_SECRET")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Initialize App
# -----------------------------
app = FastAPI(title="ACORD Extraction API", version="2.0")

# -----------------------------
# CORS Setup
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Root & Health
# -----------------------------
@app.get("/")
def root():
    return {"message": "ACORD Extraction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# -----------------------------
# PDF Text Extraction
# -----------------------------
def extract_text_from_pdf(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# -----------------------------
# Chunking for Large PDFs
# -----------------------------
def chunk_text(text, size=3000):
    return [text[i:i+size] for i in range(0, len(text), size)]

# -----------------------------
# AI Extraction Logic
# -----------------------------
def extract_json_from_text(text):

    prompt = """
You are an expert insurance ACORD extraction engine.

Return STRICT valid JSON only.
Do NOT add explanation.

Schema:
{
  "agency_name": string | null,
  "insured_name": string | null,
  "insured_address": {
      "street": string | null,
      "city": string | null,
      "state": string | null,
      "zip": string | null
  },
  "policy_start_date": string | null,
  "policy_end_date": string | null,
  "state": string | null,
  "liability_limit": string | null,
  "class_code": string | null,
  "business_description": string | null,
  "general_information": string | null
}
If value not found, return null.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a strict JSON extraction engine."},
            {"role": "user", "content": prompt + "\n\nDocument Text:\n" + text}
        ]
    )

    return json.loads(response.choices[0].message.content)

# -----------------------------
# Main Extraction Endpoint
# -----------------------------
@app.post("/extract-acord")
async def extract_acord(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):

    request_id = str(uuid.uuid4())

    # 🔐 API Authentication
    if API_SECRET and x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if file.content_type != "application/pdf":
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files allowed", "request_id": request_id}
        )

    try:
        start_time = time.time()

        file_bytes = await file.read()

        # 📏 File Size Limit (5MB)
        if len(file_bytes) > 5 * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={"error": "File too large (max 5MB)", "request_id": request_id}
            )

        # 🔍 Optional: hash file (future caching support)
        file_hash = hashlib.md5(file_bytes).hexdigest()
        print(f"[{request_id}] File Hash: {file_hash}")

        text = extract_text_from_pdf(file_bytes)

        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "No readable text found in PDF", "request_id": request_id}
            )

        # 📦 Chunk if large
        chunks = chunk_text(text)
        combined_text = "\n".join(chunks)

        data = extract_json_from_text(combined_text)

        end_time = time.time()
        print(f"[{request_id}] Extraction completed in {end_time - start_time:.2f} seconds")

        return {
            "status": "success",
            "request_id": request_id,
            "data": data
        }

    except Exception as e:
        print(f"[{request_id}] ERROR: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "request_id": request_id}
        )
