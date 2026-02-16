import os
import json
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Load Environment
# -----------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found")

client = OpenAI(api_key=api_key)

# -----------------------------
# Initialize App
# -----------------------------
app = FastAPI(title="ACORD Extraction API", version="1.0")

# -----------------------------
# CORS Setup
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Health Check Endpoint
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
# AI Extraction Logic
# -----------------------------
def extract_json_from_text(text):

    prompt = """
You are an expert insurance ACORD extraction engine.

Return ONLY valid JSON. No explanation.
return yes/no questions that are coming in General Information section in pdf 

Fields:
- agency_name
- insured_name
- insured_address (object with street, city, state, zip)
- policy_start_date
- policy_end_date
- state
- liability_limit
- class_code
- business_description
- general information

If not found, return null.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
async def extract_acord(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files allowed"}
        )

    try:
        file_bytes = await file.read()
        text = extract_text_from_pdf(file_bytes)
        data = extract_json_from_text(text)

        return {
            "status": "success",
            "data": data
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )