import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)


# -----------------------------
# 1️⃣ Extract Text from PDF
# -----------------------------
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


# -----------------------------
# 2️⃣ Extract Structured JSON
# -----------------------------
def extract_json_from_text(text):
    prompt = """
You are an expert insurance ACORD form extraction engine.

Carefully analyze the document text and extract structured data.

Important:
- Class code is usually a 4 or 5 digit number near business classification.
- Liability limit is typically a monetary value like 1,000,000.
- Policy dates are in MM/DD/YYYY format.
- Address should include street, city, state, and zip if available.

Return ONLY valid JSON. No explanation. No markdown.

Fields:
- agency_name
- insured_name
- insured_address
- policy_start_date
- policy_end_date
- state
- liability_limit
- class_code
- business_description

If a value is not clearly present, return null.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},  # 🔥 Enforce strict JSON
        messages=[
            {"role": "system", "content": "You are a strict JSON extraction engine."},
            {"role": "user", "content": prompt + "\n\nDocument Text:\n" + text}
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# 3️⃣ Validate Output
# -----------------------------
def validate_output(data):
    required_fields = [
        "agency_name",
        "insured_name",
        "insured_address",
        "policy_start_date",
        "policy_end_date",
        "state",
        "liability_limit",
        "class_code",
        "business_description"
    ]

    for field in required_fields:
        if field not in data:
            data[field] = None

    return data


# -----------------------------
# 4️⃣ Main Execution
# -----------------------------
if __name__ == "__main__":
    pdf_path = "sample_policy.pdf"

    print("📄 Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)

    print("🤖 Sending to AI for structured extraction...")
    result = extract_json_from_text(pdf_text)

    try:
        parsed_json = json.loads(result)
        validated_data = validate_output(parsed_json)

        print("\n✅ Extracted Structured JSON:\n")
        print(json.dumps(validated_data, indent=4))

        # Optional: Save to file
        with open("output.json", "w") as f:
            json.dump(validated_data, f, indent=4)

        print("\n💾 JSON saved to output.json")

    except json.JSONDecodeError:
        print("❌ AI response was not valid JSON:")
        print(result)