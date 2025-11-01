from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from pypdf import PdfReader
import requests
import json

# =====================================================
# 🔧 Load environment variables
# =====================================================
load_dotenv()

# --- GEMINI API Configuration ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
API_KEY = os.getenv('GEMINI_API_KEY')

# =====================================================
# ⚙️ Initialize Flask app
# =====================================================
app = Flask(__name__)
CORS(app)

# =====================================================
# 🗂️ In-memory storage for patient records
# =====================================================
patient_records = {}

# =====================================================
# 📄 Load PDF health record
# =====================================================
def load_pdf_record():
    """Reads the PDF file and stores extracted text in memory."""
    try:
        reader = PdfReader('sample-data.txt.pdf')
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Store in memory
        patient_records['test_patient'] = text
        print(f"✅ PDF health record loaded successfully! ({len(text)} characters)")
        print(f"✅ Preview: {text[:300]}...")

    except Exception as e:
        print(f"❌ Error loading PDF: {e}")

# Load sample data at startup
load_pdf_record()

# =====================================================
# 🧠 Gemini AI Agent Function
# =====================================================
def get_ai_response(user_question, patient_id='test_patient'):
    """Generates an AI response from Gemini using the patient's health record."""

    # Retrieve stored health record
    health_records = patient_records.get(patient_id, "No health records found.")

    # Truncate if too long to prevent token overflow
    if len(health_records) > 5000:
        health_records = health_records[:5000] + "\n\n[Note: Record truncated for AI analysis due to size limit.]"

    # Log input size for debugging
    print(f"📏 Input record length: {len(health_records)} characters")

    # Define AI behavior
    system_prompt = """You are an intelligent health assistant for the Ayu-Chain AI platform.

Your role:
1. Analyze patient health records carefully
2. Answer questions accurately based on the available data
3. Identify potential risks or patterns in health data
4. Be empathetic, simple, and clear in communication
5. If data is missing or unclear, state that and suggest consulting a doctor
"""

    # Create user context
    user_message = f"""
PATIENT HEALTH RECORDS:
{health_records}

PATIENT QUESTION:
{user_question}

Please analyze the records and answer the patient's question accurately and clearly.
"""

    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_message}]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1200   # Increased output capacity
        },
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        }
    }

    try:
        if not API_KEY:
            return "Error: Missing GEMINI_API_KEY in your .env file."

        # Send request to Gemini API
        response = requests.post(
            f"{GEMINI_API_URL}?key={API_KEY}",
            headers=headers,
            json=payload
        )

        response.raise_for_status()
        data = response.json()

        # Log entire Gemini response (for debugging)
        print("🔍 Full Gemini API response:")
        print(json.dumps(data, indent=2))

        # Extract AI-generated text safely
        if data.get('candidates'):
            candidate = data['candidates'][0]
            content = candidate.get('content', {})
            parts = content.get('parts', [])
            if parts and isinstance(parts[0], dict) and 'text' in parts[0]:
                return parts[0]['text']
            else:
                finish_reason = candidate.get("finishReason", "Unknown")
                return f"No valid text found in Gemini response. Finish reason: {finish_reason}. Response: {json.dumps(data, indent=2)}"

        # Handle known error cases
        error_message = data.get('error', {}).get('message', 'Unknown error or blocked content.')
        if response.status_code == 400:
            return f"Gemini API Error (400): {error_message}. Full details: {json.dumps(data, indent=2)}"

        return f"Gemini API Error: {error_message}"

    except requests.exceptions.RequestException as req_e:
        return f"Error connecting to Gemini API: {str(req_e)}"

    except Exception as e:
        return f"Error getting AI response: {str(e)}"

# =====================================================
# 🌐 API Endpoints
# =====================================================
@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Ayu-Chain AI Agent is running (Powered by Gemini)!",
        "version": "2.1.0",
        "records_loaded": len(patient_records) > 0
    })

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Handles patient health questions"""
    try:
        data = request.json
        question = data.get('question')
        patient_id = data.get('patient_id', 'test_patient')

        if not question:
            return jsonify({"error": "Question is required"}), 400

        print(f"📝 Question received: {question}")

        # Get AI answer
        answer = get_ai_response(question, patient_id)
        print("✅ AI Response generated successfully!")

        return jsonify({
            "success": True,
            "question": question,
            "answer": answer,
            "patient_id": patient_id
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_health():
    """Performs a proactive analysis of the patient record"""
    try:
        data = request.json
        patient_id = data.get('patient_id', 'test_patient')

        analysis_prompt = """Please analyze this patient's health record and provide:
1. Summary of their current health
2. Any patterns, risks, or trends noticed
3. Recommendations or lifestyle improvements
4. Questions the patient should ask their doctor
"""

        print("🔍 Performing proactive health analysis...")

        analysis = get_ai_response(analysis_prompt, patient_id)

        return jsonify({
            "success": True,
            "analysis": analysis,
            "patient_id": patient_id
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# =====================================================
# 🚀 Run the Server
# =====================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Starting Ayu-Chain AI Agent (Gemini)")
    print("📡 Server running at: http://localhost:5001")
    print("💡 Test endpoint: http://localhost:5001/")
    print("="*70 + "\n")
    app.run(debug=True, port=5001)
