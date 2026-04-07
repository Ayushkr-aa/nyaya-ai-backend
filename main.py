from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from chatbot.rag_engine import get_rag_response
from chatbot.memory import memory
from captcha.captcha_engine import generate_captcha, verify_captcha
import uuid

from chatbot.local_llm import get_local_llm

app = FastAPI(title="DoJ Legal Assistant API")

@app.on_event("startup")
def startup_event():
    # Pre-load local model if enabled to avoid first-request lag
    from chatbot.rag_engine import USE_LOCAL_MODEL
    if USE_LOCAL_MODEL:
        try:
            print("⏳ Pre-loading local AI model...")
            get_local_llm()
            print("🚀 Local AI ready!")
        except Exception as e:
            print(f"❌ Failed to pre-load local model: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""
    language: str = "auto"  # "auto", "en", "hi"


class CaptchaVerifyRequest(BaseModel):
    captcha_id: str
    user_answer: str


@app.get("/")
def root():
    return {"status": "DoJ Legal Assistant API running", "version": "2.0"}


@app.get("/api/health")
def health():
    return {"status": "ok", "rag": True, "llm": "gemini"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    # Get RAG response
    result = get_rag_response(
        message=req.message,
        session_id=session_id,
        language=req.language
    )

    # Check if we should show lawyer CTA
    turn_count = memory.get_user_turn_count(session_id)
    show_lawyer_cta = turn_count >= 3

    return {
        "session_id": session_id,
        "response": result["response"],
        "sources": result["sources"],
        "follow_ups": result["follow_ups"],
        "intent": result["intent"],
        "language": result["language"],
        "show_lawyer_cta": show_lawyer_cta,
        "turn_count": turn_count
    }


@app.post("/api/chat/upload")
async def chat_with_upload(
    file: UploadFile = File(...),
    session_id: str = "",
    language: str = "auto"
):
    """Upload a document (PDF/text) and ask questions about it."""
    session_id = session_id or str(uuid.uuid4())

    # Read file content
    content = await file.read()
    text = ""

    if file.filename.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(content))
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            raise HTTPException(400, f"Could not read PDF: {e}")
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

    if not text.strip():
        raise HTTPException(400, "Could not extract text from the uploaded file.")

    # Truncate if too long
    text = text[:5000]

    # Ask the LLM to analyze the document
    query = f"The user uploaded a document. Please analyze it and explain its legal significance:\n\n{text[:3000]}"

    result = get_rag_response(
        message=query,
        session_id=session_id,
        language=language
    )

    return {
        "session_id": session_id,
        "response": result["response"],
        "sources": result["sources"],
        "follow_ups": result["follow_ups"],
        "intent": "document_analysis",
        "language": result["language"],
        "filename": file.filename
    }


@app.get("/api/captcha/generate")
def get_captcha():
    captcha_id, image_b64, _ = generate_captcha()
    return {"captcha_id": captcha_id, "image": image_b64}


@app.post("/api/captcha/verify")
def verify(req: CaptchaVerifyRequest):
    result = verify_captcha(req.captcha_id, req.user_answer)
    return {"success": result}
