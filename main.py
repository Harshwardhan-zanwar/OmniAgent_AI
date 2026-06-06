"""
omni-agent-ai — main.py
FastAPI entry point: routes, middleware, request/response models, endpoints.
"""

# Imports
import os
import uuid
import shutil
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

import google.generativeai as genai

#Load environment variables from .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

#Directory where uploaded files are temporarily stored
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

#Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omni-agent-ai")


#App Initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("omni-agent-ai starting up...")
    yield
    # Clean up upload folder on shutdown
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    logger.info("omni-agent-ai shut down cleanly.")

app = FastAPI(
    title="omni-agent-ai",
    description="Agentic app that accepts text, image, PDF, and audio inputs and autonomously performs tasks.",
    version="1.0.0",
    lifespan=lifespan,
)

#Allow frontend(served separately during dev, or same origin in prod) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      #tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Serve the frontend folder as static files
frontend_path = Path("frontend")
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


#Pydantic Models
class HealthResponse(BaseModel):
    status: str
    version: str

class EstimateRequest(BaseModel):
    query: str
    file_names: Optional[list[str]] = []

class EstimateResponse(BaseModel):
    input_tokens: int
    estimated_cost_usd: float
    model: str
    note: str

class AgentStep(BaseModel):
    step: int
    tool: str
    description: str
    status: str          #"success" | "failed" | "skipped"
    output_preview: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    query: str
    extracted_text: Optional[str] = None
    plan_trace: list[AgentStep]
    result: str
    follow_up_question: Optional[str] = None
    total_tokens_used: Optional[int] = None


#POST /chat  — Main Agent Endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(
    query: str = Form(default=""),
    files: list[UploadFile] = File(default=[]),
):
    """
    Accepts a text query and/or uploaded files (image, PDF, audio).
    Passes everything to the agent planner and returns:
      - extracted text from files
      - plan trace (which tools ran, in order)
      - final result
      - follow-up question if intent was unclear
    """
    #Lazy import to keep startup fast
    from agent.planner import run_agent

    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[{session_id}] New request — query='{query[:60]}', files={[f.filename for f in files]}")

    #Save uploaded files to disk 
    saved_paths: list[Path] = []
    for upload in files:
        if not upload.filename:
            continue
        safe_name = f"{session_id}_{upload.filename}"
        dest = UPLOAD_DIR / safe_name
        with dest.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
        saved_paths.append(dest)
        logger.info(f"[{session_id}] Saved upload → {dest}")

    #Run Agent
    try:
        result = await run_agent(
            session_id=session_id,
            query=query,
            file_paths=saved_paths,
        )
    except Exception as exc:
        logger.error(f"[{session_id}] Agent error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(exc)}")
    finally:
        #Clean up uploaded files after processing
        for path in saved_paths:
            path.unlink(missing_ok=True)

    return ChatResponse(
        session_id=session_id,
        query=query,
        extracted_text=result.get("extracted_text"),
        plan_trace=[AgentStep(**s) for s in result.get("plan_trace", [])],
        result=result.get("result", ""),
        follow_up_question=result.get("follow_up_question"),
        total_tokens_used=result.get("total_tokens_used"),
    )


#GET /health — Uptime Check for Render
@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Render (and any monitoring tool) pings this endpoint to verify the app is alive.
    Must return 200 OK quickly.
    """
    return HealthResponse(status="ok", version="1.0.0")


#POST /estimate — Cost Estimator(Bonus)

#Gemini 1.5 Flash pricing
COST_PER_1M_INPUT_TOKENS  = 0.075   # USD
COST_PER_1M_OUTPUT_TOKENS = 0.30    # USD
AVG_FILE_TOKENS = {
    "pdf":   3000,   #rough average per page × ~3 pages
    "image": 500,
    "audio": 2000,   #per minute of audio
    "other": 200,
}

@app.post("/estimate", response_model=EstimateResponse)
async def estimate(request: EstimateRequest):
    """
    Before running the agent, estimate how many tokens the request will consume
    and what it will cost. Shown in the UI so the user knows before clicking Send.
    """
    #Count query tokens (approx:1 token ≈ 4 chars)
    token_count = max(1, len(request.query) // 4)

    #Add file token estimates
    for fname in (request.file_names or []):
        ext = Path(fname).suffix.lower().lstrip(".")
        if ext in ("jpg", "jpeg", "png", "webp", "gif"):
            token_count += AVG_FILE_TOKENS["image"]
        elif ext == "pdf":
            token_count += AVG_FILE_TOKENS["pdf"]
        elif ext in ("mp3", "wav", "m4a", "ogg"):
            token_count += AVG_FILE_TOKENS["audio"]
        else:
            token_count += AVG_FILE_TOKENS["other"]

    #Add overhead for system prompt + tool descriptions (~800 tokens)
    token_count += 800

    cost = (token_count / 1_000_000) * COST_PER_1M_INPUT_TOKENS

    return EstimateResponse(
        input_tokens=token_count,
        estimated_cost_usd=round(cost, 6),
        model="gemini-2.5-flash",
        note="Estimate only. Actual cost may vary based on output length and tool calls.",
    )


#Serve Frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serve the chat UI. In production (Render), this is the only page users visit.
    """
    index = Path("frontend/index.html")
    return HTMLResponse(content=index.read_text(encoding="utf-8"))


#ENTRYPOINT (local dev only)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,       # auto-reload
        log_level="info",
    )