"""omni-agent-ai main entrypoint."""
import os
import uuid
import shutil
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

gemini_key=os.getenv("GEMINI_API_KEY")
upload_dir=Path("uploads")
upload_dir.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger("omni-agent-ai")

@asynccontextmanager
async def lifespan(app:FastAPI):
    logger.info("omni-agent-ai starting...")
    yield
    shutil.rmtree(upload_dir,ignore_errors=True)
    upload_dir.mkdir(exist_ok=True)
    logger.info("omni-agent-ai shut down.")

app=FastAPI(
    title="omni-agent-ai",
    description="Multi-modal agentic app.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path=Path("frontend")
if frontend_path.exists():
    app.mount("/static",StaticFiles(directory="frontend"),name="static")

class HealthResponse(BaseModel):
    status:str
    version:str

class EstimateRequest(BaseModel):
    query:str
    file_names:Optional[list[str]]=[]

class EstimateResponse(BaseModel):
    input_tokens:int
    estimated_cost_usd:float
    model:str
    note:str

class AgentStep(BaseModel):
    step:int
    tool:str
    description:str
    status:str
    output_preview:Optional[str]=None

class ChatResponse(BaseModel):
    session_id:str
    query:str
    extracted_text:Optional[str]=None
    plan_trace:list[AgentStep]
    result:str
    follow_up_question:Optional[str]=None
    total_tokens_used:Optional[int]=None

@app.post("/chat",response_model=ChatResponse)
async def chat(
    query:str=Form(default=""),
    files:list[UploadFile]=File(default=[]),
):
    """Handle chat query and files via agent planner."""
    from agent.planner import run_agent

    sid=str(uuid.uuid4())[:8]
    logger.info(f"[{sid}] Request: query='{query[:60]}', files={[f.filename for f in files]}")

    paths=[]
    for up in files:
        if not up.filename:
            continue
        name=f"{sid}_{up.filename}"
        dest=upload_dir/name
        with dest.open("wb") as out:
            shutil.copyfileobj(up.file,out)
        paths.append(dest)
        logger.info(f"[{sid}] Saved: {dest}")

    try:
        res=await run_agent(
            session_id=sid,
            query=query,
            file_paths=paths,
        )
    except Exception as e:
        logger.error(f"[{sid}] Agent error: {e}",exc_info=True)
        raise HTTPException(status_code=500,detail=f"Agent error: {str(e)}")
    finally:
        for p in paths:
            p.unlink(missing_ok=True)

    return ChatResponse(
        session_id=sid,
        query=query,
        extracted_text=res.get("extracted_text"),
        plan_trace=[AgentStep(**s) for s in res.get("plan_trace",[])],
        result=res.get("result",""),
        follow_up_question=res.get("follow_up_question"),
        total_tokens_used=res.get("total_tokens_used"),
    )

@app.get("/health",response_model=HealthResponse)
async def health():
    """Uptime health check."""
    return HealthResponse(status="ok",version="1.0.0")

cost_in=0.075
cost_out=0.30
avg_file_tokens={
    "pdf":3000,
    "image":500,
    "audio":2000,
    "other":200,
}

@app.post("/estimate",response_model=EstimateResponse)
async def estimate(request:EstimateRequest):
    """Estimate token count and cost."""
    tokens=max(1,len(request.query)//4)

    for name in (request.file_names or []):
        ext=Path(name).suffix.lower().lstrip(".")
        if ext in ("jpg","jpeg","png","webp","gif"):
            tokens+=avg_file_tokens["image"]
        elif ext=="pdf":
            tokens+=avg_file_tokens["pdf"]
        elif ext in ("mp3","wav","m4a","ogg"):
            tokens+=avg_file_tokens["audio"]
        else:
            tokens+=avg_file_tokens["other"]

    tokens+=800
    cost=(tokens/1000000)*cost_in

    return EstimateResponse(
        input_tokens=tokens,
        estimated_cost_usd=round(cost,6),
        model="gemini-2.5-flash",
        note="Estimate only.",
    )

@app.get("/",response_class=HTMLResponse)
async def serve_frontend():
    """Serve UI."""
    index=Path("frontend/index.html")
    return HTMLResponse(content=index.read_text(encoding="utf-8"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )