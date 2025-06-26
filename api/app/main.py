from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from pathlib import Path
from .ingest import ingest_stream, topic_pie, job_status, model_stats, daily_activity
import json

app = FastAPI(title="ChatGPT Analytics API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA = Path(".cache")
DATA.mkdir(exist_ok=True)
JOBS = {}

@app.post("/upload")
async def upload(file: UploadFile, bg: BackgroundTasks):
    """Upload ChatGPT export JSON and start background analysis"""
    jid = uuid4().hex
    path = DATA / f"{jid}.json"
    
    # Stream file to disk
    with path.open("wb") as out:
        while chunk := await file.read(1 << 20):  # 1 MiB chunks
            out.write(chunk)
    
    # Initialize job tracking
    JOBS[jid] = {"progress": 0, "ready": False, "error": None}
    
    # Start background processing
    bg.add_task(ingest_stream, path, jid, JOBS)
    
    return {"job_id": jid}

@app.get("/status/{jid}")
def status(jid: str):
    """Get job processing status"""
    return job_status(jid, JOBS)

@app.get("/topics/{jid}")
def topics(jid: str):
    """Get topic analysis results"""
    return topic_pie(jid, JOBS)

@app.get("/models/{jid}")
def models(jid: str):
    """Get model usage statistics"""
    return model_stats(jid, JOBS)

@app.get("/daily/{jid}")
def daily(jid: str):
    """Get daily message activity for time series chart"""
    return daily_activity(jid, JOBS)

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok"} 