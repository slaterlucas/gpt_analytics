from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from pathlib import Path
from .ingest_minimal import ingest_stream, topic_pie, job_status, model_stats

app = FastAPI(title="ChatGPT Analytics API - Minimal")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for testing
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

@app.get("/analysis/{jid}")
def full_analysis(jid: str):
    """Get comprehensive analysis including all metadata"""
    if jid not in JOBS:
        return {"error": "Job not found"}
    
    job = JOBS[jid]
    if not job.get("ready"):
        return {"error": "Job not ready"}
    
    if job.get("error"):
        return {"error": job["error"]}
    
    result = job.get("result", {})
    
    return {
        "job_id": jid,
        "summary": {
            "conversations": result.get("conversation_count", 0),
            "total_messages": result.get("total_messages", 0),
            "analyzed_messages": job.get("message_count", 0),
            "user_messages": result.get("user_messages", 0),
            "assistant_messages": result.get("assistant_messages", 0),
            "api_requests": job.get("model_requests", 0)
        },
        "content_types": result.get("content_types", {}),
        "topics": result.get("topics", []),
        "models": result.get("models", [])
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "version": "minimal"}

@app.get("/")
def root():
    """Root endpoint"""
    return {"message": "ChatGPT Analytics API - Minimal Version"} 