from fastapi import FastAPI, UploadFile, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from pathlib import Path
from .ingest import ingest_stream, topic_pie, job_status, model_stats, daily_activity, embedding_based_topic_analysis, estimate_analysis_cost
import json
from typing import Optional
from pydantic import BaseModel

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

class CostEstimateRequest(BaseModel):
    num_conversations: int
    num_clusters: int = 25
    use_llm_naming: bool = True

@app.post("/estimate-cost")
async def estimate_cost(request: CostEstimateRequest):
    """
    Estimate the cost of running embedding-based topic analysis with text-embedding-3-large
    """
    try:
        cost_breakdown = estimate_analysis_cost(
            num_conversations=request.num_conversations,
            num_clusters=request.num_clusters,
            use_llm_naming=request.use_llm_naming
        )
        return cost_breakdown
    except Exception as e:
        return {"error": f"Cost estimation error: {str(e)}"}

@app.post("/estimate-file-cost")
async def estimate_file_cost(file: UploadFile, 
                            num_clusters: int = 25, 
                            use_llm_naming: bool = True):
    """
    Estimate the cost of analyzing an uploaded ChatGPT export file with text-embedding-3-large
    """
    try:
        # Read and parse the file to count conversations
        content = await file.read()
        
        try:
            data = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError:
            return {"error": "Invalid JSON file format"}
        
        # Count conversations based on file format
        conversation_count = 0
        
        if isinstance(data, list):
            # Raw export format or cleaned conversations list
            conversation_count = len(data)
        elif isinstance(data, dict):
            if 'conversations' in data:
                # Cleaned format
                conversation_count = len(data['conversations'])
            elif 'mapping' in data:
                # Raw ChatGPT export format
                mapping = data.get('mapping', {})
                # Count nodes that are conversation roots (have user messages)
                conversation_count = 0
                for node_id, node_data in mapping.items():
                    message = node_data.get('message')
                    if message and message.get('author', {}).get('role') == 'user':
                        # This is a user message, count as conversation start
                        conversation_count += 1
            else:
                return {"error": "Unrecognized file format"}
        else:
            return {"error": "Invalid file structure"}
        
        if conversation_count == 0:
            return {"error": "No conversations found in file"}
        
        # Generate cost estimate
        cost_breakdown = estimate_analysis_cost(
            num_conversations=conversation_count,
            num_clusters=num_clusters,
            use_llm_naming=use_llm_naming
        )
        
        # Add file info
        cost_breakdown["file_info"] = {
            "filename": file.filename,
            "size_bytes": len(content),
            "detected_format": "cleaned" if 'conversations' in data else "raw_export" if 'mapping' in data else "list"
        }
        
        return cost_breakdown
        
    except Exception as e:
        return {"error": f"File analysis error: {str(e)}"}

@app.post("/upload")
async def upload(file: UploadFile, bg: BackgroundTasks, api_key: Optional[str] = Form(None)):
    """Upload ChatGPT export JSON and start background analysis"""
    jid = uuid4().hex
    path = DATA / f"{jid}.json"
    
    # Stream file to disk
    with path.open("wb") as out:
        while chunk := await file.read(1 << 20):  # 1 MiB chunks
            out.write(chunk)
    
    # Initialize job tracking
    JOBS[jid] = {"progress": 0, "ready": False, "error": None}
    
    # Start background processing with optional API key
    bg.add_task(ingest_stream, path, jid, JOBS, api_key)
    
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

@app.post("/analyze-cleaned")
async def analyze_cleaned_conversations(file: UploadFile, bg: BackgroundTasks, 
                                       num_clusters: int = 25, 
                                       max_conversations: Optional[int] = None,
                                       use_llm_naming: bool = True,
                                       api_key: Optional[str] = Form(None)):
    """
    Analyze cleaned conversation data using embedding-based topic clustering
    Expects a JSON file with cleaned conversations in the format from clean_conversations.py
    """
    jid = uuid4().hex
    
    try:
        # Read the uploaded cleaned conversations file
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        # Extract conversations from the cleaned format
        if 'conversations' in data:
            conversations = data['conversations']
        elif isinstance(data, list):
            conversations = data
        else:
            return {"error": "Invalid file format. Expected cleaned conversations JSON."}
        
        if not conversations:
            return {"error": "No conversations found in file"}
        
        # Initialize job tracking
        JOBS[jid] = {"progress": 0, "ready": False, "error": None, "result": None}
        
        # Start background analysis
        bg.add_task(run_embedding_analysis, conversations, jid, JOBS, num_clusters, max_conversations, use_llm_naming, api_key)
        
        return {"job_id": jid, "total_conversations": len(conversations)}
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON file"}
    except Exception as e:
        return {"error": f"File processing error: {str(e)}"}

@app.get("/embedding-results/{jid}")
def embedding_results(jid: str):
    """Get embedding-based topic analysis results"""
    if jid not in JOBS:
        return {"error": "Job not found"}
    
    job = JOBS[jid]
    if not job.get("ready"):
        return {"error": "Analysis not ready", "progress": job.get("progress", 0)}
    
    if job.get("error"):
        return {"error": job["error"]}
    
    return job.get("result", {})

def run_embedding_analysis(conversations, job_id: str, jobs: dict, num_clusters: int, max_conversations: Optional[int], use_llm_naming: bool, api_key: Optional[str]):
    """Background task to run embedding-based topic analysis"""
    try:
        jobs[job_id]["progress"] = 10
        
        # Set OpenAI API key if provided (client will be created in analysis functions)
        if api_key and api_key.strip():
            import openai
            # The client will be created in the analysis functions with the API key
        
        jobs[job_id]["progress"] = 20
        
        # Run embedding-based analysis
        result = embedding_based_topic_analysis(
            conversations=conversations,
            num_clusters=num_clusters,
            max_conversations=max_conversations,
            use_llm_naming=use_llm_naming,
            api_key=api_key
        )
        
        jobs[job_id]["progress"] = 90
        
        # Store result
        jobs[job_id]["result"] = result
        jobs[job_id]["ready"] = True
        jobs[job_id]["progress"] = 100
        
        print(f"✅ Embedding analysis complete for job {job_id}")
        
    except Exception as e:
        print(f"❌ Error in embedding analysis for job {job_id}: {e}")
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["ready"] = True

@app.get("/debug/jobs")
def debug_jobs():
    """Debug endpoint to see all jobs"""
    return {"jobs": JOBS} 