"""
Whiteboard Animator API
FastAPI backend for video generation
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
UPLOAD_DIR = Path("/tmp/whiteboard-uploads")
OUTPUT_DIR = Path("/tmp/whiteboard-outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Whiteboard Animator API", version="1.0.0")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job storage (in-memory for now, Redis later)
jobs: dict = {}

class JobStatus(BaseModel):
    id: str
    status: str  # pending, processing, complete, error
    progress: int  # 0-100
    created_at: str
    output_url: Optional[str] = None
    error: Optional[str] = None

class CreateJobRequest(BaseModel):
    duration: float = 8.0
    sketch: bool = True
    width: int = 1920
    height: int = 1080

async def process_video(job_id: str, input_path: Path, settings: dict):
    """Background task to generate the whiteboard video."""
    import subprocess
    import sys
    
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 10
        
        output_path = OUTPUT_DIR / f"{job_id}.mp4"
        
        # Build command
        cmd = [
            sys.executable,
            str(BASE_DIR / "whiteboard_animator_v3.py"),
            str(input_path),
            str(output_path),
            "--duration", str(settings.get("duration", 8)),
            "--width", str(settings.get("width", 1920)),
            "--height", str(settings.get("height", 1080)),
        ]
        
        if not settings.get("sketch", True):
            cmd.append("--no-sketch")
        
        jobs[job_id]["progress"] = 20
        
        # Run the animator
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Animator failed: {stderr.decode()}")
        
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["output_url"] = f"/api/download/{job_id}"
        
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

@app.get("/api")
async def api_root():
    return {"message": "Whiteboard Animator API", "version": "1.0.0"}

@app.post("/api/upload")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    duration: float = 8.0,
    sketch: bool = True,
    width: int = 1920,
    height: int = 1080,
):
    """Upload an image and start video generation."""
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    ext = Path(file.filename).suffix if file.filename else ".png"
    input_path = UPLOAD_DIR / f"{job_id}{ext}"
    
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create job record
    jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "progress": 0,
        "created_at": datetime.utcnow().isoformat(),
        "output_url": None,
        "error": None,
    }
    
    # Start background processing
    settings = {
        "duration": duration,
        "sketch": sketch,
        "width": width,
        "height": height,
    }
    background_tasks.add_task(process_video, job_id, input_path, settings)
    
    return {"job_id": job_id, "status": "pending"}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]

@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    """Download completed video."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    if job["status"] != "complete":
        raise HTTPException(400, f"Job not complete: {job['status']}")
    
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    if not output_path.exists():
        raise HTTPException(404, "Video file not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"whiteboard_{job_id}.mp4"
    )

@app.get("/api/jobs")
async def list_jobs():
    """List all jobs (for debugging)."""
    return list(jobs.values())

# Serve static frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
