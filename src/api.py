"""
FastAPI application for Pudgy GIF generation service.
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pathlib import Path

from src.database import init_db, get_db, create_job, get_job
from src.models import GenerateRequest, GenerateResponse, StatusResponse, HealthResponse
from src.config import Config

app = FastAPI(
    title="Pudgy GIF Generator API",
    description="AI-powered Pudgy Penguin GIF generation service",
    version="1.0.0"
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()
    print("✅ Database initialized")


@app.get("/", response_model=HealthResponse)
def root():
    """Root endpoint."""
    return {
        "status": "ok",
        "message": "Pudgy GIF Generator API is running"
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "API is operational"
    }


@app.post("/generate", response_model=GenerateResponse, status_code=status.HTTP_201_CREATED)
def generate_gif(request: GenerateRequest, db: Session = Depends(get_db)):
    try:
        job = create_job(
            db=db,
            prompt=request.prompt,
            use_prompt_enhancer=request.use_prompt_enhancer,
            lora_scale=request.lora_scale,
            image_seed=request.image_seed,
            video_seed=request.video_seed
        )

        return {
            "job_id": job.id,
            "status": job.status,
            "message": "Job created successfully. Use /status/{job_id} to check progress."
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}"
        )


@app.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str, db: Session = Depends(get_db)):
    """
    Get the status of a generation job.

    Args:
        job_id: Job ID returned from /generate
        db: Database session

    Returns:
        Job status and details
    """
    job = get_job(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    response = {
        "job_id": job.id,
        "status": job.status,
        "prompt": job.prompt,
        "created_at": job.created_at,
        "updated_at": job.updated_at
    }

    # Add GIF URL if completed
    if job.status == "completed" and job.gif_path:
        response["gif_url"] = f"/download/{job.id}"

    # Add error message if failed
    if job.status == "failed" and job.error_message:
        response["error_message"] = job.error_message

    return response


@app.get("/download/{job_id}")
def download_gif(job_id: str, db: Session = Depends(get_db)):
    """
    Download the generated GIF file.

    Args:
        job_id: Job ID
        db: Database session

    Returns:
        GIF file
    """
    job = get_job(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not completed yet. Current status: {job.status}"
        )

    if not job.gif_path or not Path(job.gif_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="GIF file not found"
        )

    return FileResponse(
        path=job.gif_path,
        media_type="image/gif",
        filename=f"pudgy_{job_id}.gif"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.API_RELOAD
    )
