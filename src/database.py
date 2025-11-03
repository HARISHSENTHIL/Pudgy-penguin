from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid

from src.config import Config

engine = create_engine(Config.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Job(Base):
    """Job model for tracking GIF generation requests."""
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    status = Column(String, default="queued")  # queued, processing, completed, failed
    gif_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Optional parameters stored as JSON-serializable fields
    use_prompt_enhancer = Column(String, default="true")
    lora_scale = Column(String, default="0.8")
    image_seed = Column(String, nullable=True)
    video_seed = Column(String, nullable=True)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_job(db, prompt: str, use_prompt_enhancer: bool = True,
               lora_scale: float = 0.8, image_seed: int = None,
               video_seed: int = None) -> Job:
    """Create a new job in the database."""
    job_id = str(uuid.uuid4())

    job = Job(
        id=job_id,
        prompt=prompt,
        status="queued",
        use_prompt_enhancer=str(use_prompt_enhancer).lower(),
        lora_scale=str(lora_scale),
        image_seed=str(image_seed) if image_seed else None,
        video_seed=str(video_seed) if video_seed else None
    )

    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db, job_id: str) -> Job:
    """Get job by ID."""
    return db.query(Job).filter(Job.id == job_id).first()


def update_job_status(db, job_id: str, status: str,
                      gif_path: str = None, error_message: str = None):
    """Update job status and optional fields."""
    job = get_job(db, job_id)
    if job:
        job.status = status
        job.updated_at = datetime.utcnow()
        if gif_path:
            job.gif_path = gif_path
        if error_message:
            job.error_message = error_message
        db.commit()
        db.refresh(job)
    return job


def get_queued_jobs(db, limit=None):
    """Get all jobs with status 'queued'."""
    query = db.query(Job).filter(Job.status == "queued")
    if limit:
        query = query.limit(limit)
    return query.all()


def claim_job(db, job_id: str) -> bool:
    """
    Atomically claim a job by updating status from 'queued' to 'processing'.
    Returns True if successfully claimed, False if already claimed by another thread.

    This prevents race conditions where multiple threads try to process the same job.
    """
    job = db.query(Job).filter(Job.id == job_id, Job.status == "queued").first()
    if job:
        job.status = "processing"
        job.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(job)
        return True
    return False
