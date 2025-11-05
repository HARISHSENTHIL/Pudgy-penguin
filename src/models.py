"""
Pydantic models for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class GenerateRequest(BaseModel):
    """Request model for GIF generation."""
    prompt: str = Field(..., description="Simple user prompt (e.g., 'pudgy penguin as ironman')")
    use_prompt_enhancer: bool = Field(True, description="Use OpenAI to enhance prompts")
    lora_scale: float = Field(0.8, ge=0.6, le=1.2, description="LoRA strength (0.6-1.2)")
    image_seed: Optional[int] = Field(None, description="Random seed for image generation")
    video_seed: Optional[int] = Field(None, description="Random seed for video generation")


class GenerateResponse(BaseModel):
    """Response model for job creation."""
    job_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    """Response model for job status check."""
    job_id: str
    status: str
    prompt: str
    created_at: datetime
    updated_at: datetime
    gif_url: Optional[str] = None
    webp_url: Optional[str] = None
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    message: str
