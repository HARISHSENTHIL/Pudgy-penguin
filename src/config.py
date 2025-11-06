"""
Configuration management using environment variables.
"""

import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# Load .env file (override=True allows reloading when .env changes)
load_dotenv()


class Config:
    """Configuration class for all settings."""

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Device Configuration
    DEVICE = os.getenv("DEVICE", "cuda")
    DTYPE_STR = os.getenv("DTYPE", "bfloat16")

    @staticmethod
    def get_dtype():
        """Get torch dtype from string."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(Config.DTYPE_STR, torch.bfloat16)

    # Model Paths
    FLUX_MODEL = os.getenv("FLUX_MODEL", "black-forest-labs/FLUX.1-dev")
    LORA_ID = os.getenv("LORA_ID", "Harish-as-harry/Pudgy-penguin")
    COGVIDEO_MODEL = os.getenv("COGVIDEO_MODEL", "THUDM/CogVideoX-5b-I2V")

    # Generation Defaults
    DEFAULT_LORA_SCALE = float(os.getenv("DEFAULT_LORA_SCALE", "0.8"))
    DEFAULT_IMAGE_WIDTH = int(os.getenv("DEFAULT_IMAGE_WIDTH", "720"))
    DEFAULT_IMAGE_HEIGHT = int(os.getenv("DEFAULT_IMAGE_HEIGHT", "480"))
    DEFAULT_IMAGE_STEPS = int(os.getenv("DEFAULT_IMAGE_STEPS", "25"))
    DEFAULT_IMAGE_GUIDANCE = float(os.getenv("DEFAULT_IMAGE_GUIDANCE", "3.5"))

    DEFAULT_NUM_FRAMES = int(os.getenv("DEFAULT_NUM_FRAMES", "49"))
    DEFAULT_VIDEO_STEPS = int(os.getenv("DEFAULT_VIDEO_STEPS", "50"))
    DEFAULT_VIDEO_GUIDANCE = float(os.getenv("DEFAULT_VIDEO_GUIDANCE", "6.0"))

    DEFAULT_GIF_FPS = int(os.getenv("DEFAULT_GIF_FPS", "10"))
    DEFAULT_GIF_WIDTH = int(os.getenv("DEFAULT_GIF_WIDTH", "640"))

    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8001"))
    API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

    # Worker Configuration
    WORKER_CHECK_INTERVAL = int(os.getenv("WORKER_CHECK_INTERVAL", "5"))

    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pudgy_jobs.db")

    # Output Directory
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./pudgy_outputs"))

    @classmethod
    def validate(cls):
        """Validate critical configuration."""
        if not cls.OPENAI_API_KEY:
            print("⚠️  Warning: OPENAI_API_KEY not set - prompt enhancement will be disabled")

        if cls.DEVICE == "cuda" and not torch.cuda.is_available():
            print("⚠️  Warning: CUDA not available, falling back to CPU")
            cls.DEVICE = "cpu"

        # Ensure output directory exists
        cls.OUTPUT_DIR.mkdir(exist_ok=True)

    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("📋 Configuration:")
        print(f"   Device: {cls.DEVICE}")
        print(f"   Dtype: {cls.DTYPE_STR}")
        print(f"   OpenAI API: {'✅ Configured' if cls.OPENAI_API_KEY else '❌ Not set'}")
        print(f"   FLUX Model: {cls.FLUX_MODEL}")
        print(f"   LoRA: {cls.LORA_ID}")
        print(f"   CogVideo: {cls.COGVIDEO_MODEL}")
        print(f"   Output Dir: {cls.OUTPUT_DIR}")
        print(f"   Database: {cls.DATABASE_URL}")


# Validate configuration on import
Config.validate()
