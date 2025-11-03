"""
Background worker for processing GIF generation jobs sequentially.
Monitors database for queued jobs and processes them one-by-one.
"""

import time
from datetime import datetime
from pathlib import Path

from src.database import SessionLocal, get_queued_jobs, update_job_status, get_job
from src.image_generator import PudgyGIFGenerator
from src.config import Config


class JobWorker:
    """Background worker for sequential GIF generation job processing."""

    def __init__(self, check_interval: int = None):
        """
        Initialize the worker with single model for sequential processing.

        Args:
            check_interval: Seconds to wait between database checks (defaults to Config)
        """
        self.check_interval = check_interval or Config.WORKER_CHECK_INTERVAL
        self.generator = None

        print("🔧 Initializing Job Worker...")
        Config.print_config()
        print(f"\n   Check interval: {self.check_interval}s")
        print(f"   Processing mode: Sequential (1 job at a time)")
        print(f"   Memory efficient: Single model instance\n")

        # Pre-load model at startup
        print("🚀 Loading AI models (one time only)...")
        print("   This may take 1-2 minutes...\n")
        self._initialize_generator()

    def _initialize_generator(self):
        """Initialize the single PudgyGIFGenerator instance."""
        self.generator = PudgyGIFGenerator()
        print("✅ Model loaded and ready!")
        print("   Jobs will now be processed sequentially.\n")

    def process_job(self, job_id: str):
        """
        Process a single job sequentially.

        Args:
            job_id: Job ID to process
        """
        db = SessionLocal()
        try:
            job = get_job(db, job_id)

            if not job:
                print(f"⚠️  Job {job_id} not found")
                return

            # Update status to processing
            job.status = "processing"
            job.updated_at = datetime.utcnow()
            db.commit()

            print(f"\n{'='*60}")
            print(f"📋 Processing Job: {job.id}")
            print(f"{'='*60}")
            print(f"Prompt: {job.prompt}")

            try:
                # Parse parameters from database
                use_prompt_enhancer = job.use_prompt_enhancer.lower() == "true"
                lora_scale = float(job.lora_scale)
                image_seed = int(job.image_seed) if job.image_seed else None
                video_seed = int(job.video_seed) if job.video_seed else None

                # Generate GIF using pre-loaded model
                result = self.generator.generate_gif(
                    prompt=job.prompt,
                    output_name=f"job_{job.id}",
                    use_prompt_enhancer=use_prompt_enhancer,
                    lora_scale=lora_scale,
                    image_seed=image_seed,
                    video_seed=video_seed
                )

                # Update job with completed status
                gif_path = str(result["gif_path"])
                update_job_status(db, job.id, "completed", gif_path=gif_path)

                print(f"\n✅ Job {job.id} completed successfully!")
                print(f"   GIF: {gif_path}")

            except Exception as e:
                error_msg = str(e)
                print(f"\n❌ Job {job.id} failed: {error_msg}")

                # Update job with failed status
                update_job_status(db, job.id, "failed", error_message=error_msg)

        finally:
            db.close()

    def run(self):
        """Main worker loop - processes jobs sequentially one-by-one."""
        print("\n🚀 Worker started! Monitoring for new jobs...")
        print(f"   Processing: One job at a time (sequential)")
        print(f"   Press Ctrl+C to stop\n")

        try:
            while True:
                db = SessionLocal()
                try:
                    # Get one queued job
                    queued_jobs = get_queued_jobs(db, limit=1)

                    if queued_jobs:
                        job = queued_jobs[0]
                        print(f"\n🚀 Picked up job: {job.id}")

                        # Process the job
                        self.process_job(job.id)

                        print(f"✅ Job {job.id} finished. Looking for next job...\n")
                    else:
                        print(f"⏳ No jobs in queue. Checking again in {self.check_interval}s...", end='\r')
                        time.sleep(self.check_interval)

                finally:
                    db.close()

        except KeyboardInterrupt:
            print("\n\n⛔ Worker stopped by user")
        except Exception as e:
            print(f"\n\n❌ Worker crashed: {e}")
            raise


if __name__ == "__main__":
    worker = JobWorker()
    worker.run()
