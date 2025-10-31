"""
Background worker for processing GIF generation jobs.
Monitors database for queued jobs and processes them.
"""

import torch
import time
from pathlib import Path
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.database import SessionLocal, get_queued_jobs, update_job_status
from src.image_generator import PudgyGIFGenerator
from src.config import Config


class JobWorker:
    """Background worker for processing GIF generation jobs."""

    def __init__(self, check_interval: int = None, concurrency: int = None):
        """
        Initialize the worker.

        Args:
            check_interval: Seconds to wait between database checks (defaults to Config)
            concurrency: Number of jobs to process in parallel (defaults to Config)
        """
        self.check_interval = check_interval or Config.WORKER_CHECK_INTERVAL
        self.concurrency = concurrency or Config.WORKER_CONCURRENCY
        self.generator = None
        self.generator_lock = threading.Lock()

        print("🔧 Initializing Job Worker...")
        Config.print_config()
        print(f"\n   Check interval: {self.check_interval}s")
        print(f"   Concurrency: {self.concurrency} job(s) in parallel")

    def initialize_generator(self):
        """Initialize the PudgyGIFGenerator (lazy loading). Uses Config defaults. Thread-safe."""
        with self.generator_lock:
            if self.generator is None:
                print("\n🐧 Loading AI models (this may take a while)...")
                # Generator will use Config values automatically
                self.generator = PudgyGIFGenerator()
                print("✅ Models loaded and ready!\n")

    def process_job(self, job_id: str):
        """
        Process a single job. Thread-safe with separate DB session.

        Args:
            job_id: Job ID to process
        """
        # Create separate DB session for this thread
        db = SessionLocal()
        try:
            from src.database import get_job
            job = get_job(db, job_id)

            if not job or job.status != "queued":
                return  # Job already being processed or doesn't exist

            print(f"\n{'='*60}")
            print(f"📋 Processing Job: {job.id}")
            print(f"{'='*60}")
            print(f"Prompt: {job.prompt}")
            print(f"Status: {job.status} → processing")

            # Update status to processing
            update_job_status(db, job.id, "processing")

            try:
                # Initialize generator if not already done (thread-safe)
                self.initialize_generator()

                # Parse parameters from database
                use_prompt_enhancer = job.use_prompt_enhancer.lower() == "true"
                lora_scale = float(job.lora_scale)
                image_seed = int(job.image_seed) if job.image_seed else None
                video_seed = int(job.video_seed) if job.video_seed else None

                # Generate GIF (GPU will handle concurrent requests)
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
        """Main worker loop - continuous processing. As soon as a job completes, pick next one."""
        print("\n🚀 Worker started! Monitoring for new jobs...")
        print(f"   Concurrency: Up to {self.concurrency} job(s) running simultaneously")
        print(f"   Press Ctrl+C to stop\n")

        try:
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                active_futures = {}  # Track running jobs: {future: job_id}

                while True:
                    db = SessionLocal()
                    try:
                        # Check completed jobs and remove from active set
                        completed = [f for f in active_futures if f.done()]
                        for future in completed:
                            job_id = active_futures.pop(future)
                            try:
                                future.result()
                                print(f"✅ Job {job_id} completed")
                            except Exception as e:
                                print(f"❌ Job {job_id} failed: {e}")

                        # Calculate how many slots are available
                        available_slots = self.concurrency - len(active_futures)

                        if available_slots > 0:
                            # Get queued jobs to fill available slots
                            queued_jobs = get_queued_jobs(db)

                            if queued_jobs:
                                jobs_to_start = queued_jobs[:available_slots]

                                for job in jobs_to_start:
                                    print(f"🚀 Starting job {job.id} ({len(active_futures) + 1}/{self.concurrency} slots)")
                                    future = executor.submit(self.process_job, job.id)
                                    active_futures[future] = job.id

                        # Show status
                        if active_futures:
                            print(f"⏳ {len(active_futures)}/{self.concurrency} job(s) running...", end='\r')
                        else:
                            print(f"⏳ No jobs in queue. Checking again in {self.check_interval}s...", end='\r')

                        time.sleep(1)  # Check every second for completed jobs

                    finally:
                        db.close()

        except KeyboardInterrupt:
            print("\n\n⛔ Worker stopped by user")
        except Exception as e:
            print(f"\n\n❌ Worker crashed: {e}")
            raise


if __name__ == "__main__":
    worker = JobWorker()  # Uses Config.WORKER_CHECK_INTERVAL
    worker.run()
