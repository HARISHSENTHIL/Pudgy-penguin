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

from src.database import SessionLocal, get_queued_jobs, update_job_status, claim_job
from src.image_generator import PudgyGIFGenerator
from src.config import Config


class JobWorker:
    """Background worker for processing GIF generation jobs."""

    def __init__(self, check_interval: int = None, concurrency: int = None, preload_models: bool = True):
        """
        Initialize the worker.

        Args:
            check_interval: Seconds to wait between database checks (defaults to Config)
            concurrency: Number of jobs to process in parallel (defaults to Config)
            preload_models: Pre-load models before starting worker loop (default True)
        """
        self.check_interval = check_interval or Config.WORKER_CHECK_INTERVAL
        self.concurrency = concurrency or Config.WORKER_CONCURRENCY
        self.db_lock = threading.Lock()  # SQLite thread safety lock
        self.preload_models = preload_models

        # Shared model instance pool (one per worker thread slot)
        self.model_pool = []
        self.model_pool_lock = threading.Lock()

        print("🔧 Initializing Job Worker...")
        Config.print_config()
        print(f"\n   Check interval: {self.check_interval}s")
        print(f"   Concurrency: {self.concurrency} job(s) in parallel")

        if self.concurrency > 1:
            print(f"   ⚡ Multi-threaded mode enabled!")

        if self.preload_models:
            print(f"\n🚀 Pre-loading {self.concurrency} model instance(s)...")
            print(f"   This will take a few minutes but ensures instant job pickup!\n")
            self._preload_all_models()

    def _preload_all_models(self):
        """
        Pre-load all model instances (one per concurrency slot).
        This prevents the first N jobs from blocking while loading models.
        """
        with self.model_pool_lock:
            for i in range(self.concurrency):
                print(f"   Loading model instance {i+1}/{self.concurrency}...")
                generator = PudgyGIFGenerator()
                self.model_pool.append(generator)
                print(f"   ✅ Instance {i+1} ready!\n")

            print(f"✅ All {self.concurrency} model instances pre-loaded and ready!")
            print(f"   Jobs can now start processing immediately without loading delays.\n")

    def get_generator(self):
        """
        Get an available generator from the pre-loaded pool.
        Thread-safe: each thread gets exclusive access to one generator.

        Returns:
            PudgyGIFGenerator instance from the pool
        """
        with self.model_pool_lock:
            if self.model_pool:
                # Pop a generator from the pool for this thread to use
                return self.model_pool.pop(0)
            else:
                # Fallback: create new instance if pool is empty (shouldn't happen)
                thread_id = threading.current_thread().name
                print(f"\n⚠️  [{thread_id}] Pool exhausted, creating new model instance...")
                return PudgyGIFGenerator()

    def return_generator(self, generator):
        """
        Return a generator back to the pool after job completes.

        Args:
            generator: PudgyGIFGenerator instance to return
        """
        with self.model_pool_lock:
            self.model_pool.append(generator)

    def process_job(self, job_id: str):
        """
        Process a single job. Thread-safe with separate DB session and model instance.

        Args:
            job_id: Job ID to process
        """
        thread_id = threading.current_thread().name

        # Get a generator from the pool (blocks if none available)
        generator = self.get_generator()

        # Create separate DB session for this thread
        db = SessionLocal()
        try:
            from src.database import get_job
            job = get_job(db, job_id)

            if not job or job.status != "processing":
                return  # Job not found or not properly claimed

            print(f"\n{'='*60}")
            print(f"📋 [{thread_id}] Processing Job: {job.id}")
            print(f"{'='*60}")
            print(f"Prompt: {job.prompt}")

            try:
                # Parse parameters from database
                use_prompt_enhancer = job.use_prompt_enhancer.lower() == "true"
                lora_scale = float(job.lora_scale)
                image_seed = int(job.image_seed) if job.image_seed else None
                video_seed = int(job.video_seed) if job.video_seed else None

                # Generate GIF using pre-loaded generator (no loading delay!)
                result = generator.generate_gif(
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

                print(f"\n✅ [{thread_id}] Job {job.id} completed successfully!")
                print(f"   GIF: {gif_path}")

            except Exception as e:
                error_msg = str(e)
                print(f"\n❌ [{thread_id}] Job {job.id} failed: {error_msg}")

                # Update job with failed status
                update_job_status(db, job.id, "failed", error_message=error_msg)

        finally:
            # Return generator to pool for reuse
            self.return_generator(generator)
            db.close()

    def run(self):
        """Main worker loop - continuous processing. Picks up jobs immediately when slots available."""
        print("\n🚀 Worker started! Monitoring for new jobs...")
        print(f"   Concurrency: Up to {self.concurrency} job(s) running simultaneously")
        print(f"   ⚡ Fast polling mode - picks up jobs immediately")
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
                                print(f"\n✅ Job {job_id} completed - slot now available!")
                            except Exception as e:
                                print(f"\n❌ Job {job_id} failed: {e}")

                        # Calculate how many slots are available
                        available_slots = self.concurrency - len(active_futures)

                        if available_slots > 0:
                            # Get queued jobs to fill available slots
                            # Use lock to ensure thread-safe SQLite access
                            with self.db_lock:
                                queued_jobs = get_queued_jobs(db, limit=available_slots)

                                if queued_jobs:
                                    for job in queued_jobs:
                                        # Atomically claim the job (prevents race conditions)
                                        if claim_job(db, job.id):
                                            active_count = len(active_futures) + 1
                                            print(f"\n🚀 Starting job {job.id} (slot {active_count}/{self.concurrency})")
                                            future = executor.submit(self.process_job, job.id)
                                            active_futures[future] = job.id
                                        else:
                                            print(f"⚠️  Job {job.id} already claimed by another thread")

                        # Show status (don't spam if running jobs)
                        if active_futures:
                            print(f"⏳ {len(active_futures)}/{self.concurrency} job(s) running...", end='\r')
                        else:
                            print(f"⏳ No jobs in queue. Waiting for new jobs...", end='\r')

                        # Fast polling when slots available, slower when all busy
                        if available_slots > 0:
                            time.sleep(0.1)  # Check every 100ms when slots available
                        else:
                            time.sleep(0.5)  # Check every 500ms when all slots busy

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
