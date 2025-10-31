"""
Main entry point to start both API server and Worker together.
"""

import multiprocessing
import uvicorn
from src.worker import JobWorker
from src.config import Config


def start_api():
    """Start FastAPI server."""
    print("🌐 Starting API server...")
    uvicorn.run(
        "src.api:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=False  # Disable reload in production mode
    )


def start_worker():
    """Start background worker."""
    print("⚙️  Starting Worker...")
    worker = JobWorker()
    worker.run()


def main():
    """Start both API and Worker in separate processes."""
    print("\n" + "="*60)
    print("🐧 PUDGY GIF GENERATOR - STARTING APPLICATION")
    print("="*60 + "\n")

    # Print configuration
    Config.print_config()

    print("\n📍 API will be available at:")
    print(f"   http://{Config.API_HOST}:{Config.API_PORT}")
    print(f"   Docs: http://{Config.API_HOST}:{Config.API_PORT}/docs")
    print()

    # Create processes for API and Worker
    api_process = multiprocessing.Process(target=start_api, name="API-Server")
    worker_process = multiprocessing.Process(target=start_worker, name="Worker")

    try:
        # Start both processes
        api_process.start()
        worker_process.start()

        print("✅ Application started successfully!")
        print("   Press Ctrl+C to stop both services\n")

        # Wait for processes
        api_process.join()
        worker_process.join()

    except KeyboardInterrupt:
        print("\n\n⛔ Shutting down...")
        api_process.terminate()
        worker_process.terminate()
        api_process.join()
        worker_process.join()
        print("✅ Application stopped\n")


if __name__ == "__main__":
    # Required for multiprocessing on Windows/Mac
    multiprocessing.set_start_method('spawn', force=True)
    main()
