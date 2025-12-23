"""
Demo runner for DRUGVISTA
Starts both backend and frontend
"""
import subprocess
import time
import os
import sys
import threading
import webbrowser

def run_backend():
    """Run the FastAPI backend"""
    print("🚀 Starting backend server...")
    os.chdir("backend")
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
    except KeyboardInterrupt:
        print("Backend stopped")
    finally:
        os.chdir("..")

def run_frontend():
    """Run the Streamlit frontend"""
    print("🌐 Starting frontend...")
    time.sleep(3)  # Wait for backend to start
    os.chdir("frontend")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("Frontend stopped")
    finally:
        os.chdir("..")

def main():
    print("🧬 DRUGVISTA Demo Launcher")
    print("==========================")
    
    # Check if setup was run
    if not os.path.exists("backend/vector_index.faiss"):
        print("❌ Vector index not found. Please run setup first:")
        print("   python setup.py")
        return
    
    print("Starting DRUGVISTA demo...")
    print("Backend will start on: http://localhost:8000")
    print("Frontend will start on: http://localhost:8501")
    print("\nPress Ctrl+C to stop both servers")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Wait a moment then open browser
    time.sleep(5)
    try:
        webbrowser.open("http://localhost:8501")
    except:
        pass
    
    # Run frontend in main thread
    try:
        run_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down DRUGVISTA demo")

if __name__ == "__main__":
    main()