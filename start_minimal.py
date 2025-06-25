#!/usr/bin/env python3
"""
Minimal ChatGPT Analytics Startup Script

This bypasses npm entirely and uses only Python + basic HTML/JS
"""

import subprocess
import sys
import os
import webbrowser
import time
from pathlib import Path

def install_minimal_deps():
    """Install only the essential Python dependencies"""
    print("📦 Installing minimal Python dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "fastapi", "uvicorn[standard]", "python-multipart"
        ])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    os.environ["PYTHONPATH"] = str(Path.cwd())
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "api.app.main_minimal:app", 
        "--reload", "--port", "8000"
    ]
    return subprocess.Popen(cmd)

def open_frontend():
    """Open the HTML frontend in browser"""
    html_path = Path.cwd() / "minimal.html"
    if html_path.exists():
        print("🌐 Opening frontend in browser...")
        webbrowser.open(f"file://{html_path}")
    else:
        print("❌ minimal.html not found")

def main():
    print("🎯 ChatGPT Analytics - Minimal Version")
    print("=" * 50)
    
    # Install dependencies
    install_minimal_deps()
    
    # Start backend
    backend_process = start_backend()
    
    # Wait for backend to start
    print("⏳ Waiting for backend to start...")
    time.sleep(3)
    
    # Open frontend
    open_frontend()
    
    print("\n✨ Setup complete!")
    print("📍 Backend: http://127.0.0.1:8000")
    print("📍 Frontend: Opens automatically in browser")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        backend_process.terminate()
        backend_process.wait()

if __name__ == "__main__":
    main() 