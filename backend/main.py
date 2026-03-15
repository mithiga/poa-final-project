"""
Backend entry point.

Runs the FastAPI application from backend/apis/main.py.
Start with: python main.py  (from the backend/ directory)
"""

import sys
import os

# Ensure the apis directory is on the Python path so absolute imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "apis"))

from apis.main import app  # noqa: F401 — re-export for uvicorn

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "apis.main:app",
        host="localhost",
        port=8000,
        reload=True
    )
