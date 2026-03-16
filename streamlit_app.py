"""Root Streamlit entrypoint for Streamlit Community Cloud."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
BACKEND_ROOT = PROJECT_ROOT / "backend"

for _p in (str(FRONTEND_ROOT), str(BACKEND_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Execute app.py directly (without module import caching) so Streamlit reruns
# always execute the frontend script fresh.
app_path = FRONTEND_ROOT / "app.py"
app_code = compile(app_path.read_text(encoding="utf-8"), str(app_path), "exec")
exec(app_code, {"__name__": "__main__", "__file__": str(app_path)})