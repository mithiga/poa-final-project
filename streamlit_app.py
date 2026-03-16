"""Root Streamlit entrypoint for Streamlit Community Cloud."""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
BACKEND_ROOT = PROJECT_ROOT / "backend"

for _p in (str(FRONTEND_ROOT), str(BACKEND_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

def _ensure_local_namespace_package(name: str, directory: Path) -> None:
    """Ensure imports like `from utils...` resolve to the frontend local package."""
    existing = sys.modules.get(name)
    if existing is not None and hasattr(existing, "__path__"):
        paths = [str(p) for p in getattr(existing, "__path__", [])]
        if str(directory) in paths:
            return

    pkg = types.ModuleType(name)
    pkg.__path__ = [str(directory)]
    pkg.__package__ = name
    sys.modules[name] = pkg


_ensure_local_namespace_package("utils", FRONTEND_ROOT / "utils")
_ensure_local_namespace_package("pages", FRONTEND_ROOT / "pages")

# Execute app.py via runpy so each Streamlit rerun gets a fresh main script context.
runpy.run_path(str(FRONTEND_ROOT / "app.py"), run_name="__main__")