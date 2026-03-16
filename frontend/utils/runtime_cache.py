"""Runtime-safe helpers for Streamlit caching.

These wrappers avoid touching Streamlit runtime APIs at import time.
"""

from __future__ import annotations

from functools import wraps


def _has_script_run_context() -> bool:
    """Return True when executing inside an active Streamlit run context."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def runtime_safe_cache_data(*cache_args, **cache_kwargs):
    """Lazily apply st.cache_data on first call when context is available."""

    def decorator(func):
        cached_callable = None

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal cached_callable
            if cached_callable is None:
                if _has_script_run_context():
                    try:
                        import streamlit as st

                        cached_callable = st.cache_data(*cache_args, **cache_kwargs)(func)
                    except Exception:
                        cached_callable = func
                else:
                    cached_callable = func
            return cached_callable(*args, **kwargs)

        return wrapper

    return decorator
