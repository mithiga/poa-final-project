"""Embedded backend adapter for Streamlit deployments.

When BACKEND_API_URL is unset, requests sent to the configured API base URL are
handled in-process by calling the backend service layer directly. This lets the
frontend and backend run as a single Streamlit app on Streamlit Community Cloud.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"


if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


_BACKEND_LOADED = False
_BACKEND_IMPORT_ERROR: Optional[Exception] = None

# These names are populated by _ensure_backend_loaded() to avoid import-time crashes
# on environments where embedded backend imports fail.
HyperparameterTuningRequest = None
HyperparameterTuningResponse = None
ModelHyperparameters = None
PredictionRequest = None
PredictionResponse = None
SUPPORTED_MODELS = []
SystemStatusResponse = None
TrainAllRequest = None
TrainAllResponse = None
TrainRequest = None
TrainResponse = None
ForecastService = None
MarketDataService = None
SystemService = None
TrainingService = None
get_all_hyperparameters = None
get_model_hyperparameters = None
load_model_metadata = None


def _ensure_backend_loaded() -> bool:
    """Lazily import backend dependencies used for embedded mode."""
    global _BACKEND_LOADED, _BACKEND_IMPORT_ERROR
    global HyperparameterTuningRequest, HyperparameterTuningResponse, ModelHyperparameters
    global PredictionRequest, PredictionResponse, SUPPORTED_MODELS, SystemStatusResponse
    global TrainAllRequest, TrainAllResponse, TrainRequest, TrainResponse
    global ForecastService, MarketDataService, SystemService, TrainingService
    global get_all_hyperparameters, get_model_hyperparameters, load_model_metadata

    if _BACKEND_LOADED:
        return True
    if _BACKEND_IMPORT_ERROR is not None:
        return False

    try:
        from apis.pydantic_models import (  # type: ignore
            HyperparameterTuningRequest as _HyperparameterTuningRequest,
            HyperparameterTuningResponse as _HyperparameterTuningResponse,
            ModelHyperparameters as _ModelHyperparameters,
            PredictionRequest as _PredictionRequest,
            PredictionResponse as _PredictionResponse,
            SUPPORTED_MODELS as _SUPPORTED_MODELS,
            SystemStatusResponse as _SystemStatusResponse,
            TrainAllRequest as _TrainAllRequest,
            TrainAllResponse as _TrainAllResponse,
            TrainRequest as _TrainRequest,
            TrainResponse as _TrainResponse,
        )
        from apis.services import ForecastService as _ForecastService  # type: ignore
        from apis.services import MarketDataService as _MarketDataService  # type: ignore
        from apis.services import SystemService as _SystemService  # type: ignore
        from apis.services import TrainingService as _TrainingService  # type: ignore
        from apis.ml_pipeline import get_all_hyperparameters as _get_all_hyperparameters  # type: ignore
        from apis.ml_pipeline import get_model_hyperparameters as _get_model_hyperparameters  # type: ignore
        from apis.ml_pipeline import load_model_metadata as _load_model_metadata  # type: ignore

        HyperparameterTuningRequest = _HyperparameterTuningRequest
        HyperparameterTuningResponse = _HyperparameterTuningResponse
        ModelHyperparameters = _ModelHyperparameters
        PredictionRequest = _PredictionRequest
        PredictionResponse = _PredictionResponse
        SUPPORTED_MODELS = _SUPPORTED_MODELS
        SystemStatusResponse = _SystemStatusResponse
        TrainAllRequest = _TrainAllRequest
        TrainAllResponse = _TrainAllResponse
        TrainRequest = _TrainRequest
        TrainResponse = _TrainResponse
        ForecastService = _ForecastService
        MarketDataService = _MarketDataService
        SystemService = _SystemService
        TrainingService = _TrainingService
        get_all_hyperparameters = _get_all_hyperparameters
        get_model_hyperparameters = _get_model_hyperparameters
        load_model_metadata = _load_model_metadata

        _BACKEND_LOADED = True
        return True
    except Exception as exc:
        _BACKEND_IMPORT_ERROR = exc
        return False


# ─── Adapter-local async Train All job manager ──────────────────────────────
_train_all_jobs: Dict[str, Dict[str, Any]] = {}
_train_all_jobs_lock = threading.Lock()


def _start_train_all_job(
    ticker: str,
    start_date: str,
    end_date: str,
    train_size: float = 0.8,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """Launch Train All in a background thread, return a job descriptor."""
    job_id = str(uuid.uuid4())

    job_state: Dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "ticker": ticker,
        "total_tickers": 1,
        "total_models": 0,
        "total_units": 0,
        "completed_units": 0,
        "current_ticker": None,
        "cancel_requested": False,
        "result": None,
        "error": None,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "finished_at": None,
    }

    with _train_all_jobs_lock:
        _train_all_jobs[job_id] = job_state

    def _run() -> None:
        try:
            with _train_all_jobs_lock:
                j = _train_all_jobs.get(job_id)
                if not j:
                    return
                if j.get("cancel_requested"):
                    j["status"] = "canceled"
                    j["finished_at"] = datetime.utcnow().isoformat() + "Z"
                    return
                j["status"] = "running"
                j["current_ticker"] = ticker

            result = TrainingService.train_all(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                train_size=train_size,
                force_retrain=force_retrain,
            )

            with _train_all_jobs_lock:
                j = _train_all_jobs.get(job_id)
                if not j:
                    return
                if j.get("cancel_requested") and j.get("status") != "completed":
                    j["status"] = "canceled"
                    j["finished_at"] = datetime.utcnow().isoformat() + "Z"
                    return
                j["status"] = "completed"
                j["result"] = result
                j["current_ticker"] = None
                j["finished_at"] = datetime.utcnow().isoformat() + "Z"
        except Exception as exc:
            with _train_all_jobs_lock:
                j = _train_all_jobs.get(job_id)
                if not j:
                    return
                j["status"] = "failed"
                j["error"] = str(exc)
                j["current_ticker"] = None
                j["finished_at"] = datetime.utcnow().isoformat() + "Z"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "queued"}


def _get_train_all_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _train_all_jobs_lock:
        j = _train_all_jobs.get(job_id)
        return dict(j) if j else None


def _cancel_train_all_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _train_all_jobs_lock:
        j = _train_all_jobs.get(job_id)
        if not j:
            return None
        j["cancel_requested"] = True
        if j.get("status") == "queued":
            j["status"] = "canceled"
            j["finished_at"] = datetime.utcnow().isoformat() + "Z"
        return dict(j)


class EmbeddedResponse:
    """Minimal requests.Response-like object for in-process API calls."""

    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload
        self.ok = 200 <= status_code < 300
        if isinstance(payload, (dict, list)):
            self.text = json.dumps(payload, default=str)
        else:
            self.text = str(payload)

    def json(self) -> Any:
        return self._payload


def configure_backend(default_base_url: str = "http://localhost:8000") -> tuple[str, str]:
    """Configure frontend/backend integration mode.

    If BACKEND_API_URL is provided, the Streamlit frontend will use that remote
    HTTP backend. Otherwise requests to the default base URL are intercepted and
    executed in-process.
    """
    api_base_url = os.getenv("BACKEND_API_URL", "").strip() or default_base_url
    if os.getenv("BACKEND_API_URL", "").strip():
        return api_base_url, "remote"

    install_requests_adapter(api_base_url)
    if _ensure_backend_loaded():
        return api_base_url, "embedded"
    return api_base_url, "embedded-unavailable"


def install_requests_adapter(api_base_url: str) -> None:
    """Patch requests.get/post so frontend API calls can run in-process."""
    already_patched = getattr(requests, "_embedded_backend_api_base_url", None)
    if already_patched == api_base_url:
        return

    original_get = getattr(requests, "_embedded_backend_original_get", requests.get)
    original_post = getattr(requests, "_embedded_backend_original_post", requests.post)

    def patched_get(url: str, *args: Any, **kwargs: Any):
        if _is_embedded_api_call(url, api_base_url):
            return dispatch_request("GET", url, params=kwargs.get("params"))
        return original_get(url, *args, **kwargs)

    def patched_post(url: str, *args: Any, **kwargs: Any):
        if _is_embedded_api_call(url, api_base_url):
            return dispatch_request("POST", url, json_body=kwargs.get("json"), params=kwargs.get("params"))
        return original_post(url, *args, **kwargs)

    requests._embedded_backend_original_get = original_get
    requests._embedded_backend_original_post = original_post
    requests._embedded_backend_api_base_url = api_base_url
    requests.get = patched_get
    requests.post = patched_post


def _is_embedded_api_call(url: str, api_base_url: str) -> bool:
    return url.startswith(api_base_url)


def _merge_params(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    parsed = urlparse(url)
    merged: Dict[str, Any] = {
        key: values[-1] if len(values) == 1 else values
        for key, values in parse_qs(parsed.query).items()
    }
    if params:
        merged.update(params)
    return merged


def _path_from_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path or "/"


def _json_response(status_code: int, payload: Any) -> EmbeddedResponse:
    return EmbeddedResponse(status_code=status_code, payload=payload)


def _error_response(exc: Exception) -> EmbeddedResponse:
    if isinstance(exc, FileNotFoundError):
        return _json_response(404, {"detail": str(exc)})
    if isinstance(exc, ValueError):
        return _json_response(400, {"detail": str(exc)})
    return _json_response(500, {"detail": str(exc)})


def dispatch_request(method: str, url: str, params: Optional[Dict[str, Any]] = None,
                     json_body: Optional[Dict[str, Any]] = None) -> EmbeddedResponse:
    path = _path_from_url(url)
    merged_params = _merge_params(url, params)

    try:
        if method == "GET":
            return _handle_get(path, merged_params)
        if method == "POST":
            return _handle_post(path, json_body or {})
        return _json_response(405, {"detail": f"Unsupported method: {method}"})
    except Exception as exc:
        return _error_response(exc)


def _handle_get(path: str, params: Dict[str, Any]) -> EmbeddedResponse:
    if not _ensure_backend_loaded():
        return _json_response(500, {"detail": f"Embedded backend unavailable: {_BACKEND_IMPORT_ERROR}"})

    if path == "/status":
        payload = SystemStatusResponse(**SystemService.get_status()).model_dump()
        return _json_response(200, payload)

    if path == "/available_models":
        return _json_response(200, {"models": SUPPORTED_MODELS})

    if path == "/available_tickers":
        return _json_response(200, {"tickers": SystemService.get_available_tickers()})

    if path == "/training-period":
        ticker = params.get("ticker")
        if not ticker:
            return _json_response(400, {"detail": "ticker is required"})
        return _json_response(200, SystemService.get_training_period(str(ticker)))

    if path == "/model-cutoff-date":
        ticker = params.get("ticker")
        model = params.get("model")
        if not ticker or not model:
            return _json_response(400, {"detail": "ticker and model are required"})
        return _json_response(200, _get_model_cutoff_date(str(ticker), str(model)))

    if path == "/market-data":
        ticker = params.get("ticker")
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        if not ticker or not start_date or not end_date:
            return _json_response(400, {"detail": "ticker, start_date, and end_date are required"})
        df = MarketDataService.get_data(str(ticker), str(start_date), str(end_date))
        df_reset = df.reset_index()
        df_reset["Date"] = pd.to_datetime(df_reset["Date"]).dt.strftime("%Y-%m-%d")
        return _json_response(200, {"ticker": ticker, "data": df_reset.to_dict(orient="records")})

    if path == "/market-overview":
        ticker = params.get("ticker")
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        if not ticker or not start_date or not end_date:
            return _json_response(400, {"detail": "ticker, start_date, and end_date are required"})
        payload = MarketDataService.get_market_overview(
            ticker=str(ticker),
            start_date=str(start_date),
            end_date=str(end_date),
        )
        return _json_response(200, payload)

    if path == "/hyperparameters":
        model = str(params.get("model", ""))
        if model not in SUPPORTED_MODELS:
            return _json_response(400, {"detail": f"Model must be one of {SUPPORTED_MODELS}"})
        payload = ModelHyperparameters(model=model, hyperparameters=get_model_hyperparameters(model)).model_dump()
        return _json_response(200, payload)

    if path == "/all_hyperparameters":
        return _json_response(200, get_all_hyperparameters())

    if path == "/train_all_async/status":
        job_id = params.get("job_id")
        if not job_id:
            return _json_response(400, {"detail": "job_id is required"})
        state = _get_train_all_job(str(job_id))
        if not state:
            return _json_response(404, {"detail": "Job not found"})
        return _json_response(200, state)

    return _json_response(404, {"detail": f"Unknown endpoint: {path}"})


def _handle_post(path: str, json_body: Dict[str, Any]) -> EmbeddedResponse:
    if not _ensure_backend_loaded():
        return _json_response(500, {"detail": f"Embedded backend unavailable: {_BACKEND_IMPORT_ERROR}"})

    if path == "/train_all_async/cancel":
        job_id = json_body.get("job_id")
        if not job_id:
            return _json_response(400, {"detail": "job_id is required"})
        state = _cancel_train_all_job(str(job_id))
        if not state:
            return _json_response(404, {"detail": "Job not found"})
        return _json_response(
            200,
            {
                "job_id": str(job_id),
                "status": state.get("status"),
                "cancel_requested": bool(state.get("cancel_requested", False)),
            },
        )

    if path == "/predict":
        request = PredictionRequest(**json_body)
        dates, predictions, metrics = ForecastService.get_predictions(
            ticker=request.ticker,
            model_type=request.model_type,
            days=request.days,
        )
        payload = PredictionResponse(
            ticker=request.ticker,
            model_used=request.model_type,
            dates=dates,
            predictions=predictions,
            metrics=metrics,
            status="Success",
        ).model_dump()
        return _json_response(200, payload)

    if path == "/train":
        request = TrainRequest(**json_body)
        result = TrainingService.train_single(
            ticker=request.ticker,
            model=request.model,
            start_date=request.start_date,
            end_date=request.end_date,
            train_size=request.train_size,
            force_retrain=request.force_retrain,
        )

        if isinstance(result, dict) and result.get("skipped"):
            payload = TrainResponse(
                ticker=request.ticker,
                model=request.model,
                rmse=0.0,
                mae=0.0,
                mape=0.0,
                status="Skipped",
                message=result.get("message", f"{request.model} skipped for {request.ticker}"),
            ).model_dump()
            return _json_response(200, payload)

        payload = TrainResponse(
            ticker=request.ticker,
            model=request.model,
            rmse=result.rmse,
            mae=result.mae,
            mape=result.mape,
            status="Success",
            message=f"{request.model} trained successfully for {request.ticker}",
        ).model_dump()
        return _json_response(200, payload)

    if path == "/train_all":
        request = TrainAllRequest(**json_body)
        results = TrainingService.train_all(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            train_size=request.train_size,
            force_retrain=request.force_retrain,
        )
        payload = TrainAllResponse(
            ticker=request.ticker,
            evaluation=results["evaluation"],
            training=results["training"],
            by_ticker=results.get("by_ticker"),
            summary=results.get("summary"),
            status="Success",
        ).model_dump()
        return _json_response(200, payload)

    if path == "/train_all_async/start":
        request = TrainAllRequest(**json_body)
        payload = _start_train_all_job(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            train_size=request.train_size,
            force_retrain=request.force_retrain,
        )
        return _json_response(200, payload)

    if path == "/train_with_tuning":
        request = HyperparameterTuningRequest(**json_body)
        result = TrainingService.train_with_hyperparameters(
            ticker=request.ticker,
            model=request.model,
            start_date=request.start_date,
            end_date=request.end_date,
            train_size=request.train_size,
            hyperparameters=request.hyperparameters,
        )
        selected_hyperparameters = request.hyperparameters
        if getattr(result, "artifacts", None) and isinstance(result.artifacts, dict):
            selected_hyperparameters = result.artifacts.get("best_hyperparameters", request.hyperparameters)

        payload = HyperparameterTuningResponse(
            ticker=request.ticker,
            model=request.model,
            hyperparameters_used=selected_hyperparameters,
            rmse=result.rmse,
            mae=result.mae,
            mape=result.mape,
            status="Success",
            message=f"{request.model} trained successfully with custom hyperparameters for {request.ticker}",
        ).model_dump()
        return _json_response(200, payload)

    return _json_response(404, {"detail": f"Unknown endpoint: {path}"})


def _get_model_cutoff_date(ticker: str, model: str) -> Dict[str, Any]:
    if not _ensure_backend_loaded():
        return {
            "ticker": ticker,
            "model": model,
            "cutoff_date": None,
            "error": f"Embedded backend unavailable: {_BACKEND_IMPORT_ERROR}",
        }

    metadata = load_model_metadata()
    ticker_symbol = ticker.replace("=X", "").replace("=", "").replace("/", "")

    if ticker_symbol in metadata:
        models_meta = metadata[ticker_symbol].get("models", {})
        model_meta = models_meta.get(model) or models_meta.get(model.upper())
        if model_meta:
            return {
                "ticker": ticker,
                "model": model,
                "cutoff_date": model_meta.get("training_cutoff_date"),
                "training_period_start": model_meta.get("training_period_start"),
                "training_period_end": model_meta.get("training_period_end"),
            }

    return {
        "ticker": ticker,
        "model": model,
        "cutoff_date": None,
        "error": "Model not found",
    }