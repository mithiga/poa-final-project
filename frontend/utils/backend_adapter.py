"""Embedded backend adapter for Streamlit deployments.

When BACKEND_API_URL is unset, requests sent to the configured API base URL are
handled in-process by calling the backend service layer directly. This lets the
frontend and backend run as a single Streamlit app on Streamlit Community Cloud.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"


if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from apis.pydantic_models import (  # noqa: E402
    HyperparameterTuningRequest,
    HyperparameterTuningResponse,
    ModelHyperparameters,
    PredictionRequest,
    PredictionResponse,
    SUPPORTED_MODELS,
    SystemStatusResponse,
    TrainAllRequest,
    TrainAllResponse,
    TrainRequest,
    TrainResponse,
)
from apis.services import ForecastService, MarketDataService, SystemService, TrainingService  # noqa: E402
from apis.ml_pipeline import get_all_hyperparameters, get_model_hyperparameters, load_model_metadata  # noqa: E402


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
    return api_base_url, "embedded"


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

    return _json_response(404, {"detail": f"Unknown endpoint: {path}"})


def _handle_post(path: str, json_body: Dict[str, Any]) -> EmbeddedResponse:
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
        )
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
        )
        payload = TrainAllResponse(
            ticker=request.ticker,
            evaluation=results["evaluation"],
            training=results["training"],
            status="Success",
        ).model_dump()
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