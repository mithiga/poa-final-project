"""
Service layer — bridges the API endpoints and the ML pipeline.
"""

import os
import threading
import uuid
from datetime import datetime
from typing import Dict, Optional, List, Any

import pandas as pd

from .ml_pipeline import (
    fetch_data, clean_data, feature_engineering, split_data,
    train_model, train_all_models, generate_forecast,
    load_model_metadata, get_ticker_folder, MODELS_BASE_DIR,
    MODEL_TRAINERS, train_model_with_hyperparameters
)


class ForecastService:
    """Handles prediction requests using pre-trained models."""

    @staticmethod
    def get_predictions(ticker: str, model_type: str, days: int) -> tuple:
        """
        Load a pre-trained model and generate a forecast.

        Returns:
            (dates, predictions, metrics) tuple
        """
        # Determine the last known date from metadata
        metadata = load_model_metadata()
        ticker_symbol = ticker.replace("=X", "").replace("=", "").replace("/", "")

        cutoff_date = None
        if ticker_symbol in metadata:
            models_meta = metadata[ticker_symbol].get("models", {})
            model_meta = models_meta.get(model_type) or models_meta.get(model_type.upper())
            if model_meta:
                cutoff_date = model_meta.get("training_cutoff_date")

        if cutoff_date is None:
            # Fallback: use today minus 1 day
            cutoff_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        result = generate_forecast(
            model=model_type,
            ticker=ticker,
            start_date=cutoff_date,
            steps=days
        )

        metrics = {"RMSE": 0.0, "MAE": 0.0}

        return result["dates"], result["predictions"], metrics


class TrainingService:
    """Handles model training requests."""

    _train_all_jobs: Dict[str, Dict[str, Any]] = {}
    _train_all_jobs_lock = threading.Lock()

    @staticmethod
    def _train_all_model_count() -> int:
        # `SARIMA` is an alias of `SARIMAX` in MODEL_TRAINERS and is skipped in train_all_models.
        return len([name for name in MODEL_TRAINERS.keys() if name != "SARIMA"])

    @staticmethod
    def _run_train_all_job(
        job_id: str,
        ticker: str,
        start_date: str,
        end_date: str,
        train_size: float,
        force_retrain: bool,
    ) -> None:
        try:
            with TrainingService._train_all_jobs_lock:
                job = TrainingService._train_all_jobs.get(job_id)
                if not job:
                    return
                if job.get("cancel_requested"):
                    job["status"] = "canceled"
                    job["finished_at"] = datetime.utcnow().isoformat() + "Z"
                    return
                job["status"] = "running"
                job["current_ticker"] = ticker

            result = TrainingService.train_all(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                train_size=train_size,
                force_retrain=force_retrain,
            )

            with TrainingService._train_all_jobs_lock:
                job = TrainingService._train_all_jobs.get(job_id)
                if not job:
                    return
                if job.get("cancel_requested") and job.get("status") != "completed":
                    job["status"] = "canceled"
                    job["finished_at"] = datetime.utcnow().isoformat() + "Z"
                    return

                job["status"] = "completed"
                job["result"] = result
                job["completed_units"] = int(job.get("total_units", 0))
                job["current_ticker"] = None
                job["finished_at"] = datetime.utcnow().isoformat() + "Z"
        except Exception as exc:
            with TrainingService._train_all_jobs_lock:
                job = TrainingService._train_all_jobs.get(job_id)
                if not job:
                    return
                job["status"] = "failed"
                job["error"] = str(exc)
                job["current_ticker"] = None
                job["finished_at"] = datetime.utcnow().isoformat() + "Z"

    @staticmethod
    def start_train_all_job(
        ticker: str,
        start_date: str,
        end_date: str,
        train_size: float = 0.8,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        """Start Train All in a background thread and return a job descriptor."""
        job_id = str(uuid.uuid4())
        total_models = TrainingService._train_all_model_count()

        job_state = {
            "job_id": job_id,
            "status": "queued",
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "train_size": float(train_size),
            "force_retrain": bool(force_retrain),
            "total_tickers": 1,
            "total_models": total_models,
            "total_units": total_models,
            "completed_units": 0,
            "current_ticker": None,
            "cancel_requested": False,
            "result": None,
            "error": None,
            "started_at": datetime.utcnow().isoformat() + "Z",
            "finished_at": None,
        }

        with TrainingService._train_all_jobs_lock:
            TrainingService._train_all_jobs[job_id] = job_state

        thread = threading.Thread(
            target=TrainingService._run_train_all_job,
            args=(job_id, ticker, start_date, end_date, train_size, force_retrain),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id, "status": "queued"}

    @staticmethod
    def get_train_all_job(job_id: str) -> Optional[Dict[str, Any]]:
        """Return the current state of a Train All async job."""
        with TrainingService._train_all_jobs_lock:
            job = TrainingService._train_all_jobs.get(job_id)
            if not job:
                return None
            return dict(job)

    @staticmethod
    def cancel_train_all_job(job_id: str) -> Optional[Dict[str, Any]]:
        """Request cancellation of a Train All async job."""
        with TrainingService._train_all_jobs_lock:
            job = TrainingService._train_all_jobs.get(job_id)
            if not job:
                return None
            job["cancel_requested"] = True
            if job.get("status") == "queued":
                job["status"] = "canceled"
                job["finished_at"] = datetime.utcnow().isoformat() + "Z"
            return dict(job)

    @staticmethod
    def train_single(ticker: str, model: str, start_date: str,
                     end_date: str, train_size: float = 0.8,
                     force_retrain: bool = False):
        """Train a single model and return evaluation metrics."""
        # `force_retrain` is accepted for API/frontend compatibility.
        # Current pipeline always retrains for the requested window.
        _ = force_retrain
        result = train_model(
            start_date=start_date,
            end_date=end_date,
            ticker=ticker,
            train_size=train_size,
            model=model
        )
        return result

    @staticmethod
    def train_all(ticker: str, start_date: str, end_date: str,
                  train_size: float = 0.8,
                  force_retrain: bool = False) -> dict:
        """Train all supported models."""
        # `force_retrain` is accepted for API/frontend compatibility.
        # Current pipeline always retrains for the requested window.
        _ = force_retrain
        return train_all_models(
            start_date=start_date,
            end_date=end_date,
            ticker=ticker,
            train_size=train_size
        )

    @staticmethod
    def train_with_hyperparameters(ticker: str, model: str, start_date: str,
                                   end_date: str, train_size: float = 0.8,
                                   hyperparameters: dict = None):
        """Train a model with custom hyperparameters."""
        if hyperparameters is None:
            hyperparameters = {}
        result = train_model_with_hyperparameters(
            start_date=start_date,
            end_date=end_date,
            ticker=ticker,
            train_size=train_size,
            model=model,
            hyperparameters=hyperparameters
        )
        return result


class MarketDataService:
    """Handles market data requests."""

    DEFAULT_SENTIMENT_MODELS = ["LSTM", "GRU", "LightGBM", "ARIMA", "SARIMA", "RandomForest"]

    @staticmethod
    def get_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch and clean market data."""
        try:
            data = fetch_data(start_date, end_date, ticker)
            data = clean_data(data)
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data for {ticker}: {e}")

    @staticmethod
    def _compute_dynamic_flatish_threshold(df: pd.DataFrame) -> float:
        """Derive a pair-specific flatish threshold from recent volatility and range."""
        if df is None or df.empty or "Close" not in df.columns:
            return 0.001

        lookback_df = df.tail(30).copy()
        close = pd.to_numeric(lookback_df.get("Close"), errors="coerce")

        if close.dropna().empty:
            return 0.001

        returns = close.pct_change().dropna()
        daily_vol = float(returns.std()) if not returns.empty else 0.0

        atr_like = 0.0
        if all(col in lookback_df.columns for col in ["High", "Low", "Close"]):
            high = pd.to_numeric(lookback_df.get("High"), errors="coerce")
            low = pd.to_numeric(lookback_df.get("Low"), errors="coerce")
            valid_close = close.replace(0, pd.NA)
            range_ratio = ((high - low) / valid_close).dropna()
            if not range_ratio.empty:
                atr_like = float(range_ratio.mean())

        raw_threshold = (0.55 * daily_vol) + (0.35 * atr_like)
        return min(0.015, max(0.001, raw_threshold))

    @staticmethod
    def _trend_signal(reference_price: Optional[float], predicted_price: Optional[float], flat_threshold: float) -> tuple:
        if reference_price in (None, 0) or predicted_price is None:
            return "No Signal", None, 0

        pct_change = (predicted_price - reference_price) / reference_price
        if pct_change > flat_threshold:
            return "Bullish", float(pct_change), 1
        if pct_change < -flat_threshold:
            return "Bearish", float(pct_change), -1
        return "Flatish", float(pct_change), 0

    @staticmethod
    def get_market_overview(ticker: str, start_date: str, end_date: str, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fetch market data and compute volatility-aware model sentiment."""
        df = MarketDataService.get_data(ticker, start_date, end_date)
        df_reset = df.reset_index()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df_reset.columns:
                df_reset[col] = pd.to_numeric(df_reset[col], errors="coerce")

        reference_price = None
        if "Close" in df_reset.columns:
            closes = df_reset["Close"].dropna()
            if not closes.empty:
                reference_price = float(closes.iloc[-1])

        threshold = MarketDataService._compute_dynamic_flatish_threshold(df_reset)

        if models:
            selected_models = [m for m in models if m in MODEL_TRAINERS]
        else:
            selected_models = [m for m in MarketDataService.DEFAULT_SENTIMENT_MODELS if m in MODEL_TRAINERS]

        model_rows = []
        scores = []

        for model_name in selected_models:
            predicted_price = None
            try:
                _, preds, _ = ForecastService.get_predictions(
                    ticker=ticker,
                    model_type=model_name,
                    days=1,
                )
                if preds:
                    predicted_price = float(preds[0])
            except Exception:
                predicted_price = None

            signal, change_pct, score = MarketDataService._trend_signal(reference_price, predicted_price, threshold)
            scores.append(score)
            model_rows.append(
                {
                    "model": model_name,
                    "signal": signal,
                    "predicted_price": predicted_price,
                    "change_pct": change_pct,
                }
            )

        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score > 0.25:
                overall = "Bullish"
            elif avg_score < -0.25:
                overall = "Bearish"
            else:
                overall = "Flatish"
        else:
            overall = "No Signal"

        df_reset["Date"] = pd.to_datetime(df_reset["Date"]).dt.strftime("%Y-%m-%d")

        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "data": df_reset.to_dict(orient="records"),
            "sentiment": {
                "overall": overall,
                "flat_threshold_pct": float(threshold),
                "reference_price": reference_price,
                "models": model_rows,
            },
            "status": "Success",
        }


class SystemService:
    """Provides system status and metadata."""

    @staticmethod
    def get_status() -> dict:
        """Return system status and available trained models."""
        metadata = load_model_metadata()
        tickers_trained = list(metadata.keys())
        return {
            "status": "operational",
            "models_available": list(MODEL_TRAINERS.keys()),
            "tickers_trained": tickers_trained,
            "message": f"{len(tickers_trained)} ticker(s) have trained models"
        }

    @staticmethod
    def get_available_tickers() -> list:
        """Return list of tickers that have trained models."""
        metadata = load_model_metadata()
        return list(metadata.keys())

    @staticmethod
    def get_training_period(ticker: str) -> dict:
        """Return the training period for a given ticker."""
        metadata = load_model_metadata()
        ticker_symbol = ticker.replace("=X", "").replace("=", "").replace("/", "")
        if ticker_symbol in metadata:
            data_info = metadata[ticker_symbol].get("data_info", {})
            return {
                "start": data_info.get("training_period_start", "N/A"),
                "end": data_info.get("training_period_end", "N/A"),
                "ticker": ticker
            }
        return {"start": "N/A", "end": "N/A", "ticker": ticker}
