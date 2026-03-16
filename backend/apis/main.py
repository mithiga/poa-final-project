"""
FastAPI application — Forex AI Inference Engine.

Endpoints:
  POST /predict          — Generate forecast using a pre-trained model
  POST /train            — Train a single model
  POST /train_all        — Train all supported models
  GET  /status           — System status
  GET  /available_models — List supported model names
  GET  /available_tickers — List tickers with trained models
  GET  /training-period  — Get training period for a ticker
  GET  /market-data      — Fetch historical OHLCV data
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .pydantic_models import (
    PredictionRequest, PredictionResponse,
    TrainRequest, TrainResponse,
    TrainAllRequest, TrainAllResponse,
    SystemStatusResponse, SUPPORTED_MODELS,
    HyperparameterTuningRequest, HyperparameterTuningResponse, ModelHyperparameters,
    MarketOverviewResponse
)
from .services import ForecastService, TrainingService, MarketDataService, SystemService
from .ml_pipeline import get_model_hyperparameters, get_all_hyperparameters

app = FastAPI(
    title="Forex AI API",
    description="Decoupled ML Inference Engine for Forex Forecasting",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Prediction ──────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, tags=["Forecasting"])
async def predict(request: PredictionRequest):
    """Generate a price forecast using a pre-trained model."""
    try:
        dates, predictions, metrics = ForecastService.get_predictions(
            ticker=request.ticker,
            model_type=request.model_type,
            days=request.days
        )
        return PredictionResponse(
            ticker=request.ticker,
            model_used=request.model_type,
            dates=dates,
            predictions=predictions,
            metrics=metrics,
            status="Success"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast", tags=["Forecasting"])
async def forecast(
    ticker: str,
    model: str,
    steps: int = Query(default=30, ge=1, le=365)
):
    """
    Generate forecast via query params (alternative to /predict).
    Ticker should be the symbol without =X suffix (e.g., EURUSD).
    """
    try:
        from ml_pipeline import generate_forecast, load_model_metadata
        import pandas as pd

        # Reconstruct full ticker
        full_ticker = ticker if "=" in ticker else f"{ticker}=X"
        metadata = load_model_metadata()
        ticker_symbol = ticker.replace("=X", "").replace("=", "")

        cutoff_date = None
        if ticker_symbol in metadata:
            models_meta = metadata[ticker_symbol].get("models", {})
            model_meta = models_meta.get(model) or models_meta.get(model.upper())
            if model_meta:
                cutoff_date = model_meta.get("training_cutoff_date")

        if cutoff_date is None:
            cutoff_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        result = generate_forecast(
            model=model,
            ticker=full_ticker,
            start_date=cutoff_date,
            steps=steps
        )
        result["cutoff_date"] = cutoff_date
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Training ────────────────────────────────────────────────────────────────────

@app.post("/train", response_model=TrainResponse, tags=["Training"])
async def train(request: TrainRequest):
    """Train a single model on historical data."""
    try:
        result = TrainingService.train_single(
            ticker=request.ticker,
            model=request.model,
            start_date=request.start_date,
            end_date=request.end_date,
            train_size=request.train_size
        )
        return TrainResponse(
            ticker=request.ticker,
            model=request.model,
            rmse=result.rmse,
            mae=result.mae,
            mape=result.mape,
            status="Success",
            message=f"{request.model} trained successfully for {request.ticker}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train_all", response_model=TrainAllResponse, tags=["Training"])
async def train_all(request: TrainAllRequest):
    """Train all supported models on historical data."""
    try:
        results = TrainingService.train_all(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            train_size=request.train_size
        )
        return TrainAllResponse(
            ticker=request.ticker,
            evaluation=results["evaluation"],
            training=results["training"],
            status="Success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train_all_async/start", tags=["Training"])
async def train_all_async_start(request: TrainAllRequest):
    """Start async Train All job and return a job handle."""
    try:
        return TrainingService.start_train_all_job(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            train_size=request.train_size,
            force_retrain=request.force_retrain,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/train_all_async/status", tags=["Training"])
async def train_all_async_status(job_id: str = Query(..., description="Async job ID")):
    """Get async Train All job state."""
    state = TrainingService.get_train_all_job(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return state


@app.post("/train_all_async/cancel", tags=["Training"])
async def train_all_async_cancel(payload: dict):
    """Request cancellation for an async Train All job."""
    job_id = payload.get("job_id") if isinstance(payload, dict) else None
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id is required")
    state = TrainingService.cancel_train_all_job(str(job_id))
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": str(job_id),
        "status": state.get("status"),
        "cancel_requested": bool(state.get("cancel_requested", False)),
    }


# ─── System & Metadata ───────────────────────────────────────────────────────────

@app.get("/status", response_model=SystemStatusResponse, tags=["System"])
async def get_status():
    """Return system status and available trained models."""
    status = SystemService.get_status()
    return SystemStatusResponse(**status)


@app.get("/available_models", tags=["System"])
async def get_available_models():
    """Return list of supported model names."""
    return {"models": SUPPORTED_MODELS}


@app.get("/available_tickers", tags=["System"])
async def get_available_tickers():
    """Return list of tickers that have trained models."""
    tickers = SystemService.get_available_tickers()
    return {"tickers": tickers}


@app.get("/training-period", tags=["System"])
async def get_training_period(ticker: str = Query(..., description="Ticker symbol")):
    """Return the training period for a given ticker."""
    return SystemService.get_training_period(ticker)


@app.get("/model-cutoff-date", tags=["System"])
async def get_model_cutoff_date(ticker: str = Query(..., description="Ticker symbol"), model: str = Query(..., description="Model name")):
    """Return the training cutoff date for a specific model."""
    try:
        from .ml_pipeline import load_model_metadata
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
                    "training_period_end": model_meta.get("training_period_end")
                }
        return {"ticker": ticker, "model": model, "cutoff_date": None, "error": "Model not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Market Data ─────────────────────────────────────────────────────────────────

@app.get("/market-data", tags=["Data"])
async def get_market_data(
    ticker: str = Query(...),
    start_date: str = Query(...),
    end_date: str = Query(...)
):
    """Fetch historical OHLCV data for a ticker."""
    try:
        df = MarketDataService.get_data(ticker, start_date, end_date)
        df_reset = df.reset_index()
        df_reset["Date"] = df_reset["Date"].dt.strftime("%Y-%m-%d")
        return {"ticker": ticker, "data": df_reset.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market-overview", response_model=MarketOverviewResponse, tags=["Data"])
async def get_market_overview(
    ticker: str = Query(...),
    start_date: str = Query(...),
    end_date: str = Query(...),
):
    """Fetch market data with backend-computed model sentiment summary."""
    try:
        return MarketDataService.get_market_overview(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Hyperparameter Tuning ─────────────────────────────────────────────────────────

@app.get("/hyperparameters", tags=["Hyperparameters"])
async def get_hyperparameters(model: str = Query(..., description="Model name")):
    """Get available hyperparameters for a specific model."""
    if model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Model must be one of {SUPPORTED_MODELS}")
    
    hyperparameters = get_model_hyperparameters(model)
    return ModelHyperparameters(model=model, hyperparameters=hyperparameters)


@app.get("/all_hyperparameters", tags=["Hyperparameters"])
async def get_all_model_hyperparameters():
    """Get hyperparameters for all supported models."""
    all_hp = get_all_hyperparameters()
    result = {}
    for model_name, params in all_hp.items():
        result[model_name] = params
    return result


@app.post("/train_with_tuning", response_model=HyperparameterTuningResponse, tags=["Training"])
async def train_with_tuning(request: HyperparameterTuningRequest):
    """Train a model with custom hyperparameters."""
    try:
        result = TrainingService.train_with_hyperparameters(
            ticker=request.ticker,
            model=request.model,
            start_date=request.start_date,
            end_date=request.end_date,
            train_size=request.train_size,
            hyperparameters=request.hyperparameters
        )
        selected_hyperparameters = request.hyperparameters
        if getattr(result, "artifacts", None) and isinstance(result.artifacts, dict):
            selected_hyperparameters = result.artifacts.get("best_hyperparameters", request.hyperparameters)

        return HyperparameterTuningResponse(
            ticker=request.ticker,
            model=request.model,
            hyperparameters_used=selected_hyperparameters,
            rmse=result.rmse,
            mae=result.mae,
            mape=result.mape,
            status="Success",
            message=f"{request.model} trained successfully with custom hyperparameters for {request.ticker}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Entry Point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
