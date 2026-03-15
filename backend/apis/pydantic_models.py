"""
Pydantic schemas for the Forex Forecasting API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any


SUPPORTED_MODELS = ["ARIMA", "SARIMAX", "SARIMA", "LSTM", "GRU",
                    "Prophet", "LightGBM", "LinearRegression", "RandomForest"]


class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="e.g., EURUSD=X or GC=F")
    model_type: str = Field(..., description=f"One of: {SUPPORTED_MODELS}")
    days: int = Field(default=7, ge=1, le=90, description="Number of forecast days")

    @field_validator("model_type")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in SUPPORTED_MODELS:
            raise ValueError(f"Model must be one of {SUPPORTED_MODELS}")
        return v


class PredictionResponse(BaseModel):
    ticker: str
    model_used: str
    dates: List[str]
    predictions: List[float]
    lower: Optional[List[float]] = None
    upper: Optional[List[float]] = None
    metrics: Dict[str, float]
    status: str = "Success"


class TrainRequest(BaseModel):
    ticker: str = Field(..., description="e.g., EURUSD=X")
    model: str = Field(..., description=f"One of: {SUPPORTED_MODELS}")
    start_date: str = Field(..., description="Training start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Training end date (YYYY-MM-DD)")
    train_size: float = Field(default=0.8, ge=0.5, le=0.95,
                              description="Fraction of data used for training")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in SUPPORTED_MODELS:
            raise ValueError(f"Model must be one of {SUPPORTED_MODELS}")
        return v


class TrainResponse(BaseModel):
    ticker: str
    model: str
    rmse: float
    mae: float
    mape: float
    status: str = "Success"
    message: str = ""


class TrainAllRequest(BaseModel):
    ticker: str = Field(..., description="e.g., EURUSD=X")
    start_date: str = Field(..., description="Training start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Training end date (YYYY-MM-DD)")
    train_size: float = Field(default=0.8, ge=0.5, le=0.95)


class TrainAllResponse(BaseModel):
    ticker: str
    evaluation: Dict[str, Any]
    training: Dict[str, Any]
    status: str = "Success"


class MarketDataRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str


class ModelSentiment(BaseModel):
    model: str
    signal: str
    predicted_price: Optional[float] = None
    change_pct: Optional[float] = None


class MarketSentimentSummary(BaseModel):
    overall: str
    flat_threshold_pct: float
    reference_price: Optional[float] = None
    models: List[ModelSentiment]


class MarketOverviewResponse(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    data: List[Dict[str, Any]]
    sentiment: MarketSentimentSummary
    status: str = "Success"


class SystemStatusResponse(BaseModel):
    status: str
    models_available: List[str]
    tickers_trained: List[str]
    message: str = ""


# ─── Hyperparameter Tuning ─────────────────────────────────────────────────────────

class HyperparameterConfig(BaseModel):
    """Configuration for a single hyperparameter."""
    name: str
    type: str  # "int", "float", "categorical", "bool"
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[Any]] = None  # For categorical
    description: str = ""


class ModelHyperparameters(BaseModel):
    """Hyperparameters available for a specific model."""
    model: str
    hyperparameters: List[HyperparameterConfig]


class HyperparameterTuningRequest(BaseModel):
    """Request to train a model with custom hyperparameters."""
    ticker: str = Field(..., description="e.g., EURUSD=X")
    model: str = Field(..., description=f"One of: {SUPPORTED_MODELS}")
    start_date: str = Field(..., description="Training start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Training end date (YYYY-MM-DD)")
    train_size: float = Field(default=0.8, ge=0.5, le=0.95)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Custom hyperparameter values")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in SUPPORTED_MODELS:
            raise ValueError(f"Model must be one of {SUPPORTED_MODELS}")
        return v


class HyperparameterTuningResponse(BaseModel):
    """Response after training with custom hyperparameters."""
    ticker: str
    model: str
    hyperparameters_used: Dict[str, Any]
    rmse: float
    mae: float
    mape: float
    status: str = "Success"
    message: str = ""
