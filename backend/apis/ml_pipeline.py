"""
ML Pipeline for Forex Forecasting.

Provides data fetching, feature engineering, model training, and forecasting
for ARIMA, SARIMAX, LSTM, GRU, Prophet, LightGBM, LinearRegression, and RandomForest.
"""

import os
import json
import joblib
import shutil
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf

from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

from pydantic import BaseModel
from .pandas_compat import patch_stringdtype_unpickle_compat


patch_stringdtype_unpickle_compat()


# ─── PyTorch Model Definitions ───────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, num_layers=2, dropout=0.2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.linear1 = nn.Linear(hidden_layer_size, 25)
        self.linear2 = nn.Linear(25, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = torch.relu(self.linear1(last_time_step))
        return self.linear2(out)


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, num_layers=2, dropout=0.2, output_size=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_layer_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.linear1 = nn.Linear(hidden_layer_size, 25)
        self.linear2 = nn.Linear(25, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_time_step = gru_out[:, -1, :]
        out = torch.relu(self.linear1(last_time_step))
        return self.linear2(out)


# ─── Result Schema ───────────────────────────────────────────────────────────────

class ModelResult(BaseModel):
    model_name: str
    model: object
    rmse: float
    mae: float
    mape: float
    artifacts: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True


# ─── Hyperparameter Definitions ────────────────────────────────────────────────────

MODEL_HYPERPARAMETERS = {
    "ARIMA": [
        {"name": "p", "type": "int", "default": 1, "min": 0, "max": 5, "step": 1, "description": "AR order"},
        {"name": "d", "type": "int", "default": 1, "min": 0, "max": 2, "step": 1, "description": "Differencing order"},
        {"name": "q", "type": "int", "default": 1, "min": 0, "max": 5, "step": 1, "description": "MA order"},
    ],
    "SARIMAX": [
        {"name": "p", "type": "int", "default": 1, "min": 0, "max": 3, "step": 1, "description": "AR order"},
        {"name": "d", "type": "int", "default": 1, "min": 0, "max": 2, "step": 1, "description": "Differencing order"},
        {"name": "q", "type": "int", "default": 1, "min": 0, "max": 3, "step": 1, "description": "MA order"},
        {"name": "seasonal_p", "type": "int", "default": 1, "min": 0, "max": 2, "step": 1, "description": "Seasonal AR order"},
        {"name": "seasonal_d", "type": "int", "default": 1, "min": 0, "max": 1, "step": 1, "description": "Seasonal differencing"},
        {"name": "seasonal_q", "type": "int", "default": 1, "min": 0, "max": 2, "step": 1, "description": "Seasonal MA order"},
        {"name": "s", "type": "int", "default": 12, "min": 1, "max": 365, "step": 1, "description": "Seasonal period"},
    ],
    "SARIMA": [
        {"name": "p", "type": "int", "default": 1, "min": 0, "max": 3, "step": 1, "description": "AR order"},
        {"name": "d", "type": "int", "default": 1, "min": 0, "max": 2, "step": 1, "description": "Differencing order"},
        {"name": "q", "type": "int", "default": 1, "min": 0, "max": 3, "step": 1, "description": "MA order"},
    ],
    "LSTM": [
        {"name": "hidden_layer_size", "type": "int", "default": 50, "min": 10, "max": 200, "step": 10, "description": "Number of hidden units"},
        {"name": "num_layers", "type": "int", "default": 2, "min": 1, "max": 4, "step": 1, "description": "Number of LSTM layers"},
        {"name": "dropout", "type": "float", "default": 0.2, "min": 0.0, "max": 0.5, "step": 0.1, "description": "Dropout rate"},
        {"name": "learning_rate", "type": "float", "default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001, "description": "Learning rate"},
        {"name": "epochs", "type": "int", "default": 50, "min": 10, "max": 200, "step": 10, "description": "Number of training epochs"},
        {"name": "sequence_length", "type": "int", "default": 30, "min": 10, "max": 120, "step": 5, "description": "Input sequence length"},
    ],
    "GRU": [
        {"name": "hidden_layer_size", "type": "int", "default": 50, "min": 10, "max": 200, "step": 10, "description": "Number of hidden units"},
        {"name": "num_layers", "type": "int", "default": 2, "min": 1, "max": 4, "step": 1, "description": "Number of GRU layers"},
        {"name": "dropout", "type": "float", "default": 0.2, "min": 0.0, "max": 0.5, "step": 0.1, "description": "Dropout rate"},
        {"name": "learning_rate", "type": "float", "default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001, "description": "Learning rate"},
        {"name": "epochs", "type": "int", "default": 50, "min": 10, "max": 200, "step": 10, "description": "Number of training epochs"},
        {"name": "sequence_length", "type": "int", "default": 30, "min": 10, "max": 120, "step": 5, "description": "Input sequence length"},
    ],
    "Prophet": [
        {"name": "changepoint_prior_scale", "type": "float", "default": 0.05, "min": 0.001, "max": 0.5, "step": 0.001, "description": "Trend changepoint prior"},
        {"name": "seasonality_prior_scale", "type": "float", "default": 10.0, "min": 0.01, "max": 100.0, "step": 0.1, "description": "Seasonality prior scale"},
        {"name": "seasonality_mode", "type": "categorical", "default": "multiplicative", "options": ["additive", "multiplicative"], "description": "Seasonality mode"},
        {"name": "daily_seasonality", "type": "bool", "default": False, "description": "Include daily seasonality"},
        {"name": "weekly_seasonality", "type": "bool", "default": True, "description": "Include weekly seasonality"},
        {"name": "yearly_seasonality", "type": "bool", "default": True, "description": "Include yearly seasonality"},
    ],
    "LightGBM": [
        {"name": "n_estimators", "type": "int", "default": 100, "min": 50, "max": 500, "step": 10, "description": "Number of boosting rounds"},
        {"name": "learning_rate", "type": "float", "default": 0.1, "min": 0.01, "max": 0.3, "step": 0.01, "description": "Learning rate"},
        {"name": "max_depth", "type": "int", "default": -1, "min": -1, "max": 10, "step": 1, "description": "Max tree depth (-1 for unlimited)"},
        {"name": "num_leaves", "type": "int", "default": 31, "min": 10, "max": 100, "step": 1, "description": "Number of leaves"},
        {"name": "min_child_samples", "type": "int", "default": 20, "min": 5, "max": 100, "step": 5, "description": "Min samples in leaf"},
        {"name": "subsample", "type": "float", "default": 1.0, "min": 0.5, "max": 1.0, "step": 0.1, "description": "Subsample ratio"},
        {"name": "colsample_bytree", "type": "float", "default": 1.0, "min": 0.5, "max": 1.0, "step": 0.1, "description": "Column sample ratio"},
    ],
    "LinearRegression": [
        {"name": "fit_intercept", "type": "bool", "default": True, "description": "Fit intercept"},
    ],
    "RandomForest": [
        {"name": "n_estimators", "type": "int", "default": 100, "min": 50, "max": 500, "step": 10, "description": "Number of trees"},
        {"name": "max_depth", "type": "int", "default": None, "min": 1, "max": 30, "step": 1, "description": "Max tree depth (None for unlimited)"},
        {"name": "min_samples_split", "type": "int", "default": 2, "min": 2, "max": 20, "step": 1, "description": "Min samples to split"},
        {"name": "min_samples_leaf", "type": "int", "default": 1, "min": 1, "max": 10, "step": 1, "description": "Min samples in leaf"},
        {"name": "max_features", "type": "categorical", "default": "sqrt", "options": ["sqrt", "log2", "None"], "description": "Max features per split"},
    ],
}


def get_model_hyperparameters(model: str) -> list:
    """Return hyperparameters for a specific model."""
    return MODEL_HYPERPARAMETERS.get(model, [])


def get_all_hyperparameters() -> dict:
    """Return all model hyperparameters."""
    return MODEL_HYPERPARAMETERS


def expand_hyperparameter_grid(hp: dict) -> list:
    """Expand hyperparameter inputs into a list of parameter dictionaries.

    Supports two input types:
      - exact values:              {"alpha": 0.1}
      - range definitions:         {"alpha": {"range": [0.01, 0.1, 0.01]}}

    Range values are expanded into a list of values (inclusive of max).
    """

    def build_values(v):
        if isinstance(v, dict) and "range" in v and isinstance(v["range"], (list, tuple)) and len(v["range"]) == 3:
            lo, hi, step = v["range"]
            if isinstance(lo, int) and isinstance(hi, int) and isinstance(step, int):
                return list(range(lo, hi + 1, step))
            return list(np.arange(lo, hi + step * 0.5, step))
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]

    keys = list(hp.keys())
    values_list = [build_values(hp[k]) for k in keys]

    combos = []
    for combo in itertools.product(*values_list):
        combos.append({k: v for k, v in zip(keys, combo)})
    return combos


def _coerce_bool(value, default=False):
    """Safely parse bool values coming from UI payloads."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "y", "on"):
            return True
        if lowered in ("false", "0", "no", "n", "off"):
            return False
    return default


def _coerce_optional_int(value):
    """Convert optional numeric/string values into int or None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return int(value)


def _rmse(y_true, y_pred) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def _build_time_series_splits(n_samples: int, max_splits: int = 3,
                              min_train_size: int = 30,
                              min_val_size: int = 5):
    """Build robust TimeSeriesSplit folds for small and large datasets."""
    if n_samples < (min_train_size + min_val_size + 1):
        return []

    candidate_splits = min(max_splits, n_samples - 1)
    for n_splits in range(candidate_splits, 1, -1):
        try:
            splits = list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_samples)))
        except ValueError:
            continue
        if all(len(tr) >= min_train_size and len(val) >= min_val_size for tr, val in splits):
            return splits
    return []


def _get_feature_columns(data: pd.DataFrame):
    """Return usable feature columns while excluding the target column."""
    feature_cols = [c for c in data.columns if c != "Close"]
    if not feature_cols:
        feature_cols = ["Close"]
    return feature_cols


def _to_prophet_df(data: pd.DataFrame) -> pd.DataFrame:
    """Convert a data frame with index timestamps into Prophet format."""
    out = data.reset_index()
    if "Date" in out.columns:
        out = out.rename(columns={"Date": "ds"})
    else:
        out = out.rename(columns={out.columns[0]: "ds"})
    out = out.rename(columns={"Close": "y"})
    out = out[["ds", "y"]].copy()
    out["ds"] = pd.to_datetime(out["ds"])
    return out


# ─── Utility Functions ───────────────────────────────────────────────────────────

# Base directory for model storage (relative to backend root)
MODELS_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PARAMETERS_PATH = os.path.join(MODELS_BASE_DIR, "model_parameters.json")


def _atomic_joblib_dump(payload, destination_path: str) -> None:
    """Write a joblib artifact atomically to avoid partial/corrupt files."""
    directory = os.path.dirname(destination_path)
    os.makedirs(directory, exist_ok=True)
    temp_path = os.path.join(
        directory,
        f".{os.path.basename(destination_path)}.tmp.{os.getpid()}"
    )
    try:
        joblib.dump(payload, temp_path)
        os.replace(temp_path, destination_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _save_validated_lightgbm_model(booster: lgb.Booster, destination_path: str) -> None:
    """
    Save LightGBM model atomically and validate the saved file before promotion.

    A previous known-good model is retained as `<model>.lastgood` for fallback.
    """
    directory = os.path.dirname(destination_path)
    os.makedirs(directory, exist_ok=True)
    temp_path = os.path.join(
        directory,
        f".{os.path.basename(destination_path)}.tmp.{os.getpid()}"
    )
    backup_path = f"{destination_path}.lastgood"

    try:
        booster.save_model(temp_path)

        # Validate that the just-saved artifact is readable and non-empty.
        validated = lgb.Booster(model_file=temp_path)
        if validated.num_trees() <= 0:
            raise RuntimeError("Validated LightGBM model has zero trees.")

        if os.path.exists(destination_path):
            shutil.copy2(destination_path, backup_path)

        os.replace(temp_path, destination_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _load_validated_lightgbm_model(destination_path: str) -> lgb.Booster:
    """Load LightGBM model with fallback to last known-good backup."""
    candidates = [destination_path, f"{destination_path}.lastgood"]
    failures = []

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        try:
            booster = lgb.Booster(model_file=candidate)
            if booster.num_trees() <= 0:
                raise RuntimeError("Model has zero trees.")
            return booster
        except Exception as exc:
            failures.append(f"{candidate}: {exc}")

    details = " | ".join(failures) if failures else "No model files found."
    raise RuntimeError(
        f"Unable to load a valid LightGBM model for path '{destination_path}'. {details}"
    )


def _canonical_model_name(model_name: str) -> str:
    """Normalize model names/aliases for consistent parameter persistence."""
    if not model_name:
        return model_name
    normalized = model_name.upper()
    mapping = {
        "ARIMA": "ARIMA",
        "SARIMAX": "SARIMA",
        "SARIMA": "SARIMA",
        "LSTM": "LSTM",
        "GRU": "GRU",
        "PROPHET": "Prophet",
        "LIGHTGBM": "LightGBM",
        "LINEARREGRESSION": "LinearRegression",
        "RANDOMFOREST": "RandomForest",
    }
    return mapping.get(normalized, model_name)


def get_ticker_folder(ticker: str):
    """Create and return path to ticker-specific model folder."""
    ticker_symbol = ticker.replace("=X", "").replace("=", "").replace("/", "")
    folder_path = os.path.join(MODELS_BASE_DIR, ticker_symbol)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path, ticker_symbol


def update_model_metadata(ticker: str, model_name: Optional[str], filename: Optional[str],
                          model_type: Optional[str], train_end_date: str,
                          train_start_date: Optional[str] = None):
    """Update model_metadata.json with new model info."""
    ticker_symbol = ticker.replace("=X", "").replace("=", "").replace("/", "")
    metadata_path = os.path.join(MODELS_BASE_DIR, "model_metadata.json")

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    if ticker_symbol not in metadata:
        metadata[ticker_symbol] = {
            "models": {},
            "data_info": {
                "symbol": ticker,
                "description": f"{ticker_symbol} Currency Pair"
            }
        }

    metadata[ticker_symbol]["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if train_start_date is not None:
        metadata[ticker_symbol]["data_info"]["training_period_start"] = train_start_date
        metadata[ticker_symbol]["data_info"]["training_period_end"] = train_end_date
        metadata[ticker_symbol]["data_info"]["prediction_valid_from"] = train_end_date

    if model_name and filename and model_type:
        metadata[ticker_symbol]["models"][model_name] = {
            "name": model_name,
            "filename": filename,
            "training_cutoff_date": train_end_date,
            "model_type": model_type,
            "data_symbol": ticker,
            "last_model_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    os.makedirs(MODELS_BASE_DIR, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Updated metadata for {model_name} in {ticker_symbol}")


def load_model_metadata() -> dict:
    """Load and return the full model metadata."""
    metadata_path = os.path.join(MODELS_BASE_DIR, "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}


def load_model_parameters() -> dict:
    """Load persisted best model hyperparameters by ticker/model."""
    if os.path.exists(MODEL_PARAMETERS_PATH):
        with open(MODEL_PARAMETERS_PATH, "r") as f:
            return json.load(f)
    return {}


def update_model_parameters(ticker: str, model_name: str,
                            best_hyperparameters: dict,
                            cv_rmse: Optional[float] = None,
                            source: str = "grid_search_cv"):
    """Persist best CV hyperparameters for reuse in future training runs."""
    ticker_symbol = ticker.replace("=X", "").replace("=", "").replace("/", "")
    canonical_model = _canonical_model_name(model_name)
    params_store = load_model_parameters()

    if ticker_symbol not in params_store:
        params_store[ticker_symbol] = {"models": {}}

    params_store[ticker_symbol]["models"][canonical_model] = {
        "best_hyperparameters": best_hyperparameters,
        "cv_rmse": cv_rmse,
        "source": source,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    os.makedirs(MODELS_BASE_DIR, exist_ok=True)
    with open(MODEL_PARAMETERS_PATH, "w") as f:
        json.dump(params_store, f, indent=2)


def get_saved_best_hyperparameters(ticker: str, model_name: str) -> Optional[dict]:
    """Return persisted best hyperparameters for a ticker/model if available."""
    ticker_symbol = ticker.replace("=X", "").replace("=", "").replace("/", "")
    canonical_model = _canonical_model_name(model_name)
    params_store = load_model_parameters()

    model_entry = params_store.get(ticker_symbol, {}).get("models", {}).get(canonical_model)
    if not model_entry:
        return None
    return model_entry.get("best_hyperparameters")


# ─── Data Pipeline ───────────────────────────────────────────────────────────────

def fetch_data(start_date: str, end_date: str, ticker: str) -> pd.DataFrame:
    """Fetch historical OHLCV data via yfinance.

    Some symbols (e.g., XAUUSD) are not available via yfinance, so we map
    common aliases to a supported ticker.
    """
    # Map common symbols to yfinance-supported tickers
    alias_map = {
        "XAUUSD": "GC=F",
        "XAUUSD=X": "GC=F",
        "XAU/USD": "GC=F",
        "GCF": "GC=F",
        "GCF=X": "GC=F",
    }
    yf_ticker = alias_map.get(ticker, ticker)

    data = yf.download(yf_ticker, start=start_date, end=end_date,
                       auto_adjust=False, progress=False, actions=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if data.empty or "Close" not in data.columns:
        raise ValueError(
            f"No market data returned for ticker '{ticker}' (resolved to '{yf_ticker}') "
            f"between {start_date} and {end_date}."
        )

    data.index = pd.to_datetime(data.index)

    if "Adj Close" not in data.columns and "Adj_Close" not in data.columns:
        data["Adj Close"] = data["Close"]

    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing values."""
    data = data.ffill()
    return data


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as features.

    If a feature cannot be computed due to insufficient history (short date range),
    it is dropped so that the pipeline can still train using just price data.
    """
    # Keep a copy of the base price column so we can fall back to it if needed.
    base_close = data[["Close"]].copy()

    data["SMA_10"] = data["Close"].rolling(window=10).mean().shift(1)
    data["SMA_50"] = data["Close"].rolling(window=50).mean().shift(1)
    data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean().shift(1)
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean().shift(1)

    exp1 = data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = data["Close"].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    data["MACD"] = macd.shift(1)
    data["MACD_Signal"] = macd.ewm(span=9, adjust=False).mean().shift(1)

    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = (100 - (100 / (1 + rs))).shift(1)

    for i in range(1, 6):
        data[f"Lag_{i}"] = data["Close"].shift(i)

    data["Volatility"] = data["Close"].rolling(window=5).std().shift(1)
    data["BB_MA"] = data["Close"].rolling(window=20).mean().shift(1)
    data["BB_Std"] = data["Close"].rolling(window=20).std().shift(1)
    data["BB_Upper"] = data["BB_MA"] + (2 * data["BB_Std"])
    data["BB_Lower"] = data["BB_MA"] - (2 * data["BB_Std"])

    # Drop any features that could not be computed (all NaNs) due to limited history.
    feature_cols = [c for c in data.columns if c != "Close"]
    cols_to_drop = [c for c in feature_cols if data[c].isna().all()]
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)

    # Drop rows with remaining NaNs. If this results in an empty frame, fall back to Close-only.
    data = data.dropna()
    if data.empty or list(data.columns) == ["Close"]:
        return base_close

    return data


def split_data(data: pd.DataFrame, train_size: float = 0.8):
    """Split data into train and test sets."""
    n = int(len(data) * train_size)
    return data.iloc[:n], data.iloc[n:]


def evaluate_preds(y_true, y_pred, model_name: str = ""):
    """Compute RMSE, MAE, MAPE."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    print(f"--- {model_name} --- RMSE: {rmse:.4f}  MAE: {mae:.4f}  MAPE: {mape:.4f}")
    return rmse, mae, mape


# ─── Model Training Functions ────────────────────────────────────────────────────

def arima_model(train: pd.DataFrame, test: Optional[pd.DataFrame], ticker: str = "EURUSD=X",
                final_fit: bool = False) -> ModelResult:
    print("Tuning ARIMA hyperparameters...")
    model_auto = auto_arima(train["Close"], start_p=1, start_q=1, max_p=3, max_q=3,
                            m=1, start_P=0, seasonal=False, d=1, D=0, trace=True,
                            error_action="ignore", suppress_warnings=True, stepwise=True)
    best_order = model_auto.order
    print(f"Optimal ARIMA Order: {best_order}")

    fitted = ARIMA(np.asarray(train["Close"], dtype=np.float64), order=best_order).fit()
    if final_fit or test is None or len(test) == 0:
        rmse = mae = mape = float("nan")
    else:
        predictions = np.asarray(fitted.forecast(steps=len(test)), dtype=np.float64)
        rmse, mae, mape = evaluate_preds(test["Close"], predictions, "ARIMA")

    ticker_folder, ticker_symbol = get_ticker_folder(ticker)
    filename = f"ARIMA_{ticker_symbol}.pkl"
    joblib.dump(fitted, os.path.join(ticker_folder, filename))
    train_end = train.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, "ARIMA", filename, "arima", train_end)

    return ModelResult(model_name="ARIMA", model=fitted, rmse=rmse, mae=mae, mape=mape)


def sarimax_model(train: pd.DataFrame, test: Optional[pd.DataFrame], ticker: str = "EURUSD=X",
                  final_fit: bool = False) -> ModelResult:
    print("Tuning SARIMAX hyperparameters...")
    model_auto = auto_arima(train["Close"], start_p=1, start_q=1, max_p=3, max_q=3,
                            m=1, start_P=0, seasonal=True, d=1, D=0, trace=True,
                            error_action="ignore", suppress_warnings=True, stepwise=True)
    best_order = model_auto.order
    print(f"Optimal SARIMAX Order: {best_order}")

    fitted = SARIMAX(np.asarray(train["Close"], dtype=np.float64), order=best_order).fit(disp=False)
    if final_fit or test is None or len(test) == 0:
        rmse = mae = mape = float("nan")
    else:
        predictions = np.asarray(fitted.forecast(steps=len(test)), dtype=np.float64)
        rmse, mae, mape = evaluate_preds(test["Close"], predictions, "SARIMAX")

    ticker_folder, ticker_symbol = get_ticker_folder(ticker)
    filename = f"SARIMA_{ticker_symbol}.pkl"
    joblib.dump(fitted, os.path.join(ticker_folder, filename))
    train_end = train.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, "SARIMA", filename, "sarima", train_end)

    return ModelResult(model_name="SARIMAX", model=fitted, rmse=rmse, mae=mae, mape=mape)


def _train_sequence_model(model_class, model_label: str, train: pd.DataFrame,
                          test: Optional[pd.DataFrame], ticker: str,
                          hidden_layer_size: int = 50,
                          num_layers: int = 2,
                          dropout: float = 0.2,
                          learning_rate: float = 0.001,
                          epochs: int = 20,
                          sequence_length: int = 60,
                          save_artifacts: bool = True,
                          final_fit: bool = False) -> ModelResult:
    """Shared training logic for LSTM and GRU.

    All hyperparameters are exposed for grid-search when called from the
    *_with_hp helper functions.
    """
    ticker_folder, ticker_symbol = get_ticker_folder(ticker)

    train_data = train[["Close"]].values
    test_data = np.empty((0, 1)) if test is None else test[["Close"]].values

    if len(train_data) < 2:
        raise ValueError(
            f"Insufficient data for {model_label}: need at least 2 training rows, got {len(train_data)}."
        )

    # Guard against invalid sequence lengths that would create an empty dataset.
    if sequence_length >= len(train_data):
        adjusted_seq_len = len(train_data) - 1
        print(
            f"{model_label}: sequence_length={sequence_length} is too large for "
            f"{len(train_data)} samples. Using {adjusted_seq_len} instead."
        )
        sequence_length = adjusted_seq_len

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data)

    def create_sequences(data, seq_len):
        x, y = [], []
        for i in range(seq_len, len(data)):
            x.append(data[i - seq_len:i, 0])
            y.append(data[i, 0])
        return np.array(x), np.array(y)

    X_seq, y_seq = create_sequences(scaled_train, sequence_length)
    if len(X_seq) == 0:
        raise ValueError(
            f"No training sequences generated for {model_label}. "
            f"Try a smaller sequence_length or a larger date range."
        )

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = model_class(input_size=1, hidden_layer_size=hidden_layer_size,
                        num_layers=num_layers, dropout=dropout)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        for seq, labels in dataloader:
            optimizer.zero_grad()
            loss = loss_fn(model(seq), labels)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch} loss: {loss.item():.6f}")

    # Autoregressive forecasting
    model.eval()
    predictions = []
    current_seq = scaled_train[-sequence_length:].copy()
    current_seq_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        for _ in range(len(test_data)):
            pred = model(current_seq_tensor)
            predictions.append(pred.item())
            next_val = pred.unsqueeze(1)
            current_seq_tensor = torch.cat((current_seq_tensor[:, 1:, :], next_val), dim=1)

    if final_fit or len(test_data) == 0:
        rmse = mae = mape = float("nan")
    else:
        predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        rmse, mae, mape = evaluate_preds(test_data.reshape(-1, 1), predictions_inv, model_label)

    if save_artifacts:
        # Save model weights
        pth_filename = f"{model_label}_{ticker_symbol}.pth"
        torch.save(model.state_dict(), os.path.join(ticker_folder, pth_filename))

        # Save scaler + last sequence
        data_filename = f"{model_label}_{ticker_symbol}_data.pkl"
        joblib.dump({
            "scaler": scaler,
            "last_sequence": scaled_train[-sequence_length:],
            "sequence_length": sequence_length
        }, os.path.join(ticker_folder, data_filename))

        train_end = train.index[-1].strftime("%Y-%m-%d")
        update_model_metadata(ticker, model_label, data_filename, model_label.lower(), train_end)
        print(f"{model_label} model saved to {ticker_folder}/{pth_filename}")

    return ModelResult(
        model_name=model_label,
        model=model,
        rmse=rmse,
        mae=mae,
        mape=mape,
        artifacts={
            "scaler": scaler,
            "last_sequence": scaled_train[-sequence_length:],
            "sequence_length": sequence_length
        }
    )


def lstm_model(train: pd.DataFrame, test: Optional[pd.DataFrame], ticker: str = "EURUSD=X",
               final_fit: bool = False) -> ModelResult:
    print("Training LSTM model...")
    return _train_sequence_model(LSTMModel, "LSTM", train, test, ticker, final_fit=final_fit)


def gru_model(train: pd.DataFrame, test: Optional[pd.DataFrame], ticker: str = "EURUSD=X",
              final_fit: bool = False) -> ModelResult:
    print("Training GRU model...")
    return _train_sequence_model(GRUModel, "GRU", train, test, ticker, final_fit=final_fit)


NOTEBOOK_PROPHET_DEFAULT_PARAMS = {
    # Notebook behavior used a fixed daily_seasonality=False setup and selected
    # this pair from grid-search in baseline experiments.
    "changepoint_prior_scale": 0.1,
    "seasonality_prior_scale": 0.1,
    "daily_seasonality": False,
}


def _resolve_prophet_params(hyperparameters: Optional[dict]) -> dict:
    """Resolve Prophet params from saved/custom input with notebook-style fallbacks."""
    hp = hyperparameters or {}
    resolved = {
        "changepoint_prior_scale": float(hp.get(
            "changepoint_prior_scale", NOTEBOOK_PROPHET_DEFAULT_PARAMS["changepoint_prior_scale"]
        )),
        "seasonality_prior_scale": float(hp.get(
            "seasonality_prior_scale", NOTEBOOK_PROPHET_DEFAULT_PARAMS["seasonality_prior_scale"]
        )),
        "daily_seasonality": _coerce_bool(
            hp.get("daily_seasonality", NOTEBOOK_PROPHET_DEFAULT_PARAMS["daily_seasonality"]),
            NOTEBOOK_PROPHET_DEFAULT_PARAMS["daily_seasonality"],
        ),
    }

    # Keep explicit user/saved options when present; otherwise use Prophet defaults.
    if "seasonality_mode" in hp:
        resolved["seasonality_mode"] = hp["seasonality_mode"]
    if "weekly_seasonality" in hp:
        resolved["weekly_seasonality"] = _coerce_bool(hp["weekly_seasonality"], True)
    if "yearly_seasonality" in hp:
        resolved["yearly_seasonality"] = _coerce_bool(hp["yearly_seasonality"], True)

    return resolved


def prophet_model(train: pd.DataFrame, test: Optional[pd.DataFrame], ticker: str = "EURUSD=X",
                  final_fit: bool = False, hyperparameters: Optional[dict] = None) -> ModelResult:
    print("Training Prophet model...")
    ticker_folder, ticker_symbol = get_ticker_folder(ticker)

    train_df = train.reset_index().rename(columns={"Date": "ds", "Close": "y"})
    if "ds" not in train_df.columns:
        train_df = train_df.rename(columns={train_df.columns[0]: "ds"})
    train_df["ds"] = pd.to_datetime(train_df["ds"])

    test_df = None
    if test is not None and len(test) > 0:
        test_df = test.reset_index().rename(columns={"Date": "ds", "Close": "y"})
        if "ds" not in test_df.columns:
            test_df = test_df.rename(columns={test_df.columns[0]: "ds"})
        test_df["ds"] = pd.to_datetime(test_df["ds"])

    # Normal train/train_all path: do NOT run CV search. Use saved params when
    # available, else notebook-style defaults.
    m_best = Prophet(**_resolve_prophet_params(hyperparameters))
    m_best.add_country_holidays(country_name="US")
    m_best.fit(train_df)

    filename = f"PROPHET_{ticker_symbol}.pkl"
    joblib.dump(m_best, os.path.join(ticker_folder, filename))
    train_end = train.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, "Prophet", filename, "prophet", train_end)

    if final_fit or test_df is None or len(test_df) == 0:
        rmse = mae = mape = float("nan")
    else:
        forecast_test = m_best.predict(test_df[["ds"]])
        rmse, mae, mape = evaluate_preds(test_df["y"].values, forecast_test["yhat"].values, "Prophet")

    return ModelResult(model_name="Prophet", model=m_best, rmse=rmse, mae=mae, mape=mape)


def lightgbm_model(train: pd.DataFrame, test: Optional[pd.DataFrame], ticker: str = "EURUSD=X",
                   final_fit: bool = False) -> ModelResult:
    print("Training LightGBM model...")
    ticker_folder, ticker_symbol = get_ticker_folder(ticker)

    feature_cols = [c for c in train.columns if c != "Close"]
    X_train, y_train = train[feature_cols], train["Close"]
    has_eval = (not final_fit and test is not None and len(test) > 0)
    if has_eval:
        X_test, y_test = test[feature_cols], test["Close"]

    params = {
        "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
        "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.9,
        "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1,
        "n_estimators": 200, "random_state": 42
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    if has_eval:
        predictions = model.predict(X_test)
        rmse, mae, mape = evaluate_preds(y_test.values, predictions, "LightGBM")
    else:
        rmse = mae = mape = float("nan")

    lgbm_filename = f"LightGBM_{ticker_symbol}.txt"
    _save_validated_lightgbm_model(model.booster_, os.path.join(ticker_folder, lgbm_filename))

    data_filename = f"LightGBM_{ticker_symbol}_data.pkl"
    _atomic_joblib_dump({
        "feature_cols": feature_cols,
        "last_features": X_train.iloc[-1].values,
        "train_data": train[feature_cols + ["Close"]].values
    }, os.path.join(ticker_folder, data_filename))

    train_end = train.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, "LightGBM", lgbm_filename, "lightgbm", train_end)

    return ModelResult(model_name="LightGBM", model=model, rmse=rmse, mae=mae, mape=mape)


def linear_regression_model(train: pd.DataFrame, test: Optional[pd.DataFrame], ticker: str = "EURUSD=X",
                            final_fit: bool = False) -> ModelResult:
    print("Training LinearRegression model with hyperparameter tuning...")
    ticker_folder, ticker_symbol = get_ticker_folder(ticker)

    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.model_selection import TimeSeriesSplit

    feature_cols = [c for c in train.columns if c != "Close"]
    if not feature_cols:
        # If no engineered features exist, fall back to using Close as a feature.
        feature_cols = ["Close"]

    X_train, y_train = train[feature_cols], train["Close"]
    has_eval = (not final_fit and test is not None and len(test) > 0)
    if has_eval:
        X_test, y_test = test[feature_cols], test["Close"]

    param_grid = [
        {"model_class": LinearRegression, "params": {}},
        {"model_class": Ridge, "params": {"alpha": 0.1}},
        {"model_class": Ridge, "params": {"alpha": 1.0}},
        {"model_class": Ridge, "params": {"alpha": 10.0}},
        {"model_class": Lasso, "params": {"alpha": 0.1}},
        {"model_class": Lasso, "params": {"alpha": 1.0}},
        {"model_class": ElasticNet, "params": {"alpha": 0.1, "l1_ratio": 0.5}},
    ]

    tscv = TimeSeriesSplit(n_splits=3)
    best_rmse, best_config = float("inf"), param_grid[0]

    for config in param_grid:
        cv_rmses = []
        for tr_idx, val_idx in tscv.split(X_train):
            m = config["model_class"](**config["params"])
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = m.predict(X_train.iloc[val_idx])
            cv_rmses.append(np.sqrt(np.mean((y_train.iloc[val_idx] - preds) ** 2)))
        avg = np.mean(cv_rmses)
        if avg < best_rmse:
            best_rmse, best_config = avg, config

    model = best_config["model_class"](**best_config["params"])
    model.fit(X_train, y_train)
    if has_eval:
        predictions = model.predict(X_test)
        rmse, mae, mape = evaluate_preds(y_test.values, predictions, "LinearRegression")
    else:
        rmse = mae = mape = float("nan")

    filename = f"LinearRegression_{ticker_symbol}.pkl"
    joblib.dump(model, os.path.join(ticker_folder, filename))

    data_filename = f"LinearRegression_{ticker_symbol}_data.pkl"
    # Save the last few rows to enable recursive forecasting using lag features.
    max_lag = 5
    last_rows = train[feature_cols].tail(max_lag)
    last_close = train["Close"].tail(max_lag).values
    joblib.dump({
        "feature_cols": feature_cols,
        "last_rows": last_rows,
        "last_features": last_rows.iloc[-1].values,
        "last_close": last_close
    }, os.path.join(ticker_folder, data_filename))

    train_end = train.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, "LinearRegression", filename, "regression", train_end)

    return ModelResult(model_name="LinearRegression", model=model, rmse=rmse, mae=mae, mape=mape)


def random_forest_model(train: pd.DataFrame, test: Optional[pd.DataFrame], ticker: str = "EURUSD=X",
                        final_fit: bool = False) -> ModelResult:
    print("Training RandomForest model with hyperparameter tuning...")
    ticker_folder, ticker_symbol = get_ticker_folder(ticker)

    from sklearn.model_selection import TimeSeriesSplit

    feature_cols = [c for c in train.columns if c != "Close"]
    X_train, y_train = train[feature_cols], train["Close"]
    has_eval = (not final_fit and test is not None and len(test) > 0)
    if has_eval:
        X_test, y_test = test[feature_cols], test["Close"]

    param_grid = [
        {"n_estimators": 50, "max_depth": 5, "min_samples_split": 2},
        {"n_estimators": 50, "max_depth": 10, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
    ]

    tscv = TimeSeriesSplit(n_splits=3)
    best_rmse, best_params = float("inf"), param_grid[0]

    for params in param_grid:
        cv_rmses = []
        for tr_idx, val_idx in tscv.split(X_train):
            rf = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            rf.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = rf.predict(X_train.iloc[val_idx])
            cv_rmses.append(np.sqrt(np.mean((y_train.iloc[val_idx] - preds) ** 2)))
        avg = np.mean(cv_rmses)
        if avg < best_rmse:
            best_rmse, best_params = avg, params

    model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    if has_eval:
        predictions = model.predict(X_test)
        rmse, mae, mape = evaluate_preds(y_test.values, predictions, "RandomForest")
    else:
        rmse = mae = mape = float("nan")

    filename = f"RandomForest_{ticker_symbol}.pkl"
    joblib.dump(model, os.path.join(ticker_folder, filename))

    data_filename = f"RandomForest_{ticker_symbol}_data.pkl"
    # Save the last few rows to enable recursive forecasting using lag features.
    max_lag = 5
    last_rows = train[feature_cols].tail(max_lag)
    last_close = train["Close"].tail(max_lag).values
    joblib.dump({
        "feature_cols": feature_cols,
        "last_rows": last_rows,
        "last_features": last_rows.iloc[-1].values,
        "last_close": last_close
    }, os.path.join(ticker_folder, data_filename))

    train_end = train.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, "RandomForest", filename, "random_forest", train_end)

    return ModelResult(model_name="RandomForest", model=model, rmse=rmse, mae=mae, mape=mape)


# ─── High-Level Training Orchestrators ───────────────────────────────────────────

MODEL_TRAINERS = {
    "ARIMA": arima_model,
    "SARIMAX": sarimax_model,
    "SARIMA": sarimax_model,
    "LSTM": lstm_model,
    "GRU": gru_model,
    "Prophet": prophet_model,
    "LightGBM": lightgbm_model,
    "LinearRegression": linear_regression_model,
    "RandomForest": random_forest_model,
}


def _train_model_with_optional_hyperparameters(model_name: str,
                                               train_df: pd.DataFrame,
                                               test_df: Optional[pd.DataFrame],
                                               ticker: str,
                                               hyperparameters: Optional[dict] = None,
                                               final_fit: bool = False,
                                               tuning_requested: bool = False) -> ModelResult:
    """Train model either with default trainer or explicit hyperparameters."""
    model_upper = model_name.upper()

    if model_upper == "PROPHET":
        # Restrict CV search to explicit frontend tuning requests only.
        if tuning_requested and hyperparameters is not None:
            return _train_prophet_with_hp(train_df, test_df, ticker, hyperparameters, final_fit=final_fit)

        resolved_hp = hyperparameters if hyperparameters is not None else get_saved_best_hyperparameters(ticker, model_name)
        return prophet_model(train_df, test_df, ticker, final_fit=final_fit, hyperparameters=resolved_hp)

    if hyperparameters is None:
        trainer = MODEL_TRAINERS.get(model_name)
        if trainer is None:
            raise ValueError(f"Model '{model_name}' not recognized. Choose from: {list(MODEL_TRAINERS.keys())}")
        return trainer(train_df, test_df, ticker, final_fit=final_fit)

    if model_upper == "ARIMA":
        return _train_arima_with_hp(train_df, test_df, ticker, hyperparameters, final_fit=final_fit)
    if model_upper in ["SARIMAX", "SARIMA"]:
        return _train_sarimax_with_hp(train_df, test_df, ticker, hyperparameters, final_fit=final_fit)
    if model_upper == "LSTM":
        return _train_lstm_with_hp(train_df, test_df, ticker, hyperparameters, final_fit=final_fit)
    if model_upper == "GRU":
        return _train_gru_with_hp(train_df, test_df, ticker, hyperparameters, final_fit=final_fit)
    if model_upper == "LIGHTGBM":
        return _train_lightgbm_with_hp(train_df, test_df, ticker, hyperparameters, final_fit=final_fit)
    if model_upper == "LINEARREGRESSION":
        return _train_linear_with_hp(train_df, test_df, ticker, hyperparameters, final_fit=final_fit)
    if model_upper == "RANDOMFOREST":
        return _train_rf_with_hp(train_df, test_df, ticker, hyperparameters, final_fit=final_fit)
    raise ValueError(f"Model '{model_name}' not recognized. Choose from: {list(MODEL_TRAINERS.keys())}")


def train_model(start_date: str, end_date: str, ticker: str,
                train_size: float = 0.8, model: str = "ARIMA") -> ModelResult:
    """Fetch data, evaluate on split, then persist a full-range final model."""
    data = fetch_data(start_date, end_date, ticker)
    data = clean_data(data)
    data = feature_engineering(data)
    train, test = split_data(data, train_size)

    # 1) Evaluate on the requested train/test split.
    saved_hp = get_saved_best_hyperparameters(ticker, model) if model.upper() == "PROPHET" else None
    eval_result = _train_model_with_optional_hyperparameters(
        model,
        train,
        test,
        ticker,
        hyperparameters=saved_hp,
        tuning_requested=False,
    )

    # 2) Retrain and persist final artifacts on the full selected date range.
    if len(data) > 1:
        _train_model_with_optional_hyperparameters(
            model,
            data,
            None,
            ticker,
            hyperparameters=saved_hp,
            final_fit=True,
            tuning_requested=False,
        )

    # Keep data_info aligned with the selected training window.
    train_start = data.index[0].strftime("%Y-%m-%d")
    train_end = data.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, None, None, None, train_end, train_start_date=train_start)

    return eval_result


def train_all_models(start_date: str, end_date: str, ticker: str,
                     train_size: float = 0.8) -> dict:
    """
    Two-stage training:
    1. Train/test split for evaluation metrics.
    2. Retrain on full selected data for forecasting artifacts.
    """
    data = fetch_data(start_date, end_date, ticker)
    data = clean_data(data)
    data = feature_engineering(data)
    train_split, test_split = split_data(data, train_size)

    evaluation_results = {}
    for model_name, _ in MODEL_TRAINERS.items():
        if model_name == "SARIMA":
            continue  # Skip alias
        print(f"\n--- Evaluating {model_name} ---")
        try:
            saved_hp = get_saved_best_hyperparameters(ticker, model_name) if model_name.upper() == "PROPHET" else None
            result = _train_model_with_optional_hyperparameters(
                model_name,
                train_split,
                test_split,
                ticker,
                hyperparameters=saved_hp,
                tuning_requested=False,
            )
            evaluation_results[model_name] = {
                "rmse": result.rmse,
                "mae": result.mae,
                "mape": result.mape,
                "status": "success",
                "parameter_source": "saved_best_cv" if saved_hp else "default"
            }
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            evaluation_results[model_name] = {"status": "failed", "error": str(e)}

    # Retrain on full selected data range.
    train_full = data
    training_results = {}
    for model_name, _ in MODEL_TRAINERS.items():
        if model_name == "SARIMA":
            continue
        print(f"\n--- Retraining {model_name} on full data ---")
        try:
            saved_hp = get_saved_best_hyperparameters(ticker, model_name) if model_name.upper() == "PROPHET" else None
            _train_model_with_optional_hyperparameters(
                model_name,
                train_full,
                None,
                ticker,
                hyperparameters=saved_hp,
                final_fit=True,
                tuning_requested=False,
            )
            training_results[model_name] = {
                "status": "success",
                "parameter_source": "saved_best_cv" if saved_hp else "default"
            }
        except Exception as e:
            print(f"Error retraining {model_name}: {e}")
            training_results[model_name] = {"status": "failed", "error": str(e)}

    # Keep data_info aligned with the selected training window.
    if len(data) > 0:
        train_start = data.index[0].strftime("%Y-%m-%d")
        train_end = data.index[-1].strftime("%Y-%m-%d")
        update_model_metadata(ticker, None, None, None, train_end, train_start_date=train_start)

    return {"evaluation": evaluation_results, "training": training_results}


def train_model_with_hyperparameters(start_date: str, end_date: str, ticker: str,
                                     train_size: float = 0.8, model: str = "ARIMA",
                                     hyperparameters: dict = None) -> ModelResult:
    """
    Train a model with custom hyperparameters.
    
    Args:
        start_date: Training start date (YYYY-MM-DD)
        end_date: Training end date (YYYY-MM-DD)
        ticker: Ticker symbol (e.g., EURUSD=X)
        train_size: Train/test split ratio
        model: Model name
        hyperparameters: Dict of custom hyperparameter values
    
    Returns:
        ModelResult with training metrics
    """
    if hyperparameters is None:
        hyperparameters = {}
    
    data = fetch_data(start_date, end_date, ticker)
    data = clean_data(data)
    data = feature_engineering(data)
    train, test = split_data(data, train_size)

    # 1) Evaluate on requested split and select hyperparameters.
    eval_result = _train_model_with_optional_hyperparameters(
        model,
        train,
        test,
        ticker,
        hyperparameters=hyperparameters,
        tuning_requested=True,
    )

    # 2) Retrain final artifact on full selected date range with chosen parameters.
    selected_hp = hyperparameters
    selected_cv_rmse = None
    if getattr(eval_result, "artifacts", None) and isinstance(eval_result.artifacts, dict):
        selected_hp = eval_result.artifacts.get("best_hyperparameters", hyperparameters)
        selected_cv_rmse = eval_result.artifacts.get("cv_rmse")

    # Persist best parameters so future train_all runs can reuse them.
    update_model_parameters(ticker, model, selected_hp, cv_rmse=selected_cv_rmse)

    if len(data) > 1:
        _train_model_with_optional_hyperparameters(
            model,
            data,
            None,
            ticker,
            hyperparameters=selected_hp,
            final_fit=True,
            tuning_requested=True,
        )

    # Keep data_info aligned with the selected training window.
    train_start = data.index[0].strftime("%Y-%m-%d")
    train_end = data.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, None, None, None, train_end, train_start_date=train_start)

    return eval_result


def _train_arima_with_hp(train, test, ticker, hp, final_fit: bool = False):
    """Train ARIMA with frontend-provided grid and time-series CV."""
    ticker_folder, ticker_symbol = get_ticker_folder(ticker)
    if final_fit:
        best_params = {
            "p": int(hp.get("p", 1)),
            "d": int(hp.get("d", 1)),
            "q": int(hp.get("q", 1))
        }
        best_cv_rmse = float("nan")
    else:
        grid = expand_hyperparameter_grid(hp)
        splits = _build_time_series_splits(len(train), max_splits=3, min_train_size=40, min_val_size=5)

        best_params = None
        best_cv_rmse = float("inf")

        for params in grid:
            p = int(params.get("p", 1))
            d = int(params.get("d", 1))
            q = int(params.get("q", 1))
            order = (p, d, q)
            cv_scores = []

            try:
                if splits:
                    for tr_idx, val_idx in splits:
                        fold_train = np.asarray(train.iloc[tr_idx]["Close"], dtype=np.float64)
                        fold_val = train.iloc[val_idx]["Close"]
                        model_fit = ARIMA(fold_train, order=order).fit()
                        pred = np.asarray(model_fit.forecast(steps=len(fold_val)), dtype=np.float64)
                        cv_scores.append(_rmse(fold_val.values, pred))
                else:
                    model_fit = ARIMA(np.asarray(train["Close"], dtype=np.float64), order=order).fit()
                    pred = np.asarray(model_fit.forecast(steps=len(test)), dtype=np.float64)
                    cv_scores.append(_rmse(test["Close"].values, pred))
            except Exception as e:
                print(f"ARIMA CV error {order}: {e}")
                continue

            avg_cv_rmse = float(np.mean(cv_scores))
            if avg_cv_rmse < best_cv_rmse:
                best_cv_rmse = avg_cv_rmse
                best_params = {"p": p, "d": d, "q": q}

        if best_params is None:
            raise ValueError("No valid ARIMA hyperparameter combination found during CV.")

    final_order = (best_params["p"], best_params["d"], best_params["q"])
    final_model = ARIMA(np.asarray(train["Close"], dtype=np.float64), order=final_order).fit()
    if final_fit or test is None or len(test) == 0:
        rmse = mae = mape = float("nan")
    else:
        predictions = np.asarray(final_model.forecast(steps=len(test)), dtype=np.float64)
        rmse, mae, mape = evaluate_preds(test["Close"].values, predictions, "ARIMA")

    filename = f"ARIMA_{ticker_symbol}.pkl"
    joblib.dump(final_model, os.path.join(ticker_folder, filename))
    train_end = train.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, "ARIMA", filename, "arima", train_end)

    return ModelResult(model_name="ARIMA", model=final_model, rmse=rmse, mae=mae, mape=mape,
                       artifacts={"best_hyperparameters": best_params, "cv_rmse": best_cv_rmse})


def _train_sarimax_with_hp(train, test, ticker, hp, final_fit: bool = False):
    """Train SARIMAX with frontend-provided grid and time-series CV."""
    ticker_folder, ticker_symbol = get_ticker_folder(ticker)
    if final_fit:
        best_params = {
            "p": int(hp.get("p", 1)),
            "d": int(hp.get("d", 1)),
            "q": int(hp.get("q", 1)),
            "seasonal_p": int(hp.get("seasonal_p", 1)),
            "seasonal_d": int(hp.get("seasonal_d", 1)),
            "seasonal_q": int(hp.get("seasonal_q", 1)),
            "s": int(hp.get("s", 12))
        }
        best_cv_rmse = float("nan")
    else:
        grid = expand_hyperparameter_grid(hp)
        splits = _build_time_series_splits(len(train), max_splits=3, min_train_size=50, min_val_size=5)

        best_params = None
        best_cv_rmse = float("inf")

        for params in grid:
            p = int(params.get("p", 1))
            d = int(params.get("d", 1))
            q = int(params.get("q", 1))
            seasonal_p = int(params.get("seasonal_p", 1))
            seasonal_d = int(params.get("seasonal_d", 1))
            seasonal_q = int(params.get("seasonal_q", 1))
            s = int(params.get("s", 12))
            order = (p, d, q)
            seasonal_order = (seasonal_p, seasonal_d, seasonal_q, s)
            cv_scores = []

            try:
                if splits:
                    for tr_idx, val_idx in splits:
                        fold_train = np.asarray(train.iloc[tr_idx]["Close"], dtype=np.float64)
                        fold_val = train.iloc[val_idx]["Close"]
                        model_fit = SARIMAX(fold_train, order=order, seasonal_order=seasonal_order).fit(disp=False)
                        pred = np.asarray(model_fit.forecast(steps=len(fold_val)), dtype=np.float64)
                        cv_scores.append(_rmse(fold_val.values, pred))
                else:
                    model_fit = SARIMAX(np.asarray(train["Close"], dtype=np.float64), order=order, seasonal_order=seasonal_order).fit(disp=False)
                    pred = np.asarray(model_fit.forecast(steps=len(test)), dtype=np.float64)
                    cv_scores.append(_rmse(test["Close"].values, pred))
            except Exception as e:
                print(f"SARIMAX CV error {order} x {seasonal_order}: {e}")
                continue

            avg_cv_rmse = float(np.mean(cv_scores))
            if avg_cv_rmse < best_cv_rmse:
                best_cv_rmse = avg_cv_rmse
                best_params = {
                    "p": p,
                    "d": d,
                    "q": q,
                    "seasonal_p": seasonal_p,
                    "seasonal_d": seasonal_d,
                    "seasonal_q": seasonal_q,
                    "s": s
                }

        if best_params is None:
            raise ValueError("No valid SARIMAX hyperparameter combination found during CV.")

    final_model = SARIMAX(
        np.asarray(train["Close"], dtype=np.float64),
        order=(best_params["p"], best_params["d"], best_params["q"]),
        seasonal_order=(
            best_params["seasonal_p"],
            best_params["seasonal_d"],
            best_params["seasonal_q"],
            best_params["s"]
        )
    ).fit(disp=False)
    if final_fit or test is None or len(test) == 0:
        rmse = mae = mape = float("nan")
    else:
        predictions = np.asarray(final_model.forecast(steps=len(test)), dtype=np.float64)
        rmse, mae, mape = evaluate_preds(test["Close"].values, predictions, "SARIMAX")

    # Keep filename aligned with forecasting loader, which expects SARIMA_*.pkl
    filename = f"SARIMA_{ticker_symbol}.pkl"
    joblib.dump(final_model, os.path.join(ticker_folder, filename))
    train_end = train.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, "SARIMA", filename, "sarima", train_end)

    return ModelResult(model_name="SARIMAX", model=final_model, rmse=rmse, mae=mae, mape=mape,
                       artifacts={"best_hyperparameters": best_params, "cv_rmse": best_cv_rmse})


def _train_lstm_with_hp(train, test, ticker, hp, final_fit: bool = False):
    """Train LSTM with frontend-provided grid and time-series CV."""
    if final_fit:
        best_params = {
            "hidden_layer_size": int(hp.get("hidden_layer_size", 50)),
            "num_layers": int(hp.get("num_layers", 2)),
            "dropout": float(hp.get("dropout", 0.2)),
            "learning_rate": float(hp.get("learning_rate", 0.01)),
            "epochs": int(hp.get("epochs", 50)),
            "sequence_length": int(hp.get("sequence_length", 30))
        }
        best_cv_rmse = float("nan")
    else:
        grid = expand_hyperparameter_grid(hp)
        splits = _build_time_series_splits(len(train), max_splits=3, min_train_size=50, min_val_size=10)

        best_params = None
        best_cv_rmse = float("inf")

        for params in grid:
            hidden_layer_size = int(params.get("hidden_layer_size", 50))
            num_layers = int(params.get("num_layers", 2))
            dropout = float(params.get("dropout", 0.2))
            learning_rate = float(params.get("learning_rate", 0.01))
            epochs = int(params.get("epochs", 50))
            sequence_length = int(params.get("sequence_length", 30))

            cv_scores = []
            try:
                if splits:
                    for tr_idx, val_idx in splits:
                        fold_train = train.iloc[tr_idx]
                        fold_val = train.iloc[val_idx]
                        if len(fold_train) <= sequence_length or len(fold_val) == 0:
                            continue
                        fold_result = _train_sequence_model(
                            LSTMModel, "LSTM", fold_train, fold_val, ticker,
                            hidden_layer_size=hidden_layer_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            sequence_length=sequence_length,
                            save_artifacts=False
                        )
                        cv_scores.append(fold_result.rmse)
                else:
                    if test is not None and len(train) > sequence_length and len(test) > 0:
                        fallback_result = _train_sequence_model(
                            LSTMModel, "LSTM", train, test, ticker,
                            hidden_layer_size=hidden_layer_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            sequence_length=sequence_length,
                            save_artifacts=False
                        )
                        cv_scores.append(fallback_result.rmse)
            except Exception as e:
                print(f"LSTM CV error ({params}): {e}")
                continue

            if not cv_scores:
                continue

            avg_cv_rmse = float(np.mean(cv_scores))
            if avg_cv_rmse < best_cv_rmse:
                best_cv_rmse = avg_cv_rmse
                best_params = {
                    "hidden_layer_size": hidden_layer_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "sequence_length": sequence_length
                }

        if best_params is None:
            raise ValueError("No valid LSTM hyperparameter combination found during CV.")

    best_result = _train_sequence_model(
        LSTMModel, "LSTM", train, test, ticker,
        hidden_layer_size=best_params["hidden_layer_size"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
        epochs=best_params["epochs"],
        sequence_length=best_params["sequence_length"],
        save_artifacts=False,
        final_fit=final_fit
    )

    ticker_folder, ticker_symbol = get_ticker_folder(ticker)
    pth_filename = f"LSTM_{ticker_symbol}.pth"
    torch.save(best_result.model.state_dict(), os.path.join(ticker_folder, pth_filename))
    data_filename = f"LSTM_{ticker_symbol}_data.pkl"
    joblib.dump(best_result.artifacts, os.path.join(ticker_folder, data_filename))
    train_end = train.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, "LSTM", pth_filename, "lstm", train_end)

    best_result.artifacts = {
        **(best_result.artifacts or {}),
        "best_hyperparameters": best_params,
        "cv_rmse": best_cv_rmse
    }
    return best_result


def _train_gru_with_hp(train, test, ticker, hp, final_fit: bool = False):
    """Train GRU with frontend-provided grid and time-series CV."""
    if final_fit:
        best_params = {
            "hidden_layer_size": int(hp.get("hidden_layer_size", 50)),
            "num_layers": int(hp.get("num_layers", 2)),
            "dropout": float(hp.get("dropout", 0.2)),
            "learning_rate": float(hp.get("learning_rate", 0.01)),
            "epochs": int(hp.get("epochs", 50)),
            "sequence_length": int(hp.get("sequence_length", 30))
        }
        best_cv_rmse = float("nan")
    else:
        grid = expand_hyperparameter_grid(hp)
        splits = _build_time_series_splits(len(train), max_splits=3, min_train_size=50, min_val_size=10)

        best_params = None
        best_cv_rmse = float("inf")

        for params in grid:
            hidden_layer_size = int(params.get("hidden_layer_size", 50))
            num_layers = int(params.get("num_layers", 2))
            dropout = float(params.get("dropout", 0.2))
            learning_rate = float(params.get("learning_rate", 0.01))
            epochs = int(params.get("epochs", 50))
            sequence_length = int(params.get("sequence_length", 30))

            cv_scores = []
            try:
                if splits:
                    for tr_idx, val_idx in splits:
                        fold_train = train.iloc[tr_idx]
                        fold_val = train.iloc[val_idx]
                        if len(fold_train) <= sequence_length or len(fold_val) == 0:
                            continue
                        fold_result = _train_sequence_model(
                            GRUModel, "GRU", fold_train, fold_val, ticker,
                            hidden_layer_size=hidden_layer_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            sequence_length=sequence_length,
                            save_artifacts=False
                        )
                        cv_scores.append(fold_result.rmse)
                else:
                    if test is not None and len(train) > sequence_length and len(test) > 0:
                        fallback_result = _train_sequence_model(
                            GRUModel, "GRU", train, test, ticker,
                            hidden_layer_size=hidden_layer_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            sequence_length=sequence_length,
                            save_artifacts=False
                        )
                        cv_scores.append(fallback_result.rmse)
            except Exception as e:
                print(f"GRU CV error ({params}): {e}")
                continue

            if not cv_scores:
                continue

            avg_cv_rmse = float(np.mean(cv_scores))
            if avg_cv_rmse < best_cv_rmse:
                best_cv_rmse = avg_cv_rmse
                best_params = {
                    "hidden_layer_size": hidden_layer_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "sequence_length": sequence_length
                }

        if best_params is None:
            raise ValueError("No valid GRU hyperparameter combination found during CV.")

    best_result = _train_sequence_model(
        GRUModel, "GRU", train, test, ticker,
        hidden_layer_size=best_params["hidden_layer_size"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
        epochs=best_params["epochs"],
        sequence_length=best_params["sequence_length"],
        save_artifacts=False,
        final_fit=final_fit
    )

    ticker_folder, ticker_symbol = get_ticker_folder(ticker)
    pth_filename = f"GRU_{ticker_symbol}.pth"
    torch.save(best_result.model.state_dict(), os.path.join(ticker_folder, pth_filename))
    data_filename = f"GRU_{ticker_symbol}_data.pkl"
    joblib.dump(best_result.artifacts, os.path.join(ticker_folder, data_filename))
    train_end = train.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, "GRU", pth_filename, "gru", train_end)

    best_result.artifacts = {
        **(best_result.artifacts or {}),
        "best_hyperparameters": best_params,
        "cv_rmse": best_cv_rmse
    }
    return best_result


def _train_prophet_with_hp(train, test, ticker, hp, final_fit: bool = False):
    """Train Prophet with frontend-provided grid and time-series CV."""
    ticker_folder, ticker_symbol = get_ticker_folder(ticker)
    if final_fit:
        best_params = {
            "changepoint_prior_scale": float(hp.get("changepoint_prior_scale", 0.05)),
            "seasonality_prior_scale": float(hp.get("seasonality_prior_scale", 10.0)),
            "seasonality_mode": hp.get("seasonality_mode", "multiplicative"),
            "daily_seasonality": _coerce_bool(hp.get("daily_seasonality", False), False),
            "weekly_seasonality": _coerce_bool(hp.get("weekly_seasonality", True), True),
            "yearly_seasonality": _coerce_bool(hp.get("yearly_seasonality", True), True)
        }
        best_cv_rmse = float("nan")
    else:
        grid = expand_hyperparameter_grid(hp)
        splits = _build_time_series_splits(len(train), max_splits=3, min_train_size=50, min_val_size=7)

        best_params = None
        best_cv_rmse = float("inf")

        for params in grid:
            candidate_params = {
            "changepoint_prior_scale": float(params.get("changepoint_prior_scale", 0.05)),
            "seasonality_prior_scale": float(params.get("seasonality_prior_scale", 10.0)),
            "seasonality_mode": params.get("seasonality_mode", "multiplicative"),
            "daily_seasonality": _coerce_bool(params.get("daily_seasonality", False), False),
            "weekly_seasonality": _coerce_bool(params.get("weekly_seasonality", True), True),
            "yearly_seasonality": _coerce_bool(params.get("yearly_seasonality", True), True)
            }

            cv_scores = []
            try:
                if splits:
                    for tr_idx, val_idx in splits:
                        fold_train_df = _to_prophet_df(train.iloc[tr_idx])
                        fold_val_df = _to_prophet_df(train.iloc[val_idx])

                        model = Prophet(**candidate_params)
                        model.add_country_holidays(country_name="US")
                        model.fit(fold_train_df)

                        fold_pred = model.predict(fold_val_df[["ds"]])["yhat"].values
                        cv_scores.append(_rmse(fold_val_df["y"].values, fold_pred))
                else:
                    train_df = _to_prophet_df(train)
                    test_df = _to_prophet_df(test)
                    model = Prophet(**candidate_params)
                    model.add_country_holidays(country_name="US")
                    model.fit(train_df)
                    pred = model.predict(test_df[["ds"]])["yhat"].values
                    cv_scores.append(_rmse(test_df["y"].values, pred))
            except Exception as e:
                print(f"Prophet CV error ({candidate_params}): {e}")
                continue

            avg_cv_rmse = float(np.mean(cv_scores))
            if avg_cv_rmse < best_cv_rmse:
                best_cv_rmse = avg_cv_rmse
                best_params = candidate_params

        if best_params is None:
            raise ValueError("No valid Prophet hyperparameter combination found during CV.")

    train_df = _to_prophet_df(train)
    test_df = _to_prophet_df(test) if (test is not None and len(test) > 0) else None
    final_model = Prophet(**best_params)
    final_model.add_country_holidays(country_name="US")
    final_model.fit(train_df)

    if final_fit or test_df is None or len(test_df) == 0:
        rmse = mae = mape = float("nan")
    else:
        predictions = final_model.predict(test_df[["ds"]])["yhat"].values
        rmse, mae, mape = evaluate_preds(test_df["y"].values, predictions, "Prophet")

    filename = f"PROPHET_{ticker_symbol}.pkl"
    joblib.dump(final_model, os.path.join(ticker_folder, filename))
    train_end = train.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, "Prophet", filename, "prophet", train_end)

    return ModelResult(model_name="Prophet", model=final_model, rmse=rmse, mae=mae, mape=mape,
                       artifacts={"best_hyperparameters": best_params, "cv_rmse": best_cv_rmse})


def _train_lightgbm_with_hp(train, test, ticker, hp, final_fit: bool = False):
    """Train LightGBM with frontend-provided grid and time-series CV."""
    ticker_folder, ticker_symbol = get_ticker_folder(ticker)
    feature_cols = _get_feature_columns(train)

    train_ready = train[feature_cols + ["Close"]].dropna()
    test_ready = None
    if not final_fit:
        test_ready = test[feature_cols + ["Close"]].dropna()
    if train_ready.empty or (not final_fit and test_ready.empty):
        raise ValueError("Insufficient non-null rows for LightGBM tuning.")

    X_train = train_ready[feature_cols]
    y_train = train_ready["Close"]
    has_eval = (not final_fit and test_ready is not None and len(test_ready) > 0)
    if has_eval:
        X_test = test_ready[feature_cols]
        y_test = test_ready["Close"]

    if final_fit:
        best_params = {
            "n_estimators": int(hp.get("n_estimators", 100)),
            "learning_rate": float(hp.get("learning_rate", 0.1)),
            "num_leaves": int(hp.get("num_leaves", 31)),
            "min_child_samples": int(hp.get("min_child_samples", 20)),
            "subsample": float(hp.get("subsample", 1.0)),
            "colsample_bytree": float(hp.get("colsample_bytree", 1.0)),
            "random_state": 42,
            "verbose": -1
        }
        max_depth = _coerce_optional_int(hp.get("max_depth", -1))
        if max_depth is not None and max_depth != -1:
            best_params["max_depth"] = max_depth
        best_cv_rmse = float("nan")
    else:
        grid = expand_hyperparameter_grid(hp)
        splits = _build_time_series_splits(len(X_train), max_splits=3, min_train_size=40, min_val_size=5)

        best_params = None
        best_cv_rmse = float("inf")

        for params in grid:
            candidate_params = {
            "n_estimators": int(params.get("n_estimators", 100)),
            "learning_rate": float(params.get("learning_rate", 0.1)),
            "num_leaves": int(params.get("num_leaves", 31)),
            "min_child_samples": int(params.get("min_child_samples", 20)),
            "subsample": float(params.get("subsample", 1.0)),
            "colsample_bytree": float(params.get("colsample_bytree", 1.0)),
            "random_state": 42,
            "verbose": -1
            }
            max_depth = _coerce_optional_int(params.get("max_depth", -1))
            if max_depth is not None and max_depth != -1:
                candidate_params["max_depth"] = max_depth

            cv_scores = []
            try:
                if splits:
                    for tr_idx, val_idx in splits:
                        m = lgb.LGBMRegressor(**candidate_params)
                        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                        pred = m.predict(X_train.iloc[val_idx])
                        cv_scores.append(_rmse(y_train.iloc[val_idx].values, pred))
                else:
                    m = lgb.LGBMRegressor(**candidate_params)
                    m.fit(X_train, y_train)
                    pred = m.predict(X_test)
                    cv_scores.append(_rmse(y_test.values, pred))
            except Exception as e:
                print(f"LightGBM CV error ({candidate_params}): {e}")
                continue

            avg_cv_rmse = float(np.mean(cv_scores))
            if avg_cv_rmse < best_cv_rmse:
                best_cv_rmse = avg_cv_rmse
                best_params = candidate_params

        if best_params is None:
            raise ValueError("No valid LightGBM hyperparameter combination found during CV.")

    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    if has_eval:
        predictions = model.predict(X_test)
        rmse, mae, mape = evaluate_preds(y_test.values, predictions, "LightGBM")
    else:
        rmse = mae = mape = float("nan")

    lgbm_filename = f"LightGBM_{ticker_symbol}.txt"
    _save_validated_lightgbm_model(model.booster_, os.path.join(ticker_folder, lgbm_filename))
    data_filename = f"LightGBM_{ticker_symbol}_data.pkl"
    _atomic_joblib_dump({
        "feature_cols": feature_cols,
        "last_features": X_train.iloc[-1].values,
        "train_data": train_ready[feature_cols + ["Close"]].values
    }, os.path.join(ticker_folder, data_filename))

    train_end = train.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, "LightGBM", lgbm_filename, "lightgbm", train_end)

    return ModelResult(model_name="LightGBM", model=model, rmse=rmse, mae=mae, mape=mape,
                       artifacts={"best_hyperparameters": best_params, "cv_rmse": best_cv_rmse})


def _train_linear_with_hp(train, test, ticker, hp, final_fit: bool = False):
    """Train LinearRegression with frontend-provided grid and time-series CV."""
    ticker_folder, ticker_symbol = get_ticker_folder(ticker)
    feature_cols = _get_feature_columns(train)

    train_ready = train[feature_cols + ["Close"]].dropna()
    test_ready = None
    if not final_fit:
        test_ready = test[feature_cols + ["Close"]].dropna()
    if train_ready.empty or (not final_fit and test_ready.empty):
        raise ValueError("Insufficient non-null rows for LinearRegression tuning.")

    X_train = train_ready[feature_cols]
    y_train = train_ready["Close"]
    has_eval = (not final_fit and test_ready is not None and len(test_ready) > 0)
    if has_eval:
        X_test = test_ready[feature_cols]
        y_test = test_ready["Close"]

    if final_fit:
        best_params = {
            "fit_intercept": _coerce_bool(hp.get("fit_intercept", True), True)
        }
        best_cv_rmse = float("nan")
    else:
        grid = expand_hyperparameter_grid(hp)
        splits = _build_time_series_splits(len(X_train), max_splits=3, min_train_size=30, min_val_size=5)

        best_params = None
        best_cv_rmse = float("inf")

        for params in grid:
            candidate_params = {
                "fit_intercept": _coerce_bool(params.get("fit_intercept", True), True)
            }

            cv_scores = []
            try:
                if splits:
                    for tr_idx, val_idx in splits:
                        m = LinearRegression(**candidate_params)
                        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                        pred = m.predict(X_train.iloc[val_idx])
                        cv_scores.append(_rmse(y_train.iloc[val_idx].values, pred))
                else:
                    m = LinearRegression(**candidate_params)
                    m.fit(X_train, y_train)
                    pred = m.predict(X_test)
                    cv_scores.append(_rmse(y_test.values, pred))
            except Exception as e:
                print(f"LinearRegression CV error ({candidate_params}): {e}")
                continue

            avg_cv_rmse = float(np.mean(cv_scores))
            if avg_cv_rmse < best_cv_rmse:
                best_cv_rmse = avg_cv_rmse
                best_params = candidate_params

        if best_params is None:
            raise ValueError("No valid LinearRegression hyperparameter combination found during CV.")

    model = LinearRegression(**best_params)
    model.fit(X_train, y_train)
    if has_eval:
        predictions = model.predict(X_test)
        rmse, mae, mape = evaluate_preds(y_test.values, predictions, "LinearRegression")
    else:
        rmse = mae = mape = float("nan")

    filename = f"LinearRegression_{ticker_symbol}.pkl"
    joblib.dump(model, os.path.join(ticker_folder, filename))

    data_filename = f"LinearRegression_{ticker_symbol}_data.pkl"
    max_lag = 5
    last_rows = train_ready[feature_cols].tail(max_lag)
    last_close = train_ready["Close"].tail(max_lag).values
    joblib.dump({
        "feature_cols": feature_cols,
        "last_rows": last_rows,
        "last_features": last_rows.iloc[-1].values,
        "last_close": last_close
    }, os.path.join(ticker_folder, data_filename))

    train_end = train.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, "LinearRegression", filename, "linear_regression", train_end)

    return ModelResult(model_name="LinearRegression", model=model, rmse=rmse, mae=mae, mape=mape,
                       artifacts={"best_hyperparameters": best_params, "cv_rmse": best_cv_rmse})


def _train_rf_with_hp(train, test, ticker, hp, final_fit: bool = False):
    """Train RandomForest with frontend-provided grid and time-series CV."""
    ticker_folder, ticker_symbol = get_ticker_folder(ticker)
    feature_cols = _get_feature_columns(train)

    train_ready = train[feature_cols + ["Close"]].dropna()
    test_ready = None
    if not final_fit:
        test_ready = test[feature_cols + ["Close"]].dropna()
    if train_ready.empty or (not final_fit and test_ready.empty):
        raise ValueError("Insufficient non-null rows for RandomForest tuning.")

    X_train = train_ready[feature_cols]
    y_train = train_ready["Close"]
    has_eval = (not final_fit and test_ready is not None and len(test_ready) > 0)
    if has_eval:
        X_test = test_ready[feature_cols]
        y_test = test_ready["Close"]

    if final_fit:
        max_depth = _coerce_optional_int(hp.get("max_depth", None))
        max_features = hp.get("max_features", "sqrt")
        best_params = {
            "n_estimators": int(hp.get("n_estimators", 100)),
            "min_samples_split": int(hp.get("min_samples_split", 2)),
            "min_samples_leaf": int(hp.get("min_samples_leaf", 1)),
            "random_state": 42,
            "n_jobs": -1
        }
        if max_depth is not None:
            best_params["max_depth"] = max_depth
        if max_features not in (None, "None"):
            best_params["max_features"] = max_features
        best_cv_rmse = float("nan")
    else:
        grid = expand_hyperparameter_grid(hp)
        splits = _build_time_series_splits(len(X_train), max_splits=3, min_train_size=30, min_val_size=5)

        best_params = None
        best_cv_rmse = float("inf")

        for params in grid:
            max_depth = _coerce_optional_int(params.get("max_depth", None))
            max_features = params.get("max_features", "sqrt")
            candidate_params = {
                "n_estimators": int(params.get("n_estimators", 100)),
                "min_samples_split": int(params.get("min_samples_split", 2)),
                "min_samples_leaf": int(params.get("min_samples_leaf", 1)),
                "random_state": 42,
                "n_jobs": -1
            }
            if max_depth is not None:
                candidate_params["max_depth"] = max_depth
            if max_features not in (None, "None"):
                candidate_params["max_features"] = max_features

            cv_scores = []
            try:
                if splits:
                    for tr_idx, val_idx in splits:
                        m = RandomForestRegressor(**candidate_params)
                        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                        pred = m.predict(X_train.iloc[val_idx])
                        cv_scores.append(_rmse(y_train.iloc[val_idx].values, pred))
                else:
                    m = RandomForestRegressor(**candidate_params)
                    m.fit(X_train, y_train)
                    pred = m.predict(X_test)
                    cv_scores.append(_rmse(y_test.values, pred))
            except Exception as e:
                print(f"RandomForest CV error ({candidate_params}): {e}")
                continue

            avg_cv_rmse = float(np.mean(cv_scores))
            if avg_cv_rmse < best_cv_rmse:
                best_cv_rmse = avg_cv_rmse
                best_params = candidate_params

        if best_params is None:
            raise ValueError("No valid RandomForest hyperparameter combination found during CV.")

    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)
    if has_eval:
        predictions = model.predict(X_test)
        rmse, mae, mape = evaluate_preds(y_test.values, predictions, "RandomForest")
    else:
        rmse = mae = mape = float("nan")

    filename = f"RandomForest_{ticker_symbol}.pkl"
    joblib.dump(model, os.path.join(ticker_folder, filename))

    data_filename = f"RandomForest_{ticker_symbol}_data.pkl"
    max_lag = 5
    last_rows = train_ready[feature_cols].tail(max_lag)
    last_close = train_ready["Close"].tail(max_lag).values
    joblib.dump({
        "feature_cols": feature_cols,
        "last_rows": last_rows,
        "last_features": last_rows.iloc[-1].values,
        "last_close": last_close
    }, os.path.join(ticker_folder, data_filename))

    train_end = train.index[-1].strftime("%Y-%m-%d")
    update_model_metadata(ticker, "RandomForest", filename, "random_forest", train_end)

    return ModelResult(model_name="RandomForest", model=model, rmse=rmse, mae=mae, mape=mape,
                       artifacts={"best_hyperparameters": best_params, "cv_rmse": best_cv_rmse})


# ─── Forecasting ─────────────────────────────────────────────────────────────────

def generate_forecast(model: str, ticker: str, start_date: str, steps: int) -> dict:
    """
    Generate future price forecasts using a pre-trained model.

    Args:
        model: Model name (ARIMA, SARIMAX, Prophet, LightGBM, LSTM, GRU,
                           LinearRegression, RandomForest)
        ticker: Ticker symbol (e.g., 'EURUSD=X')
        start_date: Last known date (YYYY-MM-DD) — forecasts start from day+1
        steps: Number of future days to forecast

    Returns:
        dict with 'dates' and 'predictions' keys (and optionally 'lower'/'upper')
    """
    ticker_folder, ticker_symbol = get_ticker_folder(ticker)
    last_date = pd.to_datetime(start_date)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq="D")
    model_lower = model.lower()

    if model_lower == "prophet":
        m = joblib.load(os.path.join(ticker_folder, f"PROPHET_{ticker_symbol}.pkl"))
        future_df = pd.DataFrame({"ds": future_dates})
        fc = m.predict(future_df)
        return {
            "dates": future_dates.strftime("%Y-%m-%d").tolist(),
            "predictions": fc["yhat"].tolist(),
            "lower": fc["yhat_lower"].tolist(),
            "upper": fc["yhat_upper"].tolist()
        }

    elif model_lower == "arima":
        m = joblib.load(os.path.join(ticker_folder, f"ARIMA_{ticker_symbol}.pkl"))
        try:
            fc = np.asarray(m.forecast(steps=steps), dtype=np.float64)
        except (TypeError, ValueError):
            # pandas 2.3+ / statsmodels compat: re-initialize with numpy endog and apply saved params
            endog_np = np.asarray(m.model.endog, dtype=np.float64).ravel()
            fresh_res = ARIMA(endog_np, order=m.model.order).smooth(
                np.asarray(m.params, dtype=np.float64)
            )
            fc = np.asarray(fresh_res.forecast(steps=steps), dtype=np.float64)
        return {"dates": future_dates.strftime("%Y-%m-%d").tolist(), "predictions": fc.tolist()}

    elif model_lower in ("sarimax", "sarima"):
        m = joblib.load(os.path.join(ticker_folder, f"SARIMA_{ticker_symbol}.pkl"))
        try:
            fc = np.asarray(m.forecast(steps=steps), dtype=np.float64)
        except (TypeError, ValueError):
            # pandas 2.3+ / statsmodels compat: re-initialize with numpy endog and apply saved params
            endog_np = np.asarray(m.model.endog, dtype=np.float64).ravel()
            seasonal_order = getattr(m.model, "seasonal_order", (0, 0, 0, 0))
            fresh_res = SARIMAX(
                endog_np, order=m.model.order, seasonal_order=seasonal_order
            ).smooth(np.asarray(m.params, dtype=np.float64))
            fc = np.asarray(fresh_res.forecast(steps=steps), dtype=np.float64)
        return {"dates": future_dates.strftime("%Y-%m-%d").tolist(), "predictions": fc.tolist()}

    elif model_lower == "lightgbm":
        model_path = os.path.join(ticker_folder, f"LightGBM_{ticker_symbol}.txt")
        booster = _load_validated_lightgbm_model(model_path)
        data = joblib.load(os.path.join(ticker_folder, f"LightGBM_{ticker_symbol}_data.pkl"))
        current_features = data["last_features"].copy()
        predictions = []
        for _ in range(steps):
            pred = booster.predict(current_features.reshape(1, -1))[0]
            predictions.append(pred)
            current_features = np.roll(current_features, -1)
            current_features[-1] = pred
        return {"dates": future_dates.strftime("%Y-%m-%d").tolist(), "predictions": predictions}

    elif model_lower == "lstm":
        m = LSTMModel()
        pth_path = os.path.join(ticker_folder, f"LSTM_{ticker_symbol}.pth")
        if os.path.exists(pth_path):
            m.load_state_dict(torch.load(pth_path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"LSTM weights not found: {pth_path}")
        m.eval()
        data = joblib.load(os.path.join(ticker_folder, f"LSTM_{ticker_symbol}_data.pkl"))
        scaler, current_seq = data["scaler"], data["last_sequence"].copy()
        predictions = []
        seq_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            for _ in range(steps):
                pred = m(seq_tensor)
                predictions.append(pred.item())
                seq_tensor = torch.cat((seq_tensor[:, 1:, :], pred.unsqueeze(1)), dim=1)
        predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return {"dates": future_dates.strftime("%Y-%m-%d").tolist(),
                "predictions": predictions_inv.flatten().tolist()}

    elif model_lower == "gru":
        m = GRUModel()
        pth_path = os.path.join(ticker_folder, f"GRU_{ticker_symbol}.pth")
        if os.path.exists(pth_path):
            m.load_state_dict(torch.load(pth_path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"GRU weights not found: {pth_path}")
        m.eval()
        data = joblib.load(os.path.join(ticker_folder, f"GRU_{ticker_symbol}_data.pkl"))
        scaler, current_seq = data["scaler"], data["last_sequence"].copy()
        predictions = []
        seq_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            for _ in range(steps):
                pred = m(seq_tensor)
                predictions.append(pred.item())
                seq_tensor = torch.cat((seq_tensor[:, 1:, :], pred.unsqueeze(1)), dim=1)
        predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return {"dates": future_dates.strftime("%Y-%m-%d").tolist(),
                "predictions": predictions_inv.flatten().tolist()}

    elif model_lower == "linearregression":
        m = joblib.load(os.path.join(ticker_folder, f"LinearRegression_{ticker_symbol}.pkl"))
        data = joblib.load(os.path.join(ticker_folder, f"LinearRegression_{ticker_symbol}_data.pkl"))
        feature_cols = data["feature_cols"]

        # Recursive forecasting using lag features (if available).
        last_rows = data.get("last_rows")
        if last_rows is None or len(last_rows) == 0:
            future_features = np.tile(data["last_features"], (steps, 1))
            fc = m.predict(future_features)
            return {"dates": future_dates.strftime("%Y-%m-%d").tolist(), "predictions": fc.tolist()}

        max_lag = max((int(c.split("_")[1]) for c in feature_cols if c.startswith("Lag_")), default=0)
        last_rows = last_rows.copy()
        last_close = data.get("last_close")
        preds = []

        for step in range(steps):
            current = last_rows.iloc[-1].copy()
            for lag in range(1, max_lag + 1):
                col = f"Lag_{lag}"
                if col in current.index:
                    if last_close is not None and len(last_close) >= lag:
                        current[col] = last_close[-lag]
                    elif len(last_rows) >= lag:
                        current[col] = last_rows.iloc[-lag]["Close"]

            pred = m.predict(current[feature_cols].values.reshape(1, -1))[0]
            preds.append(pred)

            # update last_close and last_rows for the next prediction step
            if last_close is not None:
                last_close = np.append(last_close[1:], pred) if len(last_close) > 1 else np.array([pred])

            new_row = current.copy()
            if "Close" in new_row.index:
                new_row["Close"] = pred
            last_rows = pd.concat([last_rows, new_row.to_frame().T], ignore_index=True)
            if len(last_rows) > max_lag:
                last_rows = last_rows.iloc[-max_lag:]

        return {"dates": future_dates.strftime("%Y-%m-%d").tolist(), "predictions": preds}

    elif model_lower == "randomforest":
        m = joblib.load(os.path.join(ticker_folder, f"RandomForest_{ticker_symbol}.pkl"))
        data = joblib.load(os.path.join(ticker_folder, f"RandomForest_{ticker_symbol}_data.pkl"))
        feature_cols = data["feature_cols"]

        last_rows = data.get("last_rows")
        if last_rows is None or len(last_rows) == 0:
            future_features = np.tile(data["last_features"], (steps, 1))
            fc = m.predict(future_features)
            return {"dates": future_dates.strftime("%Y-%m-%d").tolist(), "predictions": fc.tolist()}

        max_lag = max((int(c.split("_")[1]) for c in feature_cols if c.startswith("Lag_")), default=0)
        last_rows = last_rows.copy()
        last_close = data.get("last_close")
        preds = []

        for _ in range(steps):
            current = last_rows.iloc[-1].copy()
            for lag in range(1, max_lag + 1):
                col = f"Lag_{lag}"
                if col in current.index:
                    if last_close is not None and len(last_close) >= lag:
                        current[col] = last_close[-lag]
                    elif len(last_rows) >= lag:
                        current[col] = last_rows.iloc[-lag]["Close"]

            pred = m.predict(current[feature_cols].values.reshape(1, -1))[0]
            preds.append(pred)

            # update last_close and last_rows for the next prediction step
            if last_close is not None:
                last_close = np.append(last_close[1:], pred) if len(last_close) > 1 else np.array([pred])

            new_row = current.copy()
            if "Close" in new_row.index:
                new_row["Close"] = pred
            last_rows = pd.concat([last_rows, new_row.to_frame().T], ignore_index=True)
            if len(last_rows) > max_lag:
                last_rows = last_rows.iloc[-max_lag:]

        return {"dates": future_dates.strftime("%Y-%m-%d").tolist(), "predictions": preds}

    else:
        raise ValueError(f"Model '{model}' not supported.")
