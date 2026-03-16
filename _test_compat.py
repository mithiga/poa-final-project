import warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np
import traceback
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from backend.apis.pandas_compat import patch_stringdtype_unpickle_compat

patch_stringdtype_unpickle_compat()

for pkl in [
    'backend/models/EURUSD/ARIMA_EURUSD.pkl',
    'backend/models/EURUSD/SARIMA_EURUSD.pkl',
    'backend/models/GCF/ARIMA_GCF.pkl',
    'backend/models/GCF/SARIMA_GCF.pkl',
]:
    import os
    if not os.path.exists(pkl):
        print(f"SKIP (not found): {pkl}")
        continue
    print(f"\n--- Testing {pkl}")
    m = joblib.load(pkl)
    print(f"  type: {type(m)}")
    try:
        fc = np.asarray(m.forecast(steps=5), dtype=np.float64)
        print(f"  forecast OK: {fc[:2]}")
    except Exception as e:
        print(f"  ERROR type : {type(e).__name__}")
        print(f"  ERROR msg  : {str(e)[:400]}")
        traceback.print_exc()

    # Also test the fallback logic currently in generate_forecast
    try:
        if hasattr(m.model, 'order'):
            endog_np = np.asarray(m.model.endog, dtype=np.float64).ravel()
            if hasattr(m.model, 'seasonal_order') and m.model.seasonal_order != (0, 0, 0, 0):
                seasonal_order = m.model.seasonal_order
                fresh_res = SARIMAX(endog_np, order=m.model.order, seasonal_order=seasonal_order
                                    ).smooth(np.asarray(m.params, dtype=np.float64))
            else:
                fresh_res = ARIMA(endog_np, order=m.model.order).smooth(
                    np.asarray(m.params, dtype=np.float64)
                )
            fc2 = np.asarray(fresh_res.forecast(steps=5), dtype=np.float64)
            print(f"  fallback OK: {fc2[:2]}")
    except Exception as e2:
        print(f"  FALLBACK ERROR type : {type(e2).__name__}")
        print(f"  FALLBACK ERROR msg  : {str(e2)[:400]}")
        traceback.print_exc()
