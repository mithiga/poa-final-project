# ForexAI Pro

This repository contains a Streamlit frontend and a FastAPI backend for forex forecasting.

## Streamlit deployment

The project is prepared for Streamlit Community Cloud as a single deployed app:

- Entry point: `streamlit_app.py`
- Python runtime: `runtime.txt`
- Dependencies: root `requirements.txt`
- Backend mode on Streamlit Cloud: embedded inside the Streamlit process

If you want the frontend to call an external backend instead, set BACKEND_API_URL in Streamlit Cloud secrets or environment variables.

### Expose FastAPI in deployment (recommended)

To access public FastAPI endpoints in production, deploy the backend as a separate web service and point Streamlit to it.

1. Deploy backend service

- Working directory: backend
- Start command:

```powershell
uvicorn apis.main:app --host 0.0.0.0 --port $PORT
```

2. Confirm backend is reachable

- Open: https://your-backend-domain/docs
- Test: https://your-backend-domain/status

3. Configure Streamlit frontend to use remote backend

Option A: Streamlit secrets (preferred)

```toml
BACKEND_API_URL = "https://your-backend-domain"
```

Option B: nested secret format

```toml
[backend]
api_url = "https://your-backend-domain"
```

Option C: environment variable

- Key: BACKEND_API_URL
- Value: https://your-backend-domain

4. Redeploy/restart Streamlit app

After restart, sidebar should show Backend mode: remote.

## Local run

### Single-process mode

```powershell
cd frontend
python -m streamlit run app.py
```

### Separate frontend and backend mode

```powershell
cd backend
python main.py
```

```powershell
cd frontend
$env:BACKEND_API_URL="http://localhost:8000"
python -m streamlit run app.py
```

## Push to GitHub

```powershell
git init
git add .
git commit -m "Prepare Streamlit deployment"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Streamlit Community Cloud settings

- Repository: your GitHub repository
- Branch: `main`
- Main file path: `streamlit_app.py`

If using a separate backend service, set BACKEND_API_URL in Streamlit app Settings -> Secrets (or Environment Variables).

## Notes

- Pretrained model artifacts under `backend/models/` are included so forecasts can work after deployment.
- Training large models on Streamlit Community Cloud may be slow due to platform resource limits.