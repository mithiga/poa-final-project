# ForexAI Pro

This repository contains a Streamlit frontend and a FastAPI backend for forex forecasting.

## Streamlit deployment

The project is prepared for Streamlit Community Cloud as a single deployed app:

- Entry point: `streamlit_app.py`
- Python runtime: `runtime.txt`
- Dependencies: root `requirements.txt`
- Backend mode on Streamlit Cloud: embedded inside the Streamlit process

If you want the frontend to call an external backend instead, set the `BACKEND_API_URL` environment variable in Streamlit Cloud.

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

## Notes

- Pretrained model artifacts under `backend/models/` are included so forecasts can work after deployment.
- Training large models on Streamlit Community Cloud may be slow due to platform resource limits.