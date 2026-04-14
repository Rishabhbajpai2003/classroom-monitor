# Classroom Monitor Monorepo

This repo now contains both pieces of the product:

- `frontend/`: Flutter web app
- `app/`, `detectors/`, `configs/`, `scripts/`: FastAPI backend and ML pipeline

## Local Development

Backend:

```powershell
.\start_backend.ps1
```

Frontend:

```powershell
.\start_frontend.ps1
```

Default backend URL:

- `http://localhost:8000`

The Flutter app stores the backend URL in browser local storage, so you can also point it at a remote deployment from the in-app URL dialog.

## Backend Contract

The frontend talks directly to the FastAPI service in `app/api/main.py`.

Main endpoints:

- `GET /health`
- `POST /api/analysis/jobs`
- `GET /api/analysis/jobs/{job_id}`
- `GET /api/dates`
- `GET /api/attendance/{date}`
- `GET /api/files?path=...`
- `POST /api/query-rag`

## Output History

Completed analysis runs write `run_summary.json` inside their output directories under `outputs/`. The attendance/history screens read those summaries through the backend instead of accessing backend filesystem paths directly.
