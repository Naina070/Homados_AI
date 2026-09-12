# Public internet deploy (not LAN-only)

The dashboard is a static React app. It talks to FastAPI over **HTTPS** using
`VITE_API_URL`. There is no hardcoded `localhost` or home Wi-Fi IP.

## Fast path for a hackathon jury on any Wi-Fi

1. Push this repo to GitHub.
2. In [Render](https://render.com), choose **New → Blueprint** and select the
   repository. `render.yaml` uses the included Dockerfile, which builds React
   and serves the compiled UI and FastAPI from one HTTPS URL.
3. Set `CORS_ORIGINS` to your final frontend URL if you later deploy the UI
   separately. Keep `*` only for a short public jury demo.

After deployment, anyone on college Wi-Fi, home Wi-Fi, or mobile data opens
the Render URL. File upload works from any network because the site and API
share a public HTTPS origin; nothing is bound to a LAN address.

## Optional separate frontend

Deploy `frontend/` to Vercel and set `VITE_API_URL` to the Render HTTPS URL.
Then set `CORS_ORIGINS=https://<your-vercel-project>.vercel.app` in Render.
This is optional—the single-service Docker deployment is the simplest demo.

## Same-machine development

```powershell
# API
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --app-dir . --host 0.0.0.0 --port 8000
```

Wait — uvicorn app-dir should be backend folder. From repo root:

```powershell
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

```powershell
cd frontend
npm install
npm run dev
```

Vite proxies `/api` to port 8000 locally. For a **public** demo URL from your
laptop, use Cloudflare Tunnel (`cloudflared tunnel --url http://127.0.0.1:8000`)
after `npm run build` so FastAPI serves the UI, or tunnel Vite + API separately.

## CORS

`.env` → `CORS_ORIGINS=*` for the SIH demo, or lock to the Vercel origin.
