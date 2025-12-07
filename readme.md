# EnForecast – Global Energy Forecasts

EnForecast is an interactive web app for exploring how countries generate electricity today and how their power mixes might evolve over the next decade. It combines cleaned historical data with machine‑learning models to forecast both total generation and low‑carbon share for more than 100 countries.

## Live demo

- Frontend: https://enforecast.netlify.app  
- API: https://energy-tracker-production-f812.up.railway.app

## Features

- Searchable country catalog with quick stats and autocomplete.
- Single‑country view with:
  - Historical electricity generation and low‑carbon share.
  - 10‑year forward forecasts visualized as charts and tables.
- Comparison view to line up multiple countries side by side.
- Model overview and dataset documentation embedded in the UI.

## Tech stack

- **Frontend:** React (Create React App), TypeScript, Recharts, Netlify for hosting.
- **Backend API:** FastAPI, async SQLAlchemy, asyncpg, Railway for deployment.
- **Database:** PostgreSQL with country and yearly electricity data.
- **ML models:** scikit‑learn + XGBoost for forecasting:
  - Predicts annual change in low‑carbon share.
  - Predicts annual change in log electricity generation.
- **Infrastructure:** Netlify (static frontend) + Railway (containerized API + Postgres).

## Project structure


- ├── api/ # FastAPI service and model serving code
- │ ├── main.py # HTTP endpoints, DB access, CORS
- │ ├── model_service.py# Feature engineering + forecasting logic
- │ └── models/ # Trained models + feature_config + metrics
- ├── data/ # ML panel / preprocessing outputs (local, optional)
- ├── frontend/ # React client (EnForecast UI)
- │ ├── src/ # Pages, components, charts
- │ └── public/ # Static assets, index.html, manifest, icons
- └── train_datasets.py # Offline training script for ML models


## Getting started (local)

1. **Backend**

cd api
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

text

2. **Frontend**

cd frontend
npm install
npm start

text

The app will be available at http://localhost:3000, talking to the API at http://localhost:8000.

## Deployment

- **Frontend:**  
- Build with `npm run build` in `frontend/`.  
- Deployed via Netlify with base directory `frontend` and publish directory `build`.

- **Backend:**  
- Deployed as a Railway service with `uvicorn main:app` as the entrypoint.  
- Uses a managed PostgreSQL instance, configured via `DATABASE_URL`.

## Status and roadmap

- [x] Global country coverage with historical data.
- [x] 10‑year forecasts for total generation and low‑carbon share.
- [x] Interactive charts and comparison view.
- [ ] Add uncertainty bands around forecasts.
- [ ] Expose downloadable CSV / API endpoints for bulk queries.
- [ ] Improve model calibration for small and volatile systems.

---

If you use or extend EnForecast, feel free to open issues or pull requests with feedback and improvements.
