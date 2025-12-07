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

.
├── api/ # FastAPI service and model serving code
│ ├── main.py # HTTP endpoints, DB access, CORS
│ ├── model_service.py# Feature engineering + forecasting logic
│ └── models/ # Trained models + feature_config + metrics
├── data/ # ML panel / preprocessing outputs (local, optional)
├── frontend/ # React client (EnForecast UI)
│ ├── src/ # Pages, components, charts
│ └── public/ # Static assets, index.html, manifest, icons
└── train_datasets.py # Offline training script for ML models

text

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
You can adjust the demo URLs to your exact Netlify and Railway addresses, and tweak the roadmap items as you like.

give me in one readme.md file all for copy paste
text
# EnForecast – Global Energy Forecasts

EnForecast is an interactive web app for exploring how countries generate electricity today and how their power mixes might evolve over the next decade. It combines cleaned historical data with machine‑learning models to forecast both total generation and low‑carbon share for more than 100 countries.[web:229]

## Live demo

- Frontend: https://enforecast.netlify.app  
- API: https://energy-tracker-production-f812.up.railway.app

## Features

- Searchable country catalog with autocomplete and quick stats.  
- Single‑country view with:
  - Historical electricity generation and low‑carbon share.
  - 10‑year forward forecasts visualized as charts and tables.
- Comparison view to line up multiple countries side by side.  
- Embedded documentation explaining the dataset and modelling approach.

## Tech stack

- **Frontend:** React (Create React App), TypeScript, Recharts, Netlify for hosting.[web:219]  
- **Backend API:** FastAPI, async SQLAlchemy, asyncpg, Railway for deployment.[web:195]  
- **Database:** PostgreSQL storing country metadata and yearly electricity data.  
- **ML models:** scikit‑learn + XGBoost for forecasting:
  - Predicts annual change in low‑carbon share.
  - Predicts annual change in log electricity generation.
- **Infrastructure:** Netlify (static frontend) + Railway (containerized API + Postgres).[web:220]

## Project structure

.
├── api/ # FastAPI service and model serving code
│ ├── main.py # HTTP endpoints, DB access, CORS
│ ├── model_service.py# Feature engineering + forecasting logic
│ └── models/ # Trained models + feature_config + metrics
├── data/ # ML panel / preprocessing outputs (local, optional)
├── frontend/ # React client (EnForecast UI)
│ ├── src/ # Pages, components, charts
│ └── public/ # Static assets, index.html, manifest, icons
└── train_datasets.py # Offline training script for ML models

text

## Getting started (local)

### Backend

cd api
python -m venv .venv

Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

text

The API will be available at `http://localhost:8000`.

### Frontend

cd frontend
npm install
npm start

text

The React app will be available at `http://localhost:3000` and will call the backend at `http://localhost:8000`.

## Deployment

### Frontend (Netlify)

- Build with:

cd frontend
npm run build

text

- Netlify settings:
- **Base directory:** `frontend`
- **Build command:** `npm run build`
- **Publish directory:** `build`[web:219]

### Backend (Railway)

- Service command: `uvicorn main:app --host 0.0.0.0 --port 8000`.  
- Environment:
- `DATABASE_URL` pointing to the managed Postgres instance.
- Any additional config (e.g., CORS origins) via env vars.

## Modelling overview

- Works on a panel dataset of annual electricity statistics per country.  
- Targets:
- `delta_lc`: yearly change in low‑carbon share (percentage points).
- `delta_log_gen`: yearly change in log total generation.  
- Trained models:
- XGBoost (or RandomForest fallback) for each target.  
- Inference:
- Builds feature vector from recent history, standardizes features using stored scaler mean and scale, then rolls the system forward year by year to generate a 10‑year trajectory.

## Status and roadmap

- [x] Global country coverage with historical data.  
- [x] 10‑year forecasts for total generation and low‑carbon share.  
- [x] Interactive charts and comparison view.  
- [ ] Add uncertainty bands around forecasts.  
- [ ] Expose downloadable CSV / bulk API endpoints.  
- [ ] Improve calibration for small and highly volatile systems.

---

If you use or extend EnForecast, feel free to open issues or pull requests with feedback and improvements.