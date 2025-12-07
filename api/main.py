# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Query
import model_service
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import os
import json
from dotenv import load_dotenv

# Load env vars (DATABASE_URL from Railway)
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI(title="Energy Forecast API")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Async DB engine for Railway Postgres
# Railway usually gives postgres://, asyncpg/SQLAlchemy expect postgresql+asyncpg://
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://")
engine = create_async_engine(ASYNC_DATABASE_URL, future=True)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# CORS (open while testing; later restrict to specific frontend URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173", "https://your-frontend.up.railway.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/countries")
async def list_countries():
    """
    Return list of countries from the countries table in Postgres.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT iso3, name FROM countries "
                "WHERE iso3 IS NOT NULL ORDER BY name"
            )
        )
        rows = result.all()
    return [{"code": r[0].strip(), "name": r[1]} for r in rows]


@app.get("/model-metrics")
def model_metrics():
    """
    Return global validation and test metrics for the forecasting models.
    Expects models/metrics.json at project root: energy-tracker/models/metrics.json
    """
    metrics_path = os.path.join(PROJECT_ROOT, "models", "metrics.json")
    if not os.path.exists(metrics_path):
        return {"error": "metrics file not found", "path": metrics_path}
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return metrics


@app.get("/forecast/{iso3}")
def forecast(iso3: str, horizon: int = Query(5, ge=1, le=10)):
    """
    Return forecast for a given country iso3 for the next `horizon` years.
    """
    try:
        return model_service.predict_horizon(iso3, horizon=horizon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
