# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import os
import json
import pandas as pd
from dotenv import load_dotenv

from model_service import predict_horizon_from_df

# Load env vars
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI(title="Energy Forecast API")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Async DB engine for Railway Postgres
# Railway usually gives postgres://, asyncpg expects postgresql+asyncpg://
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://")
engine = create_async_engine(ASYNC_DATABASE_URL, future=True)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# CORS (open for now; restrict later to your Railway frontend + localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["http://localhost:5173", "https://your-frontend.up.railway.app"]
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


async def fetch_history_df(iso3: str) -> pd.DataFrame:
    """
    Fetch full historical time series for a country from Postgres,
    matching the columns used in the ML pipeline.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                """
                SELECT
                    c.country_id,
                    c.iso3,
                    c.name,
                    c.region,
                    c.subregion,
                    c.income_group,
                    c.population_millions,
                    c.gdp_billions_usd,
                    e.year,
                    e.electricity_generation_twh,
                    e.coal_twh,
                    e.oil_twh,
                    e.gas_twh,
                    e.nuclear_twh,
                    e.hydro_twh,
                    e.solar_twh,
                    e.wind_twh,
                    e.other_renewables_twh,
                    e.low_carbon_share_pct,
                    e.fossil_share_pct
                FROM energy_yearly e
                JOIN countries c ON c.country_id = e.country_id
                WHERE c.iso3 = :iso3
                ORDER BY e.year;
                """
            ),
            {"iso3": iso3.upper()},
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    cols = [
        "country_id",
        "iso3",
        "name",
        "region",
        "subregion",
        "income_group",
        "population_millions",
        "gdp_billions_usd",
        "year",
        "electricity_generation_twh",
        "coal_twh",
        "oil_twh",
        "gas_twh",
        "nuclear_twh",
        "hydro_twh",
        "solar_twh",
        "wind_twh",
        "other_renewables_twh",
        "low_carbon_share_pct",
        "fossil_share_pct",
    ]
    return pd.DataFrame(rows, columns=cols)


@app.get("/model-metrics")
def model_metrics():
    """
    Return global validation and test metrics for the forecasting models.
    Expects models/metrics.json at project root (now also inside api/models if you moved it).
    """
    metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
    if not os.path.exists(metrics_path):
        return {"error": "metrics file not found", "path": metrics_path}
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return metrics


@app.get("/forecast/{iso3}")
async def forecast(iso3: str, horizon: int = Query(5, ge=1, le=10)):
    """
    Return forecast for a given country iso3 for the next `horizon` years.
    """
    hist_df = await fetch_history_df(iso3)
    try:
        return predict_horizon_from_df(iso3, hist_df, horizon=horizon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # model stack cannot be loaded on this Railway image
        raise HTTPException(status_code=500, detail=str(e))
