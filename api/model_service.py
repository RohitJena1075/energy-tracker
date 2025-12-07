# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import numpy as np
import joblib

# this file is api/model_service.py, models folder is at project root: /app/models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # /app
PROJECT_ROOT = BASE_DIR                                   # since api is the root in container
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# load config and models
with open(os.path.join(MODELS_DIR, "feature_config.json"), "r") as f:
    CFG = json.load(f)

SCALER = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
LC_MODEL = joblib.load(
    os.path.join(MODELS_DIR, f"{CFG['best_lc_model_type']}_lc_model.joblib")
)
GEN_MODEL = joblib.load(
    os.path.join(MODELS_DIR, f"{CFG['best_gen_model_type']}_gen_model.joblib")
)

FEATURE_COLS = CFG["feature_cols"]

def _add_shares_and_lags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    eps = 1e-9
    gen = df["electricity_generation_twh"].clip(lower=eps)
    for src in [
        "coal",
        "oil",
        "gas",
        "nuclear",
        "hydro",
        "solar",
        "wind",
        "other_renewables",
    ]:
        df[f"{src}_share"] = df[f"{src}_twh"] / gen

    lag_cols = [
        "low_carbon_share_pct",
        "electricity_generation_twh",
        "solar_share",
        "wind_share",
        "fossil_share_pct",
    ]
    df = df.sort_values("year")
    for col in lag_cols:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def _prepare_history_for_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_shares_and_lags(df)
    df = df[df["year"] >= 2000]
    df = df[df["low_carbon_share_pct_lag3"].notnull()]
    return df.sort_values("year").copy()


def predict_horizon_from_df(
    iso3: str, hist_raw: pd.DataFrame, horizon: int = 5
) -> dict:
    """
    Predict low_carbon_share_pct and electricity_generation_twh
    for horizon future years (1–10) after the last actual year,
    given a history dataframe from the database.
    """
    iso3 = iso3.upper()
    if hist_raw.empty:
        raise ValueError(f"No history for {iso3}")

    hist = _prepare_history_for_features(hist_raw)
    if hist.empty:
        raise ValueError("Not enough history to build features")

    last_row = hist.iloc[-1]
    last_year = int(last_row["year"])

    # starting levels
    lc_level = float(last_row["low_carbon_share_pct"])
    gen_level = float(last_row["electricity_generation_twh"])
    log_gen_level = float(np.log(max(gen_level, 1e-6)))

    results = []

    for step in range(1, horizon + 1):
        row = hist.iloc[-1].reindex(FEATURE_COLS).fillna(0.0)
        X = row.values.reshape(1, -1)
        X_scaled = SCALER.transform(X)

        delta_lc = float(LC_MODEL.predict(X_scaled)[0])
        delta_log_gen = float(GEN_MODEL.predict(X)[0])

        # update levels
        lc_level = lc_level + delta_lc
        lc_level = max(0.0, min(100.0, lc_level))  # clamp to [0, 100]

        log_gen_level = log_gen_level + delta_log_gen
        gen_level = float(np.exp(log_gen_level))

        target_year = last_year + step
        results.append(
            {
                "year": target_year,
                "low_carbon_share_pct": lc_level,
                "electricity_generation_twh": gen_level,
            }
        )

        # append predicted year to history for next step
        new_row = hist.iloc[-1].copy()
        new_row["year"] = target_year
        new_row["low_carbon_share_pct"] = lc_level
        new_row["electricity_generation_twh"] = gen_level

        # keep previous shares to distribute TWh
        eps = 1e-9
        gen = max(gen_level, eps)
        for src in [
            "coal",
            "oil",
            "gas",
            "nuclear",
            "hydro",
            "solar",
            "wind",
            "other_renewables",
        ]:
            share_col = f"{src}_share"
            prev_share = hist.iloc[-1].get(share_col, 0.0)
            new_row[f"{src}_twh"] = prev_share * gen

        hist = pd.concat([hist, new_row.to_frame().T], ignore_index=True)
        hist = _add_shares_and_lags(hist)

    return {
        "iso3": iso3,
        "base_year": last_year,
        "forecasts": results,
    }