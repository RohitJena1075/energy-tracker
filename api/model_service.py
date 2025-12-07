# -*- coding: utf-8 -*-
import os
import json
import logging
import pandas as pd
import numpy as np
import joblib

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /app
MODELS_DIR = os.path.join(BASE_DIR, "models")

_CFG = None
_LC_MODEL = None
_GEN_MODEL = None
_FEATURE_COLS = None


def _load_models():
    """
    Lazy-load config and models.

    The StandardScaler is not loaded as a pickled object to avoid
    binary dependencies (libgomp) on Railway; instead, its mean and
    scale are stored in feature_config.json and applied manually.
    """
    global _CFG, _LC_MODEL, _GEN_MODEL, _FEATURE_COLS
    if _CFG is not None:
        return _CFG, _LC_MODEL, _GEN_MODEL, _FEATURE_COLS

    try:
        cfg_path = os.path.join(MODELS_DIR, "feature_config.json")
        logger.info("Loading feature config from %s", cfg_path)
        with open(cfg_path, "r") as f:
            _CFG = json.load(f)

        lc_path = os.path.join(
            MODELS_DIR, f"{_CFG['best_lc_model_type']}_lc_model.joblib"
        )
        gen_path = os.path.join(
            MODELS_DIR, f"{_CFG['best_gen_model_type']}_gen_model.joblib"
        )

        logger.info("Loading LC model from %s", lc_path)
        _LC_MODEL = joblib.load(lc_path)

        logger.info("Loading GEN model from %s", gen_path)
        _GEN_MODEL = joblib.load(gen_path)

        _FEATURE_COLS = _CFG["feature_cols"]
        logger.info(
            "Models loaded OK from %s (n_features=%d)",
            MODELS_DIR,
            len(_FEATURE_COLS),
        )
    except Exception as e:
        logger.exception("Failed to load models from %s", MODELS_DIR)
        raise RuntimeError(
            "Model stack could not be loaded on this environment "
            "(likely missing or incompatible artifacts)."
        ) from e

    return _CFG, _LC_MODEL, _GEN_MODEL, _FEATURE_COLS


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
    CFG, LC_MODEL, GEN_MODEL, FEATURE_COLS = _load_models()

    # manual StandardScaler stats (same as scaler.mean_ and scaler.scale_)
    means = np.array(CFG["scaler_mean"], dtype=float)
    scales = np.array(CFG["scaler_scale"], dtype=float)

    iso3 = iso3.upper()
    if hist_raw.empty:
        raise ValueError(f"No history for {iso3}")

    hist = _prepare_history_for_features(hist_raw)
    if hist.empty:
        raise ValueError("Not enough history to build features")

    last_row = hist.iloc[-1]
    last_year = int(last_row["year"])

    lc_level = float(last_row["low_carbon_share_pct"])
    gen_level = float(last_row["electricity_generation_twh"])
    log_gen_level = float(np.log(max(gen_level, 1e-6)))

    results = []

    for step in range(1, horizon + 1):
        row = hist.iloc[-1].reindex(FEATURE_COLS).fillna(0.0)
        # ensure float features
        X = row.astype(float).values.reshape(1, -1)

        X_scaled = (X - means) / scales
        delta_lc = float(LC_MODEL.predict(X_scaled)[0])
        delta_log_gen = float(GEN_MODEL.predict(X)[0])

        lc_level = lc_level + delta_lc
        lc_level = max(0.0, min(100.0, lc_level))
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

        new_row = hist.iloc[-1].copy()
        new_row["year"] = target_year
        new_row["low_carbon_share_pct"] = lc_level
        new_row["electricity_generation_twh"] = gen_level

        eps = 1e-9
        gen = float(max(gen_level, eps))
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
            prev_share = float(hist.iloc[-1].get(share_col, 0.0))
            new_row[f"{src}_twh"] = prev_share * gen

        # ensure numeric dtypes stay float
        for col in new_row.index:
            if col.endswith("_twh") or col.endswith("_share"):
                new_row[col] = float(new_row[col])

        hist = pd.concat([hist, new_row.to_frame().T], ignore_index=True)
        hist = _add_shares_and_lags(hist)


    return {
        "iso3": iso3,
        "base_year": last_year,
        "forecasts": results,
    }