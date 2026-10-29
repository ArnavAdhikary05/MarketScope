
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, engine
from app.ml.train import train_model
from app.providers.yahoo import fetch_stock_data
from app.services.features import create_features
from app.services.model_comparison import compare_models

Base.metadata.create_all(bind=engine)

logger = logging.getLogger(__name__)
app = FastAPI(title="MarketScope AI API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


def validate_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,30}", normalized):
        raise HTTPException(status_code=422, detail="Invalid stock symbol")
    return normalized


@app.get("/api/stocks/{symbol}/history")
def history(
    symbol: str,
    interval: str = "1d",
    period: str = "1y",
):
    symbol = validate_symbol(symbol)
    try:
        df = fetch_stock_data(symbol, period, interval)
        df["timestamp"] = df["timestamp"].astype(str)
        return {
            "symbol": symbol,
            "interval": interval,
            "source": "yfinance",
            "data_status": "provider-dependent; delayed data may be returned",
            "last_observed_at": df["timestamp"].iloc[-1],
            "data": df.to_dict(orient="records"),
        }
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to fetch market data: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Market history request failed for %s", symbol)
        raise HTTPException(
            status_code=502,
            detail="Unable to fetch market data from the configured provider",
        ) from exc


@app.post("/api/stocks/{symbol}/train")
def train(
    symbol: str,
    interval: str = "1d",
):
    symbol = validate_symbol(symbol)
    try:
        return train_model(symbol, interval)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Model training failed for %s", symbol)
        raise HTTPException(status_code=500, detail="Model training failed") from exc


@app.get("/api/stocks/{symbol}/compare")
def compare(
    symbol: str,
    interval: str = "1d",
    period: str = "5y",
    horizon: int = Query(default=1, ge=1, le=20),
    model: str = "auto",
):
    symbol = validate_symbol(symbol)
    try:
        market_data = fetch_stock_data(symbol, period=period, interval=interval)
        return compare_models(
            market_data,
            symbol=symbol,
            interval=interval,
            horizon=horizon,
            selected_model=model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Model comparison failed for %s", symbol)
        raise HTTPException(
            status_code=502,
            detail="Model comparison failed; check data availability and server logs",
        ) from exc


@app.get("/api/stocks/{symbol}/predict")
def predict(
    symbol: str,
    interval: str = "1d",
):
    symbol = validate_symbol(symbol)
    path = Path(
        "artifacts"
    ) / f"{symbol.replace('.', '_')}_{interval}.joblib"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model not trained. Call the train endpoint first.",
        )

    try:
        bundle = joblib.load(path)
        raw = fetch_stock_data(
            symbol,
            period="60d" if interval != "1d" else "1y",
            interval=interval,
        )
        features = create_features(raw).dropna(
            subset=bundle["features"]
        )

        if features.empty:
            raise ValueError("Insufficient data for prediction")

        last = features.iloc[[-1]]
        prediction = float(
            bundle["model"].predict(last[bundle["features"]])[0]
        )

        return {
            "symbol": symbol,
            "interval": interval,
            "last_close": float(last["close"].iloc[0]),
            "predicted_next_close": prediction,
            "last_bar_timestamp": str(last["timestamp"].iloc[0]),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": type(bundle["model"]).__name__,
            "interpretation": "Experimental forecast, not a guarantee",
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed for %s", symbol)
        raise HTTPException(status_code=502, detail="Prediction failed") from exc
