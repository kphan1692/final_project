from __future__ import annotations

import logging
import math
import gzip
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.app.configs import (
    MODEL_PATH,
    PREDICTION_QUEUE,
    RABBITMQ_HOST,
    RABBITMQ_PASSWORD,
    RABBITMQ_PORT,
    RABBITMQ_USER,
    RABBITMQ_VHOST,
)
from api.app.schema import PredictRequest, PredictResponse, Prediction

logger = logging.getLogger(__name__)

FREQ_15MIN = "15min"
STEP_15MIN = timedelta(minutes=15)

FEATURE_COLUMNS: list[str] = [
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "nvm_active_power",
    "last_15min_kwh",
    "minute",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "lag_96",
    "lag_192",
    "lag_672",
    "roll_mean_4",
    "roll_min_4",
    "roll_max_4",
    "roll_mean_12",
    "roll_min_12",
    "roll_max_12",
    "roll_mean_96",
    "roll_min_96",
    "roll_max_96",
]

LAGS = (1, 2, 4, 8, 96, 192, 672)
ROLLING_WINDOWS = (4, 12, 96)


class SupportsPredict(Protocol):
    def predict(self, X: Any) -> Any: ...


class DummyModel:
    """
    Predicts a constant for every row (useful while wiring the API).
    """

    feature_names_in_ = np.array(FEATURE_COLUMNS, dtype=object)

    def __init__(self, constant: float = 0.0):
        self.constant = float(constant)

    def predict(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self.constant, dtype=float)


def _load_model(model_path: Path) -> tuple[SupportsPredict, bool, str | None]:
    try:
        with model_path.open("rb") as raw:
            magic = raw.read(2)
            raw.seek(0)
            if magic == b"\x1f\x8b":
                with gzip.open(raw, "rb") as f:
                    model = pickle.load(f)
            else:
                model = pickle.load(raw)

        if not hasattr(model, "predict"):
            raise TypeError("Loaded object does not implement predict()")

        return model, True, str(model_path)
    except FileNotFoundError:
        logger.warning("MODEL_PATH not found (%s); using DummyModel", model_path)
        return DummyModel(constant=0.0), False, None
    except Exception:
        logger.exception("Failed to load model from %s; using DummyModel", model_path)
        return DummyModel(constant=0.0), False, None


def _time_features(ts: datetime) -> dict[str, float | int]:
    hour = ts.hour
    dayofweek = ts.weekday()
    month = ts.month
    dayofyear = ts.timetuple().tm_yday

    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    dow_sin = math.sin(2 * math.pi * dayofweek / 7)
    dow_cos = math.cos(2 * math.pi * dayofweek / 7)

    return {
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "dayofyear": dayofyear,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "minute": ts.minute,
    }


def _lag_features(past_targets: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for lag in LAGS:
        out[f"lag_{lag}"] = float(past_targets[-lag])
    return out


def _rolling_features(past_targets: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for window in ROLLING_WINDOWS:
        values = past_targets[-window:]
        out[f"roll_mean_{window}"] = float(np.mean(values))
        out[f"roll_min_{window}"] = float(np.min(values))
        out[f"roll_max_{window}"] = float(np.max(values))
    return out


def _forecast_timestamps(req: PredictRequest) -> list[datetime]:
    last_ts = req.history[-1].ts
    if req.future_covariates is not None:
        return [p.ts for p in req.future_covariates]
    return [last_ts + STEP_15MIN * (i + 1) for i in range(req.horizon_steps)]


def _forecast_covariates(req: PredictRequest, timestamps: list[datetime]) -> list[dict[str, float]]:
    if req.future_covariates is None:
        last = req.history[-1]
        return [
            {
                "nvm_active_power": float(last.nvm_active_power),
                "last_15min_kwh": float(last.last_15min_kwh),
            }
            for _ in timestamps
        ]

    by_ts = {p.ts: p for p in req.future_covariates}
    out: list[dict[str, float]] = []
    for ts in timestamps:
        cov = by_ts.get(ts)
        if cov is None:
            raise ValueError(f"Missing future_covariates entry for ts={ts.isoformat()}")
        out.append(
            {
                "nvm_active_power": float(cov.nvm_active_power),
                "last_15min_kwh": float(cov.last_15min_kwh),
            }
        )
    return out


def _feature_row(
    *,
    ts: datetime,
    covariates: dict[str, float],
    past_targets: list[float],
) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    row.update(_time_features(ts))
    row.update(covariates)
    row.update(_lag_features(past_targets))
    row.update(_rolling_features(past_targets))
    return row


def _feature_columns_for_model(model: SupportsPredict) -> list[str]:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return FEATURE_COLUMNS

    cols = list(map(str, list(feature_names)))
    if cols != FEATURE_COLUMNS:
        raise RuntimeError(
            "Loaded model feature columns do not match the locked contract.\n"
            f"Expected: {FEATURE_COLUMNS}\n"
            f"Got     : {cols}"
        )
    return cols


def predict_with_model(
    *,
    req: PredictRequest,
    model: SupportsPredict,
    feature_columns: list[str],
    model_path: str | None,
) -> PredictResponse:
    timestamps = _forecast_timestamps(req)
    covariates = _forecast_covariates(req, timestamps)

    # Only the last 672 targets are required for lags/rolling stats.
    past_targets = [float(p.target_ac_power) for p in req.history[-672:]]

    yhat_values: list[float] = []
    for ts, cov in zip(timestamps, covariates):
        row = _feature_row(ts=ts, covariates=cov, past_targets=past_targets)
        X = pd.DataFrame([row], columns=feature_columns)
        yhat = float(np.asarray(model.predict(X)).reshape(-1)[0])
        yhat_values.append(yhat)
        past_targets.append(yhat)

    preds = [Prediction(ts=ts, yhat=y) for ts, y in zip(timestamps, yhat_values)]
    return PredictResponse(
        freq=req.freq,
        horizon_steps=req.horizon_steps,
        predictions=preds,
        model_path=model_path,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Albuquerque Energy Forecast API")

    @app.on_event("startup")
    def _startup() -> None:
        model, loaded, loaded_path = _load_model(Path(MODEL_PATH))
        app.state.model = model
        app.state.model_loaded = loaded
        app.state.model_path = loaded_path
        app.state.feature_columns = _feature_columns_for_model(model)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_loaded": bool(getattr(app.state, "model_loaded", False))}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        model: SupportsPredict | None = getattr(app.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            return predict_with_model(
                req=req,
                model=model,
                feature_columns=app.state.feature_columns,
                model_path=getattr(app.state, "model_path", None),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed") from e

    return app


app = create_app()


def run_worker() -> None:
    """
    RabbitMQ RPC worker:
    - consumes PredictRequest JSON from PREDICTION_QUEUE
    - publishes PredictResponse JSON to props.reply_to with same props.correlation_id
    """
    try:
        import pika  # type: ignore[import-not-found]
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError("Worker requires `pika` (pip install pika).") from e

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    model, loaded, loaded_path = _load_model(Path(MODEL_PATH))
    feature_columns = _feature_columns_for_model(model)

    logger.info(
        "worker start host=%s queue=%s model_loaded=%s model_path=%s",
        RABBITMQ_HOST,
        PREDICTION_QUEUE,
        loaded,
        loaded_path,
    )

    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            virtual_host=RABBITMQ_VHOST,
            credentials=credentials,
            heartbeat=0,
        )
    )
    channel = connection.channel()
    channel.queue_declare(queue=PREDICTION_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)

    def _publish(ch, *, reply_to: str, correlation_id: str | None, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        ch.basic_publish(
            exchange="",
            routing_key=reply_to,
            properties=pika.BasicProperties(
                correlation_id=correlation_id,
                content_type="application/json",
            ),
            body=body,
        )

    def on_request(ch, method, props, body: bytes) -> None:  # type: ignore[no-untyped-def]
        corr_id = getattr(props, "correlation_id", None)
        reply_to = getattr(props, "reply_to", None)

        logger.info("consume corr_id=%s reply_to=%s", corr_id, reply_to)

        if not reply_to:
            logger.error("missing reply_to; dropping request corr_id=%s", corr_id)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info("ack corr_id=%s", corr_id)
            return

        try:
            payload = json.loads(body)
            req = PredictRequest.model_validate(payload)

            logger.info("predict start corr_id=%s", corr_id)
            resp = predict_with_model(
                req=req,
                model=model,
                feature_columns=feature_columns,
                model_path=loaded_path,
            )
            logger.info("predict ok corr_id=%s", corr_id)

            _publish(
                ch,
                reply_to=reply_to,
                correlation_id=corr_id,
                payload=resp.model_dump(mode="json"),
            )
            logger.info("publish corr_id=%s to=%s", corr_id, reply_to)

            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info("ack corr_id=%s", corr_id)
        except Exception as e:
            logger.exception("handle failed corr_id=%s", corr_id)
            error_payload: dict[str, Any] = {
                "error": {"type": type(e).__name__, "message": str(e)},
            }

            # If this is a Pydantic ValidationError, include structured error detail.
            errors = getattr(e, "errors", None)
            if callable(errors):
                try:
                    error_payload["error"]["errors"] = errors()
                except Exception:
                    pass

            try:
                _publish(
                    ch,
                    reply_to=reply_to,
                    correlation_id=corr_id,
                    payload=error_payload,
                )
                logger.info("publish error corr_id=%s to=%s", corr_id, reply_to)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                logger.info("ack corr_id=%s", corr_id)
            except Exception:
                logger.exception("failed to publish error corr_id=%s; requeueing", corr_id)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    channel.basic_consume(queue=PREDICTION_QUEUE, on_message_callback=on_request, auto_ack=False)

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("worker shutdown requested")
        channel.stop_consuming()
    finally:
        connection.close()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["api", "worker"], nargs="?", default="api")
    args = parser.parse_args()

    if args.mode == "worker":
        run_worker()
    else:
        import uvicorn

        from api.app.configs import HOST, LISTEN_PORT

        uvicorn.run(app, host=HOST, port=LISTEN_PORT, log_level="info")
