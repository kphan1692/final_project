from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone
from typing import Any

import streamlit as st

from api_client import check_health, predict_http
from mq_client import RabbitMQError, RabbitMQTimeoutError, rpc_predict

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

STEP_15MIN = timedelta(minutes=15)
MIN_HISTORY_POINTS = 672


def _utc_now_15min() -> datetime:
    now = datetime.now(timezone.utc)
    minute = (now.minute // 15) * 15
    return now.replace(minute=minute, second=0, microsecond=0)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _clamp_non_negative(x: float) -> float:
    return float(x) if x >= 0 else 0.0


def build_payload(
    *,
    last_ts: datetime,
    history_points: int,
    horizon_steps: int,
    target_base: float,
    target_amp: float,
    target_noise: float,
    history_pattern: str,
    nvm_active_power: float,
    last_15min_kwh: float,
    provide_future_covariates: bool,
    future_nvm_active_power: float,
    future_last_15min_kwh: float,
) -> dict[str, Any]:
    if history_points < MIN_HISTORY_POINTS:
        raise ValueError(f"history_points must be >= {MIN_HISTORY_POINTS}")

    start_ts = last_ts - STEP_15MIN * (history_points - 1)
    timestamps = [start_ts + STEP_15MIN * i for i in range(history_points)]

    # Simple synthetic target generator (enough to drive lags/rolling features).
    # Keep values non-negative for nicer demos.
    import math
    import random

    target_values: list[float] = []
    for ts in timestamps:
        base = float(target_base)
        amp = float(target_amp)
        noise = float(target_noise)

        if history_pattern == "Daily sinusoid":
            # One cycle per day.
            minutes = ts.hour * 60 + ts.minute
            phase = 2 * math.pi * minutes / (24 * 60)
            value = base + amp * math.sin(phase)
        elif history_pattern == "Ramp":
            # Slow upward trend across the history window.
            idx = int((ts - start_ts) / STEP_15MIN)
            value = base + amp * (idx / max(history_points - 1, 1))
        elif history_pattern == "Mostly zeros":
            # Many zeros with occasional spikes.
            if random.random() < 0.9:
                value = 0.0
            else:
                value = base + amp
        else:  # "Constant"
            value = base

        if noise > 0:
            value += random.gauss(0.0, noise)

        target_values.append(_clamp_non_negative(value))

    history = [
        {
            "ts": _iso_z(ts),
            "target_ac_power": float(y),
            "nvm_active_power": float(nvm_active_power),
            "last_15min_kwh": float(last_15min_kwh),
        }
        for ts, y in zip(timestamps, target_values)
    ]

    payload: dict[str, Any] = {
        "freq": "15min",
        "horizon_steps": int(horizon_steps),
        "history": history,
    }

    if provide_future_covariates:
        future = []
        for i in range(horizon_steps):
            ts = last_ts + STEP_15MIN * (i + 1)
            future.append(
                {
                    "ts": _iso_z(ts),
                    "nvm_active_power": float(future_nvm_active_power),
                    "last_15min_kwh": float(future_last_15min_kwh),
                }
            )
        payload["future_covariates"] = future

    return payload


def _render_response(resp: dict[str, Any]) -> None:
    if "error" in resp:
        err = resp.get("error") or {}
        msg = err.get("message") or "Unknown error"
        st.error(msg)
        with st.expander("Raw error JSON", expanded=True):
            st.json(resp)
        return

    required = {"freq", "horizon_steps", "predictions"}
    missing = required - set(resp.keys())
    if missing:
        st.error(f"Response missing fields: {sorted(missing)}")
        with st.expander("Raw response JSON", expanded=True):
            st.json(resp)
        return

    predictions = resp.get("predictions") or []
    if not isinstance(predictions, list) or not predictions:
        st.error("No predictions returned.")
        with st.expander("Raw response JSON", expanded=True):
            st.json(resp)
        return

    st.subheader("Predictions")
    if pd is not None:
        df = pd.DataFrame(predictions)
        st.dataframe(df, use_container_width=True)
        if "yhat" in df.columns:
            try:
                chart_df = df.copy()
                chart_df["ts"] = pd.to_datetime(chart_df["ts"], utc=True, errors="coerce")
                chart_df = chart_df.dropna(subset=["ts"]).set_index("ts")
                st.line_chart(chart_df["yhat"])
            except Exception:
                pass
    else:
        st.table(predictions)

    with st.expander("Raw response JSON"):
        st.json(resp)


def main() -> None:
    st.set_page_config(page_title="Energy Forecast UI", layout="wide")
    st.title("Energy Forecast UI")

    api_url = os.environ.get("API_URL", "http://api:8000")
    st.caption(f"API_URL: {api_url}")

    flow = st.sidebar.radio(
        "Flow",
        options=["HTTP", "RPC"],
        help="HTTP calls FastAPI directly; RPC sends the same payload via RabbitMQ to the worker.",
    )

    with st.sidebar.expander("Connection Settings", expanded=True):
        st.write("HTTP")
        st.code(f"API_URL={api_url}", language="text")
        st.write("RabbitMQ")
        for key in [
            "RABBITMQ_HOST",
            "RABBITMQ_PORT",
            "RABBITMQ_USER",
            "RABBITMQ_PASSWORD",
            "RABBITMQ_VHOST",
            "PREDICTION_QUEUE",
        ]:
            if key in os.environ:
                st.code(f"{key}={os.environ[key]}", language="text")

    st.subheader("Request Inputs")

    default_last = _utc_now_15min()
    col1, col2, col3 = st.columns(3)
    with col1:
        horizon_steps = st.slider("horizon_steps", min_value=1, max_value=96, value=8, step=1)
    with col2:
        history_points = st.slider(
            "history_points",
            min_value=MIN_HISTORY_POINTS,
            max_value=2000,
            value=MIN_HISTORY_POINTS,
            step=24,
            help="Must be >= 672 (max lag).",
        )
    with col3:
        timeout_s = st.number_input(
            "timeout_seconds",
            min_value=1.0,
            max_value=120.0,
            value=15.0,
            step=1.0,
        )

    date_col, time_col = st.columns(2)
    with date_col:
        last_date = st.date_input("last_timestamp (date, UTC)", value=default_last.date())
    with time_col:
        last_time = st.time_input("last_timestamp (time, UTC)", value=time(default_last.hour, default_last.minute))

    last_ts = datetime.combine(last_date, last_time, tzinfo=timezone.utc)

    pattern = st.selectbox(
        "history_pattern",
        options=["Daily sinusoid", "Constant", "Ramp", "Mostly zeros"],
        index=0,
    )

    val1, val2, val3 = st.columns(3)
    with val1:
        target_base = st.number_input("target_base", value=50.0, step=1.0)
    with val2:
        target_amp = st.number_input("target_amplitude", value=30.0, step=1.0)
    with val3:
        target_noise = st.number_input("target_noise_std", value=2.0, step=0.5)

    cov1, cov2 = st.columns(2)
    with cov1:
        nvm_active_power = st.number_input("nvm_active_power (history)", value=10.0, step=0.5)
    with cov2:
        last_15min_kwh = st.number_input("last_15min_kwh (history)", value=1.0, step=0.1)

    provide_future_covariates = st.checkbox(
        "Provide future_covariates",
        value=True,
        help="If unchecked, API/worker will forward-fill from last history point.",
    )

    future_nvm_active_power = nvm_active_power
    future_last_15min_kwh = last_15min_kwh
    if provide_future_covariates:
        f1, f2 = st.columns(2)
        with f1:
            future_nvm_active_power = st.number_input(
                "nvm_active_power (future)",
                value=float(nvm_active_power),
                step=0.5,
            )
        with f2:
            future_last_15min_kwh = st.number_input(
                "last_15min_kwh (future)",
                value=float(last_15min_kwh),
                step=0.1,
            )

    payload_error: str | None = None
    payload: dict[str, Any] | None = None
    try:
        payload = build_payload(
            last_ts=last_ts,
            history_points=int(history_points),
            horizon_steps=int(horizon_steps),
            target_base=float(target_base),
            target_amp=float(target_amp),
            target_noise=float(target_noise),
            history_pattern=str(pattern),
            nvm_active_power=float(nvm_active_power),
            last_15min_kwh=float(last_15min_kwh),
            provide_future_covariates=bool(provide_future_covariates),
            future_nvm_active_power=float(future_nvm_active_power),
            future_last_15min_kwh=float(future_last_15min_kwh),
        )
    except Exception as e:
        payload_error = str(e)

    with st.expander("Request JSON (preview)"):
        if payload_error:
            st.error(payload_error)
        else:
            st.json(payload)

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        do_predict = st.button("Predict", type="primary", disabled=payload is None)

    with col_btn2:
        if flow == "HTTP":
            try:
                health = check_health(timeout_s=float(timeout_s))
                st.success(f"API health: {health}")
            except Exception as e:
                st.warning(f"API health check failed: {e}")
        else:
            st.info("RPC flow selected. Ensure broker + worker are running.")

    if not do_predict or payload is None:
        return

    st.subheader(f"Result ({flow})")
    try:
        with st.spinner("Predicting..."):
            if flow == "HTTP":
                resp = predict_http(payload, timeout_s=float(timeout_s))
            else:
                resp = rpc_predict(payload, timeout_s=float(timeout_s))
        _render_response(resp)
    except RabbitMQTimeoutError as e:
        st.warning(str(e))
    except RabbitMQError as e:
        st.error(str(e))
    except Exception as e:
        st.error(str(e))


if __name__ == "__main__":
    main()
