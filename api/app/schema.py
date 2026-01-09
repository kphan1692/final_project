from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Frequency = Literal["15min"]

FREQ_15MIN: Frequency = "15min"
STEP_15MIN = timedelta(minutes=15)
MIN_HISTORY_POINTS = 672


def _require_timezone_aware(dt: datetime, *, field_name: str) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(
            f"{field_name} must include timezone info (e.g. 'Z' or '+00:00')"
        )
    return dt


def _require_finite_number(value: float, *, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")
    return value


def _validate_regular_time_grid(
    timestamps: list[datetime],
    *,
    step: timedelta,
    label: str,
) -> None:
    if not timestamps:
        raise ValueError(f"{label} must not be empty")

    for idx, ts in enumerate(timestamps):
        _require_timezone_aware(ts, field_name=f"{label}[{idx}]")

    for idx in range(1, len(timestamps)):
        prev = timestamps[idx - 1]
        cur = timestamps[idx]
        if cur <= prev:
            raise ValueError(
                f"{label} must be strictly increasing; got {label}[{idx-1}]={prev.isoformat()} "
                f"and {label}[{idx}]={cur.isoformat()}"
            )
        delta = cur - prev
        if delta != step:
            raise ValueError(
                f"{label} must be spaced by {int(step.total_seconds() // 60)} minutes; "
                f"got {label}[{idx-1}]={prev.isoformat()} and {label}[{idx}]={cur.isoformat()} "
                f"(delta={delta})"
            )


class HistoryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: datetime = Field(..., description="Timestamp (ISO-8601 date-time).")
    target_ac_power: float = Field(..., description="Observed target value at ts.")
    nvm_active_power: float = Field(..., description="Observed nvm_active_power at ts.")
    last_15min_kwh: float = Field(..., description="Observed last_15min_kwh at ts.")

    @field_validator("ts")
    @classmethod
    def _validate_ts(cls, v: datetime) -> datetime:
        return _require_timezone_aware(v, field_name="ts")

    @field_validator("target_ac_power", "nvm_active_power", "last_15min_kwh")
    @classmethod
    def _validate_floats(cls, v: float, info) -> float:  # type: ignore[override]
        return _require_finite_number(v, field_name=info.field_name)


class FutureCovariatePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: datetime = Field(..., description="Forecast timestamp (ISO-8601 date-time).")
    nvm_active_power: float = Field(..., description="Known/assumed nvm_active_power at ts.")
    last_15min_kwh: float = Field(..., description="Known/assumed last_15min_kwh at ts.")

    @field_validator("ts")
    @classmethod
    def _validate_ts(cls, v: datetime) -> datetime:
        return _require_timezone_aware(v, field_name="ts")

    @field_validator("nvm_active_power", "last_15min_kwh")
    @classmethod
    def _validate_floats(cls, v: float, info) -> float:  # type: ignore[override]
        return _require_finite_number(v, field_name=info.field_name)


class PredictRequest(BaseModel):
    """
    Forecast request for the trained model at `MODEL_PATH` (see README.md).

    Contract choice: "latest known values + horizon".
    The worker derives the model feature columns (lags/rollings/time features) from `history`
    and optionally uses `future_covariates` (else forward-fills covariates from the last
    history point).
    """

    model_config = ConfigDict(extra="forbid")

    freq: Frequency = Field(
        default=FREQ_15MIN,
        description="Sampling frequency. Only '15min' is supported by this model.",
    )
    horizon_steps: int = Field(
        ...,
        ge=1,
        description="Number of 15-minute steps to forecast (>= 1).",
    )
    history: list[HistoryPoint] = Field(
        ...,
        min_length=MIN_HISTORY_POINTS,
        description=(
            "Most recent observations, in ascending timestamp order. "
            f"Must contain at least {MIN_HISTORY_POINTS} points at 15-minute frequency."
        ),
    )
    future_covariates: list[FutureCovariatePoint] | None = Field(
        default=None,
        description=(
            "Optional covariates for each forecast timestamp; must have length == horizon_steps. "
            "If omitted, the service should forward-fill covariates from the last history row."
        ),
    )

    @model_validator(mode="after")
    def _validate_request(self) -> "PredictRequest":
        if self.freq != FREQ_15MIN:
            raise ValueError("freq must be '15min'")

        # Validate only the tail needed for lag/rolling feature creation.
        # (Allows callers to send longer history without requiring the entire series to be perfect.)
        history_tail = self.history[-MIN_HISTORY_POINTS:]
        history_ts = [p.ts for p in history_tail]
        _validate_regular_time_grid(history_ts, step=STEP_15MIN, label="history.ts[-672:]")

        if self.future_covariates is not None:
            if len(self.future_covariates) != self.horizon_steps:
                raise ValueError("future_covariates must have length equal to horizon_steps")

            future_ts = [p.ts for p in self.future_covariates]
            _validate_regular_time_grid(
                future_ts, step=STEP_15MIN, label="future_covariates.ts"
            )

            expected_start = history_tail[-1].ts + STEP_15MIN
            if future_ts[0] != expected_start:
                raise ValueError(
                    "future_covariates[0].ts must equal last history ts + 15 minutes "
                    f"({expected_start.isoformat()})"
                )

            for idx in range(self.horizon_steps):
                expected = expected_start + STEP_15MIN * idx
                if future_ts[idx] != expected:
                    raise ValueError(
                        f"future_covariates[{idx}].ts must be {expected.isoformat()} "
                        "(15-minute steps from last history ts)"
                    )

        return self


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: datetime = Field(..., description="Forecast timestamp (ISO-8601 date-time).")
    yhat: float = Field(..., description="Forecast value for target_ac_power.")

    @field_validator("ts")
    @classmethod
    def _validate_ts(cls, v: datetime) -> datetime:
        return _require_timezone_aware(v, field_name="ts")

    @field_validator("yhat")
    @classmethod
    def _validate_yhat(cls, v: float) -> float:
        return _require_finite_number(v, field_name="yhat")


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    freq: Frequency = Field(..., description="Echo of request frequency.")
    horizon_steps: int = Field(..., ge=1, description="Echo of request horizon_steps.")
    predictions: list[Prediction] = Field(
        ...,
        min_length=1,
        description="Forecasted points (length == horizon_steps).",
    )
    model_path: str | None = Field(
        default=None,
        description="Path of the model artifact used to serve the request.",
    )

    @model_validator(mode="after")
    def _validate_response(self) -> "PredictResponse":
        if self.freq != FREQ_15MIN:
            raise ValueError("freq must be '15min'")
        if len(self.predictions) != self.horizon_steps:
            raise ValueError("predictions must have length equal to horizon_steps")
        return self
