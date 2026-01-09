from __future__ import annotations

import os
from typing import Any

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    import requests  # type: ignore


def _get_api_url(explicit_api_url: str | None = None) -> str:
    api_url = (explicit_api_url or os.environ.get("API_URL") or "http://api:8000").strip()
    return api_url.rstrip("/")

def _get_json(url: str, *, timeout_s: float) -> tuple[int, Any]:
    if httpx is not None:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(url)
        return resp.status_code, resp.json() if resp.content else None

    resp = requests.get(url, timeout=timeout_s)
    return resp.status_code, resp.json() if resp.content else None


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> tuple[int, Any]:
    if httpx is not None:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, json=payload)
        return resp.status_code, resp.json() if resp.content else None

    resp = requests.post(url, json=payload, timeout=timeout_s)
    return resp.status_code, resp.json() if resp.content else None


def check_health(*, api_url: str | None = None, timeout_s: float = 5.0) -> dict[str, Any]:
    """
    Calls `GET {API_URL}/health`.
    Returns the parsed JSON body.
    """
    base = _get_api_url(api_url)
    url = f"{base}/health"

    status_code, body = _get_json(url, timeout_s=timeout_s)
    if status_code != 200:
        raise RuntimeError(f"Health check failed ({status_code}): {body}")

    if not isinstance(body, dict):
        raise RuntimeError(f"Health check returned non-JSON object: {body}")
    return body


def predict_http(
    payload: dict[str, Any],
    *,
    api_url: str | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """
    Calls `POST {API_URL}/predict` with `payload` as JSON.
    Returns the parsed JSON body.
    """
    base = _get_api_url(api_url)
    url = f"{base}/predict"

    status_code, body = _post_json(url, payload, timeout_s=timeout_s)
    if status_code != 200:
        raise RuntimeError(f"Predict failed ({status_code}): {body}")

    if not isinstance(body, dict):
        raise RuntimeError(f"Predict returned non-JSON object: {body}")
    return body
