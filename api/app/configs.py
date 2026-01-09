from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    if line.startswith("export "):
        line = line[len("export ") :].lstrip()

    if "=" not in line:
        return None

    key, raw = line.split("=", 1)
    key = key.strip()
    if not key or not _ENV_KEY_RE.match(key):
        return None

    value = raw.strip()
    if not value:
        return key, ""

    # Quoted values: preserve everything inside quotes.
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return key, value[1:-1]

    # Unquoted values: allow inline comments like `KEY=val # comment`
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return key, value


def find_dotenv(
    *,
    env_file: str | Path | None = None,
    start_dir: str | Path | None = None,
    filenames: tuple[str, ...] = (".env",),
) -> Path | None:
    """
    Find a dotenv file by searching upward from `start_dir` (default: cwd),
    then from this file's directory as a fallback.
    """
    if env_file is None:
        env_file = os.environ.get("ENV_FILE")

    if env_file is not None:
        p = Path(env_file)
        return p if p.exists() else None

    start = Path(start_dir) if start_dir is not None else Path.cwd()

    def search_up(base: Path) -> Path | None:
        for parent in (base, *base.parents):
            for name in filenames:
                candidate = parent / name
                if candidate.exists() and candidate.is_file():
                    return candidate
        return None

    found = search_up(start)
    if found is not None:
        return found

    return search_up(Path(__file__).resolve().parent)


def load_dotenv(
    *,
    env_file: str | Path | None = None,
    override: bool = False,
) -> Path | None:
    """
    Load key/value pairs from a `.env` file into `os.environ`.

    - Does not require third-party packages (no python-dotenv dependency).
    - By default, does not overwrite existing environment variables.
    """
    path = find_dotenv(env_file=env_file)
    if path is None:
        return None

    # Use utf-8-sig to tolerate BOM (common on Windows-created files).
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parsed = _parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if not override and key in os.environ:
            continue
        os.environ[key] = value

    return path


def _require_non_empty(value: str, *, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _parse_int(value: str, *, name: str) -> int:
    try:
        return int(value)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name} must be an integer") from e


def _parse_port(value: str, *, name: str) -> int:
    port = _parse_int(value, name=name)
    if not (1 <= port <= 65535):
        raise ValueError(f"{name} must be between 1 and 65535")
    return port


@dataclass(frozen=True, slots=True)
class Settings:
    model_path: Path
    host: str
    listen_port: int
    rabbitmq_host: str
    rabbitmq_port: int
    rabbitmq_user: str
    rabbitmq_password: str
    rabbitmq_vhost: str
    prediction_queue: str

    @classmethod
    def from_mapping(cls, env: Mapping[str, str]) -> "Settings":
        model_path = Path(
            env.get("MODEL_PATH", "notebook/final/outputs/model.pkl")
        )
        host = _require_non_empty(env.get("HOST", "0.0.0.0"), name="HOST")
        listen_port = _parse_port(env.get("LISTEN_PORT", "8000"), name="LISTEN_PORT")
        rabbitmq_host = _require_non_empty(
            env.get("RABBITMQ_HOST", "localhost"), name="RABBITMQ_HOST"
        )
        rabbitmq_port = _parse_port(env.get("RABBITMQ_PORT", "5672"), name="RABBITMQ_PORT")
        rabbitmq_user = _require_non_empty(
            env.get("RABBITMQ_USER", "guest"), name="RABBITMQ_USER"
        )
        rabbitmq_password = _require_non_empty(
            env.get("RABBITMQ_PASSWORD", "guest"), name="RABBITMQ_PASSWORD"
        )
        rabbitmq_vhost = _require_non_empty(
            env.get("RABBITMQ_VHOST", "/"), name="RABBITMQ_VHOST"
        )
        prediction_queue = _require_non_empty(
            env.get("PREDICTION_QUEUE", "prediction_queue"),
            name="PREDICTION_QUEUE",
        )
        return cls(
            model_path=model_path,
            host=host,
            listen_port=listen_port,
            rabbitmq_host=rabbitmq_host,
            rabbitmq_port=rabbitmq_port,
            rabbitmq_user=rabbitmq_user,
            rabbitmq_password=rabbitmq_password,
            rabbitmq_vhost=rabbitmq_vhost,
            prediction_queue=prediction_queue,
        )


@lru_cache
def get_settings() -> Settings:
    """
    Load `.env` (if present) then return parsed settings.

    Environment variables take precedence over values in `.env`.
    """
    load_dotenv(override=False)
    return Settings.from_mapping(os.environ)  # type: ignore[arg-type]


# Convenience exports (mirrors the env var names).
_SETTINGS = get_settings()
MODEL_PATH: Path = _SETTINGS.model_path
HOST: str = _SETTINGS.host
LISTEN_PORT: int = _SETTINGS.listen_port
RABBITMQ_HOST: str = _SETTINGS.rabbitmq_host
RABBITMQ_PORT: int = _SETTINGS.rabbitmq_port
RABBITMQ_USER: str = _SETTINGS.rabbitmq_user
RABBITMQ_PASSWORD: str = _SETTINGS.rabbitmq_password
RABBITMQ_VHOST: str = _SETTINGS.rabbitmq_vhost
PREDICTION_QUEUE: str = _SETTINGS.prediction_queue
