from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any


class RabbitMQError(RuntimeError):
    pass


class RabbitMQTimeoutError(TimeoutError):
    pass


@dataclass(frozen=True, slots=True)
class MQSettings:
    rabbitmq_host: str
    rabbitmq_port: int
    rabbitmq_user: str
    rabbitmq_password: str
    rabbitmq_vhost: str
    prediction_queue: str


def _get_settings() -> MQSettings:
    return MQSettings(
        rabbitmq_host=(os.environ.get("RABBITMQ_HOST") or "rabbitmq").strip(),
        rabbitmq_port=int(os.environ.get("RABBITMQ_PORT") or "5672"),
        rabbitmq_user=(os.environ.get("RABBITMQ_USER") or "guest").strip(),
        rabbitmq_password=(os.environ.get("RABBITMQ_PASSWORD") or "guest").strip(),
        rabbitmq_vhost=(os.environ.get("RABBITMQ_VHOST") or "/").strip(),
        prediction_queue=(os.environ.get("PREDICTION_QUEUE") or "prediction_queue").strip(),
    )


def rpc_predict(
    payload: dict[str, Any],
    *,
    timeout_s: float = 10.0,
    settings: MQSettings | None = None,
) -> dict[str, Any]:
    """
    RabbitMQ RPC client call:
    - publish `payload` JSON to `prediction_queue`
    - set props.reply_to to an exclusive callback queue
    - wait for response with matching correlation_id, or timeout
    """
    try:
        import pika  # type: ignore[import-not-found]
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RabbitMQError("RabbitMQ RPC client requires `pika` (pip install pika).") from e

    s = settings or _get_settings()
    corr_id = str(uuid.uuid4())

    credentials = pika.PlainCredentials(s.rabbitmq_user, s.rabbitmq_password)
    params = pika.ConnectionParameters(
        host=s.rabbitmq_host,
        port=s.rabbitmq_port,
        virtual_host=s.rabbitmq_vhost,
        credentials=credentials,
        connection_attempts=1,
        retry_delay=0,
        socket_timeout=min(float(timeout_s), 5.0),
        blocked_connection_timeout=float(timeout_s),
        heartbeat=0,
    )

    connection = None
    channel = None
    try:
        try:
            connection = pika.BlockingConnection(params)
        except Exception as e:
            raise RabbitMQError(
                "Failed to connect to RabbitMQ "
                f"(host={s.rabbitmq_host} port={s.rabbitmq_port} vhost={s.rabbitmq_vhost} user={s.rabbitmq_user}). "
                f"{type(e).__name__}: {e}"
            ) from e

        channel = connection.channel()

        # Ensure the request queue exists.
        channel.queue_declare(queue=s.prediction_queue, durable=True)

        # Exclusive, auto-deleted callback queue.
        result = channel.queue_declare(queue="", exclusive=True, auto_delete=True)
        callback_queue = result.method.queue

        response_body: bytes | None = None

        def on_response(ch, method, props, body: bytes) -> None:  # type: ignore[no-untyped-def]
            nonlocal response_body
            if getattr(props, "correlation_id", None) == corr_id:
                response_body = body

        channel.basic_consume(
            queue=callback_queue,
            on_message_callback=on_response,
            auto_ack=True,
        )

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        channel.basic_publish(
            exchange="",
            routing_key=s.prediction_queue,
            properties=pika.BasicProperties(
                reply_to=callback_queue,
                correlation_id=corr_id,
                content_type="application/json",
                delivery_mode=2,
            ),
            body=body,
        )

        deadline = time.monotonic() + float(timeout_s)
        while response_body is None and time.monotonic() < deadline:
            # Processes network events and dispatches callbacks.
            connection.process_data_events(time_limit=0.25)

        if response_body is None:
            raise RabbitMQTimeoutError(
                f"Timed out after {timeout_s:.1f}s waiting for worker response "
                f"(queue={s.prediction_queue}, corr_id={corr_id})."
            )

        try:
            return json.loads(response_body.decode("utf-8"))
        except Exception as e:
            raise RabbitMQError(
                f"Worker returned non-JSON response (corr_id={corr_id}): {response_body!r}"
            ) from e

    finally:
        try:
            if connection is not None and connection.is_open:
                connection.close()
        except Exception:
            pass


def predict_rpc(
    payload: dict[str, Any],
    *,
    timeout_s: float = 10.0,
    settings: MQSettings | None = None,
) -> dict[str, Any]:
    """
    Backwards/UX-friendly alias for `rpc_predict`.
    """
    return rpc_predict(payload, timeout_s=timeout_s, settings=settings)
