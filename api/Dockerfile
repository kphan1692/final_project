FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV HOST=0.0.0.0 \
    LISTEN_PORT=8000

WORKDIR /app

# scikit-learn/scipy wheels require OpenMP/Fortran runtimes on Debian slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/api/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/api/requirements.txt

COPY api /app/api

# Ensure the default model path exists (model is typically bind-mounted at runtime).
RUN mkdir -p /app/notebook/final/outputs

EXPOSE 8000

CMD ["python", "-m", "api.app.main", "api"]
