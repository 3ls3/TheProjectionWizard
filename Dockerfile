# ── base image ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

# system libs for pandas / pycaret / lightgbm + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# ── project code ────────────────────────────────────────────────────────────
WORKDIR /app
COPY . /app

# ── dependencies ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt

# ── FastAPI runtime settings ───────────────────────────────────────────────
ENV PORT=8000
EXPOSE $PORT
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"] 