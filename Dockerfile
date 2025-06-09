# ── base image ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

# system libs for pandas / pycaret / lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# ── project code ────────────────────────────────────────────────────────────
WORKDIR /app
COPY . /app

# ── dependencies ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt

# ── Streamlit runtime settings ─────────────────────────────────────────────
ENV PORT=8501
EXPOSE $PORT
CMD streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0 