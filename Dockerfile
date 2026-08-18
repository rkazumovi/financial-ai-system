# Container for the FastAPI quant-modeling service (src/api).
# Deliberately does NOT include torch/torch_geometric or the fraud pipeline --
# the API layer built so far (option pricing, portfolio optimization, risk
# metrics) doesn't need them. A separate image/stage would be the right home
# for a future GNN-serving endpoint, since torch alone is a very large
# dependency to drag into a lightweight API container.

FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (separate layer) so `docker build` can reuse
# the cached layer on subsequent builds when only application code changes,
# not when requirements are unchanged.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only copy what the API actually needs -- not the whole repo (data/,
# notebooks/, outputs/, the fraud pipeline, etc).
COPY src/__init__.py src/
COPY src/api/ src/api/
COPY src/quant/ src/quant/

EXPOSE 8000

# Basic container-level health check -- lets `docker ps` and orchestrators
# (Kubernetes, Docker Compose) see whether the app is actually serving
# traffic, not just whether the process is alive.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]