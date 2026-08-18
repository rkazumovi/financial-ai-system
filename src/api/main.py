"""
FastAPI application entry point.

Run from the PROJECT ROOT (not from inside src/) so that `src` resolves as
a package:

    uvicorn src.api.main:app --reload --port 8000

Then test with curl, e.g.:

    curl http://127.0.0.1:8000/health
    curl -X POST http://127.0.0.1:8000/quant/black-scholes \
      -H "Content-Type: application/json" \
      -d '{"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "sigma": 0.2}'

Interactive docs (Swagger UI) are auto-generated at /docs once running.
"""
import sys
from pathlib import Path

# Defensive path setup: guarantees `src` is importable as a package even if
# uvicorn is invoked in a way that doesn't already put the project root on
# sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI

from src.api.routes.quant_routes import router as quant_router

app = FastAPI(
    title="Financial AI System API",
    description="Quant modeling endpoints: option pricing, portfolio optimization, risk metrics.",
    version="0.1.0",
)

app.include_router(quant_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "name": "Financial AI System API",
        "status": "running",
        "docs": "/docs",
        "note": "Fraud-scoring endpoints are not yet included -- the fraud "
                "pipeline's trained models (baseline.py, train.py, ensemble.py) "
                "aren't currently persisted to disk as loadable checkpoints.",
    }