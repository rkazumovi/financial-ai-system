"""
Quant-system API routes: option pricing, portfolio optimization, and risk
metrics. All stateless, pure-function endpoints -- no model checkpoint
loading required, unlike a future fraud-scoring endpoint would need.
"""
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.quant.stochastic import (
    black_scholes_price, simulate_heston_paths, mc_call_price, implied_vol,
    S0 as HESTON_S0, RATE as HESTON_RATE, MATURITY as HESTON_MATURITY,
    V0, KAPPA, THETA, SIGMA_V, RHO,
)
from src.quant.portfolio import (
    min_variance_portfolio_numerical, max_sharpe_portfolio_numerical, portfolio_stats,
)
from src.quant.risk import risk_report

router = APIRouter(prefix="/quant", tags=["quant"])


# ---- Black-Scholes -----------------------------------------------------

class BlackScholesRequest(BaseModel):
    spot: float = Field(..., gt=0, description="Current underlying price S")
    strike: float = Field(..., gt=0, description="Strike price K")
    maturity: float = Field(..., gt=0, description="Time to maturity in years")
    rate: float = Field(0.05, description="Risk-free rate")
    sigma: float = Field(..., gt=0, description="Volatility")


class BlackScholesResponse(BaseModel):
    price: float


@router.post("/black-scholes", response_model=BlackScholesResponse)
def price_black_scholes(req: BlackScholesRequest):
    price = black_scholes_price(req.spot, req.strike, req.maturity, req.rate, req.sigma)
    return BlackScholesResponse(price=float(price))


# ---- Heston volatility smile --------------------------------------------

class HestonSmileRequest(BaseModel):
    strikes: List[float] = Field(..., min_length=1, max_length=20)
    spot: float = Field(HESTON_S0, gt=0)
    rate: float = Field(HESTON_RATE)
    maturity: float = Field(HESTON_MATURITY, gt=0)
    n_paths: int = Field(50_000, ge=1000, le=500_000, description="Monte Carlo path count")


class SmilePoint(BaseModel):
    strike: float
    price: float
    implied_vol: Optional[float]


class HestonSmileResponse(BaseModel):
    points: List[SmilePoint]


@router.post("/heston-smile", response_model=HestonSmileResponse)
def heston_smile(req: HestonSmileRequest):
    S_T = simulate_heston_paths(
        req.spot, V0, KAPPA, THETA, SIGMA_V, RHO, req.rate, req.maturity,
        n_paths=req.n_paths, seed=42,
    )
    points = []
    for K in req.strikes:
        price, _ = mc_call_price(S_T, K, req.rate, req.maturity)
        iv = implied_vol(price, req.spot, K, req.maturity, req.rate)
        points.append(SmilePoint(strike=K, price=float(price),
                                  implied_vol=None if np.isnan(iv) else float(iv)))
    return HestonSmileResponse(points=points)


# ---- Portfolio optimization ----------------------------------------------

class PortfolioRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=2)
    expected_returns: List[float]
    covariance: List[List[float]]
    risk_free_rate: float = Field(0.02)
    long_only: bool = Field(True)


class PortfolioResponse(BaseModel):
    min_variance_weights: dict
    min_variance_stats: dict
    max_sharpe_weights: dict
    max_sharpe_stats: dict


@router.post("/portfolio/optimize", response_model=PortfolioResponse)
def optimize_portfolio(req: PortfolioRequest):
    n = len(req.tickers)
    if len(req.expected_returns) != n:
        raise HTTPException(422, "expected_returns length must match tickers length")
    if len(req.covariance) != n or any(len(row) != n for row in req.covariance):
        raise HTTPException(422, "covariance must be an NxN matrix matching tickers length")

    mu = np.array(req.expected_returns)
    Sigma = np.array(req.covariance)

    eigenvalues = np.linalg.eigvalsh(Sigma)
    if np.any(eigenvalues < -1e-8):
        raise HTTPException(422, "covariance matrix is not positive semi-definite")

    try:
        w_minvar = min_variance_portfolio_numerical(Sigma, long_only=req.long_only)
        w_maxsharpe = max_sharpe_portfolio_numerical(mu, Sigma, r_f=req.risk_free_rate,
                                                       long_only=req.long_only)
    except RuntimeError as e:
        raise HTTPException(500, f"optimizer failed to converge: {e}")

    return PortfolioResponse(
        min_variance_weights=dict(zip(req.tickers, [float(w) for w in w_minvar])),
        min_variance_stats=portfolio_stats(w_minvar, mu, Sigma, req.risk_free_rate),
        max_sharpe_weights=dict(zip(req.tickers, [float(w) for w in w_maxsharpe])),
        max_sharpe_stats=portfolio_stats(w_maxsharpe, mu, Sigma, req.risk_free_rate),
    )


# ---- Risk report -----------------------------------------------------------

class RiskReportRequest(BaseModel):
    returns: List[float] = Field(..., min_length=10, description="Period returns, e.g. daily")
    risk_free_rate: float = Field(0.02)
    periods_per_year: int = Field(252)
    confidence: float = Field(0.95, gt=0, lt=1)


@router.post("/risk-report")
def get_risk_report(req: RiskReportRequest):
    returns = np.array(req.returns)
    if not np.all(np.isfinite(returns)):
        raise HTTPException(422, "returns must contain only finite numbers")
    return risk_report(returns, req.risk_free_rate, req.periods_per_year, req.confidence)