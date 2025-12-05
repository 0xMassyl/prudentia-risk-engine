import os
import pickle
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.domain.basel_formulas import calculate_expected_loss, calculate_rwa
from src.domain.entities import Portfolio
from src.engine.stressor import StressEngine

# --- Configuration ---
MODEL_PATH = "data/models/scorecard_model.pkl"
ml_pipeline = None

# Attempt to initialize the stress engine.
# If the config file is missing or corrupted, we catch the error to prevent the API from crashing at startup.
# The API will still work for regulatory calculations, just not for stress testing.
try:
    stress_engine: StressEngine | None = None
except Exception as e:
    print(f"Warning: Failed to initialize StressEngine: {e}")
    print("Stress testing features might be unavailable.")
    stress_engine = None


# --- Lifecycle Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle startup and shutdown logic.
    We load the heavy ML model here (once) to avoid I/O overhead on every request.
    """
    global ml_pipeline
    try:
        # Check existence first to avoid ugly tracebacks if the user skipped the training step
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                ml_pipeline = pickle.load(f)
            print("Credit Scoring model loaded successfully.")
        else:
            print(f"Info: No model found at '{MODEL_PATH}'. Running without ML capabilities.")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    yield
    # Cleanup logic would go here if needed


app = FastAPI(
    title="Prudentia Risk Engine",
    description="Regulatory capital calculation and stress testing API.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- DTOs (Data Transfer Objects) ---
class AssessmentResult(BaseModel):
    total_exposure: float
    total_expected_loss: float
    total_rwa: float
    capital_requirement: float
    average_pd: float


class StressTestResult(BaseModel):
    scenario: str
    baseline_metrics: AssessmentResult
    stressed_metrics: AssessmentResult
    capital_impact: float


# --- Helpers ---
def compute_portfolio_metrics(portfolio: Portfolio) -> AssessmentResult:
    """Computes Basel III/IV aggregates for a given loan portfolio."""
    total_ead = portfolio.total_exposure
    
    # Quick exit for empty portfolios to avoid division by zero later
    if total_ead == 0:
        return AssessmentResult(
            total_exposure=0,
            total_expected_loss=0,
            total_rwa=0,
            capital_requirement=0,
            average_pd=0,
        )

    # We iterate over loans to sum up EL and RWA individually (bottom-up approach)
    total_el = sum(calculate_expected_loss(loan) for loan in portfolio.loans)
    total_rwa = sum(calculate_rwa(loan) for loan in portfolio.loans)
    
    pds = [loan.pd for loan in portfolio.loans]
    avg_pd = np.mean(pds) if pds else 0.0

    return AssessmentResult(
        total_exposure=total_ead,
        total_expected_loss=total_el,
        total_rwa=total_rwa,
        capital_requirement=total_rwa * 0.08,  # Hardcoded 8% minimum capital ratio (Basel standard)
        average_pd=float(avg_pd),
    )


# --- Endpoints ---
@app.get("/")
def health_check():
    return {"status": "active", "system": "Prudentia Risk Engine"}


@app.post("/assess/regulatory", response_model=AssessmentResult)
def assess_regulatory_capital(portfolio: Portfolio):
    """
    Calculates standard regulatory metrics (RWA, EL, Capital) based on current PDs.
    Implements the Basel III ASRF framework.
    """
    return compute_portfolio_metrics(portfolio)


@app.post("/assess/stress-test", response_model=StressTestResult)
def run_stress_test(portfolio: Portfolio, scenario: str = "adverse"):
    """
    Runs a full stress testing simulation:
    1. Computes baseline metrics.
    2. Applies macroeconomic shock to PDs.
    3. Re-computes metrics to measure capital impact.
    """
    if stress_engine is None:
        raise HTTPException(status_code=500, detail="Stress Engine is not initialized.")

    try:
        baseline_metrics = compute_portfolio_metrics(portfolio)
        
        # Apply the Probit Shift to degrade portfolio quality
        stressed_pf = stress_engine.apply_stress(portfolio, scenario_name=scenario)
        
        stressed_metrics = compute_portfolio_metrics(stressed_pf)
        
        # Impact is the extra capital needed to cover the stressed scenario
        impact = stressed_metrics.capital_requirement - baseline_metrics.capital_requirement

        return StressTestResult(
            scenario=scenario,
            baseline_metrics=baseline_metrics,
            stressed_metrics=stressed_metrics,
            capital_impact=impact,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/score")
def predict_score(features: list[dict]):
    """
    Bonus endpoint: Real-time scoring using the trained Logistic Regression model.
    Expects raw features (Age, Income, etc.) and returns estimated Probabilities of Default.
    """
    if ml_pipeline is None:
        raise HTTPException(status_code=503, detail="ML model is not loaded.")
    try:
        df = pd.DataFrame(features)
        # We want the probability of the positive class (Default = 1)
        probas = ml_pipeline.predict_proba(df)[:, 1]
        return {"estimated_pds": probas.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # reload=True is useful for dev but should be disabled in production
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)