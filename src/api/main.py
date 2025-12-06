import os
import pickle
from contextlib import asynccontextmanager
from typing import Optional

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
stress_engine: Optional[StressEngine] = None

# Initialisation du moteur de stress
try:
    stress_engine = StressEngine()
except Exception as e:
    print(f"Error initializing StressEngine: {e}")
    stress_engine = None

# --- Lifecycle Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_pipeline
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                ml_pipeline = pickle.load(f)
            print("✅ Credit Scoring Model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Error loading .pkl model: {e}")
    else:
        print(f"ℹ️ No ML model found at {MODEL_PATH}. Prediction endpoint will be disabled.")
    
    yield

# === DÉFINITION DE L'APPLICATION ===
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
    """Helper function to aggregate risk metrics for a portfolio."""
    total_ead = portfolio.total_exposure
    
    if total_ead == 0:
        return AssessmentResult(
            total_exposure=0, total_expected_loss=0, total_rwa=0, capital_requirement=0, average_pd=0
        )

    # CORRECTION E741: Remplacement de 'l' par 'loan' pour éviter l'ambiguïté
    total_el = sum(calculate_expected_loss(loan) for loan in portfolio.loans)
    total_rwa = sum(calculate_rwa(loan) for loan in portfolio.loans)
    
    pds = [loan.pd for loan in portfolio.loans]
    avg_pd = np.mean(pds) if pds else 0.0

    return AssessmentResult(
        total_exposure=total_ead,
        total_expected_loss=total_el,
        total_rwa=total_rwa,
        capital_requirement=total_rwa * 0.08, # Basel minimum 8%
        average_pd=float(avg_pd),
    )

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "active", "system": "Prudentia Risk Engine"}

@app.post("/assess/regulatory", response_model=AssessmentResult)
def assess_regulatory_capital(portfolio: Portfolio):
    return compute_portfolio_metrics(portfolio)

@app.post("/assess/stress-test", response_model=StressTestResult)
def run_stress_test(portfolio: Portfolio, scenario: str = "adverse"):
    if stress_engine is None:
        raise HTTPException(status_code=500, detail="Stress Engine is not initialized check server logs.")

    try:
        baseline_metrics = compute_portfolio_metrics(portfolio)
        stressed_portfolio = stress_engine.apply_stress(portfolio, scenario_name=scenario)
        stressed_metrics = compute_portfolio_metrics(stressed_portfolio)
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
    if ml_pipeline is None:
        raise HTTPException(status_code=503, detail="ML model is not loaded.")
    try:
        df = pd.DataFrame(features)
        probas = ml_pipeline.predict_proba(df)[:, 1]
        return {"estimated_pds": probas.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)