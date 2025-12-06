from scipy.stats import norm
from pydantic import BaseModel
from typing import Optional

from src.domain.entities import Portfolio

class MacroScenario(BaseModel):
    name: str
    description: str
    gdp_growth: float
    unemployment_rate: float
    shock_factor: float

class StressEngine:
    """
    Moteur de simulation de stress (Stress Testing).
    Définit uniquement la logique de calcul.
    """
    
    def __init__(self, config_path: str = "ignored"):
        # Scénarios en dur pour éviter les bugs de lecture de fichier
        self.scenarios = {
            "baseline": MacroScenario(name="baseline", description="Baseline", gdp_growth=0.015, unemployment_rate=0.07, shock_factor=0.0),
            "adverse": MacroScenario(name="adverse", description="Adverse", gdp_growth=-0.01, unemployment_rate=0.09, shock_factor=1.5),
            "severely_adverse": MacroScenario(name="severely_adverse", description="Severe", gdp_growth=-0.05, unemployment_rate=0.12, shock_factor=3.0)
        }
        print("✅ Stress Engine initialized.")

    def apply_stress(self, portfolio: Portfolio, scenario_name: str, sensitivity: float = 1.0) -> Portfolio:
        # Récupération sécurisée
        scenario = self.scenarios.get(scenario_name)
        if scenario is None:
            scenario = self.scenarios["adverse"]

        if scenario.shock_factor == 0:
            return portfolio

        new_loans = []
        for loan in portfolio.loans:
            stressed_loan = loan.model_copy()
            
            # Formule Vasicek Probit Shift
            pd_safe = max(min(loan.pd, 0.999), 1e-5)
            z_score = float(norm.ppf(pd_safe))
            shifted_z = z_score + (scenario.shock_factor * sensitivity)
            stressed_pd = float(norm.cdf(shifted_z))
            
            stressed_loan.pd = stressed_pd
            new_loans.append(stressed_loan)
            
        return Portfolio(loans=new_loans)