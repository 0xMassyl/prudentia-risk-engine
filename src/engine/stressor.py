import yaml
from pathlib import Path
from scipy.stats import norm
from pydantic import BaseModel
# Suppression de l'import 'Optional' qui faisait planter la CI
# Suppression de 'numpy' s'il était là pour rien

from src.domain.entities import Portfolio

class MacroScenario(BaseModel):
    name: str
    description: str
    gdp_growth: float
    unemployment_rate: float
    shock_factor: float

class StressEngine:
    """
    Engine for macroeconomic stress simulation (Stress Testing).
    """
    
    def __init__(self, config_path: str = "config/stress_scenarios.yaml"):
        self.scenarios = self._load_scenarios(config_path)

    def _load_scenarios(self, path: str) -> dict[str, MacroScenario]:
        root_path = Path(__file__).resolve().parent.parent.parent
        full_path = root_path / path
        
        default_scenarios = {
            "baseline": MacroScenario(name="baseline", description="Default", gdp_growth=0.015, unemployment_rate=0.07, shock_factor=0.0),
            "adverse": MacroScenario(name="adverse", description="Adverse", gdp_growth=-0.01, unemployment_rate=0.09, shock_factor=1.5),
            "severely_adverse": MacroScenario(name="severely_adverse", description="Severe", gdp_growth=-0.05, unemployment_rate=0.12, shock_factor=3.0)
        }

        if not full_path.exists():
            return default_scenarios
            
        try:
            with open(full_path, 'r') as f:
                data = yaml.safe_load(f)
            return {k: MacroScenario(name=k, **v) for k, v in data['scenarios'].items()}
        except Exception as e:
            print(f"Config Error: {e}")
            return default_scenarios

    def apply_stress(self, portfolio: Portfolio, scenario_name: str, sensitivity: float = 1.0) -> Portfolio:
        # Utilisation de .get() qui renvoie None par défaut, mais on gère le cas juste après
        scenario = self.scenarios.get(scenario_name)
        
        if scenario is None:
            scenario = self.scenarios.get("adverse")
            # Sécurité absolue si même "adverse" a disparu (impossible avec le default, mais pour le typage)
            if scenario is None:
                 return portfolio 

        if scenario.shock_factor == 0:
            return portfolio

        new_loans = []
        for loan in portfolio.loans:
            stressed_loan = loan.model_copy()
            
            pd_safe = max(min(loan.pd, 0.999), 1e-5)
            z_score = float(norm.ppf(pd_safe))
            shifted_z = z_score + (scenario.shock_factor * sensitivity)
            stressed_pd = float(norm.cdf(shifted_z))
            
            stressed_loan.pd = stressed_pd
            new_loans.append(stressed_loan)
            
        return Portfolio(loans=new_loans)
