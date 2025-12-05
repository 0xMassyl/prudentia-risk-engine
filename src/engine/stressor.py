import yaml
from pathlib import Path
from scipy.stats import norm
from pydantic import BaseModel
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
    """
    
    def __init__(self, config_path: str = "config/stress_scenarios.yaml"):
        self.scenarios = self._load_scenarios(config_path)

    def _load_scenarios(self, path: str) -> dict[str, MacroScenario]:
        # Astuce pour trouver le fichier config depuis n'importe où
        root_path = Path(__file__).resolve().parent.parent.parent
        full_path = root_path / path
        
        # Scénarios par défaut si le fichier yaml n'existe pas
        default_scenarios = {
            "baseline": MacroScenario(name="baseline", description="Default", gdp_growth=0.015, unemployment_rate=0.07, shock_factor=0.0),
            "adverse": MacroScenario(name="adverse", description="Adverse", gdp_growth=-0.01, unemployment_rate=0.09, shock_factor=1.5),
            "severely_adverse": MacroScenario(name="severely_adverse", description="Severe", gdp_growth=-0.05, unemployment_rate=0.12, shock_factor=3.0)
        }

        if not full_path.exists():
            print(f"Config stress non trouvée à {full_path}. Utilisation défaut.")
            return default_scenarios
            
        try:
            with open(full_path, 'r') as f:
                data = yaml.safe_load(f)
            return {k: MacroScenario(name=k, **v) for k, v in data['scenarios'].items()}
        except Exception as e:
            print(f"Erreur lecture YAML: {e}. Utilisation défaut.")
            return default_scenarios

    def apply_stress(self, portfolio: Portfolio, scenario_name: str, sensitivity: float = 1.0) -> Portfolio:
        if scenario_name not in self.scenarios:
            print(f"Scénario '{scenario_name}' inconnu. Fallback sur 'adverse'.")
            scenario = self.scenarios.get("adverse", list(self.scenarios.values())[0])
        else:
            scenario = self.scenarios[scenario_name]
        
        if scenario.shock_factor == 0:
            return portfolio

        new_loans = []
        for loan in portfolio.loans:
            stressed_loan = loan.model_copy()
            
            # Formule Probit Shift (Vasicek)
            pd_safe = max(min(loan.pd, 0.999), 1e-5)
            z_score = norm.ppf(pd_safe)
            shifted_z = z_score + (scenario.shock_factor * sensitivity)
            stressed_pd = float(norm.cdf(shifted_z))
            
            stressed_loan.pd = stressed_pd
            new_loans.append(stressed_loan)
            
        return Portfolio(loans=new_loans)