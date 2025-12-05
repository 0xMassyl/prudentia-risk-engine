import numpy as np
from scipy.stats import norm  # type: ignore
from src.domain.entities import Loan, ExposureType

# Regulatory constants
CONFIDENCE_LEVEL_IRB = 0.999  # 99.9% confidence level for Basel

def calculate_asset_correlation(loan: Loan) -> float:
    """
    Calculates asset correlation (R) using the Basel III (AIRB) formula.
    """
    # Ensure we work with native floats
    pd = float(max(loan.pd, 1e-7))
    
    # Corporate Standard formula
    # np.exp returns a numpy float or array; final cast forced
    numerator = 1.0 - float(np.exp(-50.0 * pd))
    denominator = 1.0 - float(np.exp(-50.0))
    k_factor = numerator / denominator
    
    r = 0.12 * k_factor + 0.24 * (1.0 - k_factor)
    
    # SME adjustment
    if loan.exposure_type == ExposureType.SME and loan.turnover is not None:
        turnover_capped = float(min(max(loan.turnover, 5e6), 50e6))
        sme_adjustment = 0.04 * (1.0 - (turnover_capped - 5e6) / 45e6)
        r -= sme_adjustment
        
    return float(max(r, 0.0))

def maturity_adjustment(loan: Loan, pd: float) -> float:
    """
    Computes the maturity adjustment factor (b).
    """
    pd = float(max(pd, 1e-7))
    
    # Calculation of b (smoothed maturity factor)
    b = float((0.11852 - 0.05478 * np.log(pd)) ** 2)
    
    # Adjustment factor
    ma = (1.0 + (loan.maturity - 2.5) * b) / (1.0 - 1.5 * b)
    return float(ma)

def vasicek_model_capital(loan: Loan) -> float:
    """
    Implements the Vasicek function to compute capital requirement (K).
    """
    if loan.pd == 0:
        return 0.0
    if loan.pd >= 1.0:
        return 0.0

    pd = float(loan.pd)
    lgd = float(loan.lgd)
    
    # 1. Compute correlation R
    rho = calculate_asset_correlation(loan)
    
    # 2. Conditional term (Inverse Probit)
    # norm.ppf returns a numpy float; cast to float
    pd_z = float(norm.ppf(pd))
    
    # 3. Systemic shock
    systemic_shock = float(norm.ppf(CONFIDENCE_LEVEL_IRB))
    
    # 4. Stressed default threshold
    conditional_pd_term = (pd_z + np.sqrt(rho) * systemic_shock) / np.sqrt(1.0 - rho)
    conditional_pd = float(norm.cdf(conditional_pd_term))
    
    # 5. Raw capital
    capital_raw = lgd * (conditional_pd - pd)
    
    # 6. Maturity adjustment
    ma = maturity_adjustment(loan, pd)
    
    k = capital_raw * ma
    
    return float(max(k, 0.0))

def calculate_rwa(loan: Loan) -> float:
    """
    Calculates Risk Weighted Assets (RWA).
    RWA = K * 12.5 * EAD
    """
    k = vasicek_model_capital(loan)
    return float(k * 12.5 * loan.ead)

def calculate_expected_loss(loan: Loan) -> float:
    """
    Calculates Expected Loss (EL).
    """
    return float(loan.pd * loan.lgd * loan.ead)
