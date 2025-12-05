import numpy as np
from scipy.stats import norm  # type: ignore
from src.domain.entities import Loan, ExposureType

# Constantes réglementaires
CONFIDENCE_LEVEL_IRB = 0.999  # Niveau de confiance 99.9% pour Bâle

def calculate_asset_correlation(loan: Loan) -> float:
    """
    Calcule la corrélation d'actifs (R) selon la formule Bâle III (AIRB).
    """
    # On s'assure de travailler avec des floats natifs
    pd = float(max(loan.pd, 1e-7))
    
    # Formule Corporate Standard
    # np.exp retourne un np.float ou array, on force le cast final
    numerator = 1.0 - float(np.exp(-50.0 * pd))
    denominator = 1.0 - float(np.exp(-50.0))
    k_factor = numerator / denominator
    
    r = 0.12 * k_factor + 0.24 * (1.0 - k_factor)
    
    # Ajustement SME
    if loan.exposure_type == ExposureType.SME and loan.turnover is not None:
        turnover_capped = float(min(max(loan.turnover, 5e6), 50e6))
        sme_adjustment = 0.04 * (1.0 - (turnover_capped - 5e6) / 45e6)
        r -= sme_adjustment
        
    return float(max(r, 0.0))

def maturity_adjustment(loan: Loan, pd: float) -> float:
    """
    Ajustement de maturité (b).
    """
    pd = float(max(pd, 1e-7))
    
    # Calcul de b (smoothed maturity factor)
    b = float((0.11852 - 0.05478 * np.log(pd)) ** 2)
    
    # Facteur d'ajustement
    ma = (1.0 + (loan.maturity - 2.5) * b) / (1.0 - 1.5 * b)
    return float(ma)

def vasicek_model_capital(loan: Loan) -> float:
    """
    Implémentation de la fonction Vasicek pour calculer l'exigence en capital (K).
    """
    if loan.pd == 0:
        return 0.0
    if loan.pd >= 1.0:
        return 0.0

    pd = float(loan.pd)
    lgd = float(loan.lgd)
    
    # 1. Calcul de la corrélation R
    rho = calculate_asset_correlation(loan)
    
    # 2. Terme conditionnel (Inverse Probit)
    # norm.ppf retourne numpy float, on cast en float
    pd_z = float(norm.ppf(pd))
    
    # 3. Choc systémique
    systemic_shock = float(norm.ppf(CONFIDENCE_LEVEL_IRB))
    
    # 4. Calcul du seuil de défaut stressé
    conditional_pd_term = (pd_z + np.sqrt(rho) * systemic_shock) / np.sqrt(1.0 - rho)
    conditional_pd = float(norm.cdf(conditional_pd_term))
    
    # 5. Capital brut
    capital_raw = lgd * (conditional_pd - pd)
    
    # 6. Ajustement Maturité
    ma = maturity_adjustment(loan, pd)
    
    k = capital_raw * ma
    
    return float(max(k, 0.0))

def calculate_rwa(loan: Loan) -> float:
    """
    Calcule les Risk Weighted Assets (RWA).
    RWA = K * 12.5 * EAD
    
    """
    k = vasicek_model_capital(loan)
    return float(k * 12.5 * loan.ead)

def calculate_expected_loss(loan: Loan) -> float:
    """
    Calcule la Perte Attendue (EL).
    """
    return float(loan.pd * loan.lgd * loan.ead)