import pytest
from src.domain.entities import Loan, ExposureType
from src.domain.basel_formulas import (
    calculate_asset_correlation,
    vasicek_model_capital,
    calculate_rwa,
    calculate_expected_loss,
)

# Constantes pour les tests (valeurs typiques selon la réglementation Bâle)
PD_LOW = 0.005  # 0.5% PD (Bonne entreprise)
PD_HIGH = 0.05   # 5% PD (Entreprise risquée)
LGD_STANDARD = 0.45 # 45% LGD (standard pour un prêt senior non garanti)
EAD_LARGE = 1_000_000 # 1 Million d'EAD

# Fixture (données de test réutilisables)
@pytest.fixture
def standard_corporate_loan() -> Loan:
    """Prêt d'entreprise standard pour une grande entité."""
    return Loan(
        id="C001",
        pd=PD_LOW,
        lgd=LGD_STANDARD,
        ead=EAD_LARGE,
        maturity=2.5,
        exposure_type=ExposureType.CORPORATE,
        turnover=100_000_000.0
    )

@pytest.fixture
def high_risk_sme_loan() -> Loan:
    """Prêt PME à haut risque (ajustement de corrélation SME appliqué)."""
    return Loan(
        id="SME001",
        pd=PD_HIGH,
        lgd=LGD_STANDARD,
        ead=EAD_LARGE / 2, 
        maturity=4.0, # Maturité plus longue que 2.5 ans
        exposure_type=ExposureType.SME,
        turnover=10_000_000.0 # Turnover bas, applique l'ajustement SME
    )

# --- 1. Tests de la fonction de Corrélation ---
def test_correlation_corporate_decreases_with_pd(standard_corporate_loan: Loan):
    """Vérifie que la corrélation R diminue lorsque la PD augmente (effet Bâle)."""
    corr_low_pd = calculate_asset_correlation(standard_corporate_loan)
    
    # Simuler une PD beaucoup plus élevée
    high_pd_loan = standard_corporate_loan.model_copy(update={'pd': 0.15})
    corr_high_pd = calculate_asset_correlation(high_pd_loan)
    
    # R doit diminuer quand PD augmente (Comportement mathématique requis par Bâle)
    assert corr_low_pd > corr_high_pd

def test_correlation_sme_adjustment(standard_corporate_loan: Loan, high_risk_sme_loan: Loan):
    """Vérifie que l'ajustement SME réduit la corrélation par rapport à un Corporate standard."""
    
    corporate_corr = calculate_asset_correlation(standard_corporate_loan)
    sme_corr = calculate_asset_correlation(high_risk_sme_loan)
    
    # La corrélation SME doit être inférieure (l'ajustement SME réduit le R)
    assert sme_corr < corporate_corr

# --- 2. Tests des sorties réglementaires K et RWA ---
def test_rwa_is_non_zero_and_reasonable(standard_corporate_loan: Loan):
    """Vérifie que le RWA est calculé et reste dans une fourchette plausible (non nul et non infini)."""
    rwa = calculate_rwa(standard_corporate_loan)
    assert rwa > 0
    # RWA typique pour une bonne entreprise est ~20-30% de l'EAD
    assert 100_000 < rwa < 750_000

def test_capital_increases_with_risk(standard_corporate_loan: Loan):
    """Vérifie que le capital K augmente avec la PD (logique métier fondamentale)."""
    k_low = vasicek_model_capital(standard_corporate_loan)
    
    # Simuler une PD 10 fois plus élevée
    high_pd_loan = standard_corporate_loan.model_copy(update={'pd': 0.05})
    k_high = vasicek_model_capital(high_pd_loan)
    
    assert k_high > k_low

def test_expected_loss_calculation(standard_corporate_loan: Loan):
    """Vérifie le calcul de l'EL : PD * LGD * EAD."""
    el = calculate_expected_loss(standard_corporate_loan)
    # EL = 0.005 * 0.45 * 1,000,000 = 2,250
    assert el == pytest.approx(2250.0)