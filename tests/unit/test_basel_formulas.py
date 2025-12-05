import pytest
from src.domain.entities import Loan, ExposureType
from src.domain.basel_formulas import (
    calculate_asset_correlation,
    vasicek_model_capital,
    calculate_rwa,
    calculate_expected_loss,
)

# Constants for tests (typical Basel regulatory values)
PD_LOW = 0.005   # 0.5% PD (Good-quality firm)
PD_HIGH = 0.05   # 5% PD (Risky firm)
LGD_STANDARD = 0.45  # 45% LGD (standard for unsecured senior loan)
EAD_LARGE = 1_000_000  # 1 million EAD

# Fixture (reusable test data)
@pytest.fixture
def standard_corporate_loan() -> Loan:
    """Standard corporate loan for a large entity."""
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
    """High-risk SME loan (SME correlation adjustment applied)."""
    return Loan(
        id="SME001",
        pd=PD_HIGH,
        lgd=LGD_STANDARD,
        ead=EAD_LARGE / 2,
        maturity=4.0,  # Longer maturity than 2.5 years
        exposure_type=ExposureType.SME,
        turnover=10_000_000.0  # Low turnover → SME adjustment applies
    )

# --- 1. Tests for Correlation Function ---
def test_correlation_corporate_decreases_with_pd(standard_corporate_loan: Loan):
    """Checks that correlation R decreases when PD increases (Basel-required behavior)."""
    corr_low_pd = calculate_asset_correlation(standard_corporate_loan)
    
    # Simulate a much higher PD
    high_pd_loan = standard_corporate_loan.model_copy(update={'pd': 0.15})
    corr_high_pd = calculate_asset_correlation(high_pd_loan)
    
    # R must decrease when PD increases
    assert corr_low_pd > corr_high_pd

def test_correlation_sme_adjustment(standard_corporate_loan: Loan, high_risk_sme_loan: Loan):
    """Checks that SME adjustment reduces correlation compared to a standard Corporate."""
    
    corporate_corr = calculate_asset_correlation(standard_corporate_loan)
    sme_corr = calculate_asset_correlation(high_risk_sme_loan)
    
    # SME correlation must be lower (SME adjustment reduces R)
    assert sme_corr < corporate_corr

# --- 2. Tests for Regulatory Outputs K and RWA ---
def test_rwa_is_non_zero_and_reasonable(standard_corporate_loan: Loan):
    """Checks that RWA is computed and stays within a plausible range (non-zero, non-infinite)."""
    rwa = calculate_rwa(standard_corporate_loan)
    assert rwa > 0
    # Typical RWA for a good firm is around 20–30% of EAD
    assert 100_000 < rwa < 750_000

def test_capital_increases_with_risk(standard_corporate_loan: Loan):
    """Checks that capital K increases with PD (core risk logic)."""
    k_low = vasicek_model_capital(standard_corporate_loan)
    
    # Simulate PD 10x higher
    high_pd_loan = standard_corporate_loan.model_copy(update={'pd': 0.05})
    k_high = vasicek_model_capital(high_pd_loan)
    
    assert k_high > k_low

def test_expected_loss_calculation(standard_corporate_loan: Loan):
    """Checks EL calculation: PD * LGD * EAD."""
    el = calculate_expected_loss(standard_corporate_loan)
    # EL = 0.005 * 0.45 * 1,000,000 = 2,250
    assert el == pytest.approx(2250.0)
