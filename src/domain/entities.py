from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator

class ExposureType(str, Enum):
    """
    Exposure types based on Basel regulatory categories.
    """
    CORPORATE = "CORPORATE"
    RETAIL = "RETAIL"
    SME = "SME"  # Small and Medium-sized Enterprises
    FINANCIAL_INSTITUTION = "FINANCIAL_INSTITUTION"

class Loan(BaseModel):
    """
    Represents an individual loan or credit line.
    
    Attributes:
        id (str): Unique loan identifier.
        pd (float): Probability of Default (0.0 to 1.0). Probability that the counterparty defaults within 1 year.
        lgd (float): Loss Given Default (0.0 to 1.0). Percentage loss in case of default.
        ead (float): Exposure at Default. Amount exposed at the moment of default (in Euros).
        maturity (float): Residual maturity in years (M). Default is 2.5 years (Basel standard).
        exposure_type (ExposureType): Regulatory category of the exposure.
        turnover (float): Company's revenue (required for SME adjustment).
    """
    id: str
    pd: float = Field(..., ge=0.0, le=1.0, description="Probability of Default (Annual)")
    lgd: float = Field(..., ge=0.0, le=1.0, description="Loss Given Default")
    ead: float = Field(..., gt=0.0, description="Exposure at Default (Amount)")
    maturity: float = Field(2.5, gt=0.0, description="Maturity in years")
    exposure_type: ExposureType = ExposureType.CORPORATE
    turnover: Optional[float] = Field(None, ge=0.0, description="Annual Turnover for SME adjustment (EUR)")

    @field_validator('pd')
    @classmethod
    def check_pd_floor(cls, v: float) -> float:
        """Basel often imposes a PD floor (e.g., 0.03%). We allow 0 for technical defaults."""
        # Note: For an educational engine, we accept anything between 0 and 1.
        return v

class Portfolio(BaseModel):
    """
    Aggregate of loans.
    """
    loans: list[Loan]
    
    @property
    def total_exposure(self) -> float:
        return sum(loan.ead for loan in self.loans)

    def __len__(self) -> int:
        return len(self.loans)
