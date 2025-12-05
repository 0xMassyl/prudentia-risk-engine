from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator

class ExposureType(str, Enum):
    """
    Types d'exposition selon la réglementation Bâloise.
    """
    CORPORATE = "CORPORATE"
    RETAIL = "RETAIL"
    SME = "SME"  # Small and Medium-sized Enterprises
    FINANCIAL_INSTITUTION = "FINANCIAL_INSTITUTION"

class Loan(BaseModel):
    """
    Représente un prêt individuel ou une ligne de crédit.
    
    Attributs:
        id (str): Identifiant unique du prêt.
        pd (float): Probability of Default (0.0 à 1.0). Probabilité que la contrepartie fasse défaut à 1 an.
        lgd (float): Loss Given Default (0.0 à 1.0). Pourcentage de perte en cas de défaut.
        ead (float): Exposure at Default. Montant exposé au moment du défaut (en Euros).
        maturity (float): Maturité résiduelle en années (M). Par défaut 2.5 ans (standard Bâle).
        exposure_type (ExposureType): Catégorie réglementaire de l'exposition.
        turnover (float): Chiffre d'affaires de l'entreprise (nécessaire pour l'ajustement SME).
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
        """Bâle impose souvent un plancher de PD (ex: 0.03%). On autorise 0 pour le défaut technique."""
        # Note: Pour un moteur pédagogique, on accepte tout entre 0 et 1.
        return v

class Portfolio(BaseModel):
    """
    Agrégat de prêts.
    """
    loans: list[Loan]
    
    @property
    def total_exposure(self) -> float:
        return sum(loan.ead for loan in self.loans)

    def __len__(self) -> int:
        return len(self.loans)