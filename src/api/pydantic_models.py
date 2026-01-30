from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime


class CreditScoringRequest(BaseModel):
    """Defines the input features for the credit scoring model.

    Can accept either:
    1. Raw customer data (Recency, Frequency, Monetary_Total, etc. as raw values)
    2. Already scaled features (as floats)

    The API will automatically apply preprocessing if needed.
    """

    model_config = ConfigDict(extra="allow")

    # Basic customer transaction data
    Recency: Optional[float] = Field(None, description="Days since last transaction")
    Frequency: Optional[float] = Field(None, description="Number of transactions")
    Monetary_Total: Optional[float] = Field(None, description="Total spend amount")
    Monetary_Mean: Optional[float] = Field(
        None, description="Average transaction value"
    )
    Monetary_Std: Optional[float] = Field(
        None, description="Standard deviation of transaction value"
    )

    # Optional transaction details (for raw data)
    TransactionStartTime: Optional[str] = Field(
        None, description="Transaction timestamp"
    )
    ChannelId: Optional[str] = Field(None, description="Channel identifier")
    ProductCategory: Optional[str] = Field(None, description="Product category")
    PricingStrategy: Optional[str] = Field(None, description="Pricing strategy")
    Value: Optional[float] = Field(None, description="Transaction value")


class CreditScoringResponse(BaseModel):
    customer_id: str
    probability_of_default: float
    credit_score: int
    risk_tier: str
    approved: bool
