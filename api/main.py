"""FastAPI service for credit risk scoring."""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from src.predict import get_scorer

INDEX_PAGE = Path(__file__).resolve().parent / "index.html"

app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0",
    description="Scores loan applicants for default risk and explains the top drivers with SHAP.",
)


class Applicant(BaseModel):
    revolving_utilization: float = Field(..., ge=0, description="Credit card balance / credit limit")
    age: int = Field(..., ge=18, le=110, description="Applicant age in years")
    past_due_30_59: int = Field(..., ge=0, description="Times 30-59 days past due in the last 2 years")
    debt_ratio: float = Field(..., ge=0, description="Monthly debt payments / monthly income")
    monthly_income: float | None = Field(default=None, ge=0, description="Monthly income (optional)")
    open_credit_lines: int = Field(..., ge=0, description="Number of open loans and credit lines")
    times_90_days_late: int = Field(..., ge=0, description="Times 90+ days late")
    real_estate_loans: int = Field(..., ge=0, description="Number of mortgage / real estate loans")
    past_due_60_89: int = Field(..., ge=0, description="Times 60-89 days past due in the last 2 years")
    dependents: int | None = Field(default=None, ge=0, description="Number of dependents (optional)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "revolving_utilization": 0.65,
                "age": 38,
                "past_due_30_59": 1,
                "debt_ratio": 0.4,
                "monthly_income": 5200,
                "open_credit_lines": 6,
                "times_90_days_late": 0,
                "real_estate_loans": 1,
                "past_due_60_89": 0,
                "dependents": 2,
            }
        }
    )


class Factor(BaseModel):
    feature: str
    value: float
    effect: str
    impact: float


class Prediction(BaseModel):
    risk_score: float
    threshold: float
    decision: str
    top_factors: list[Factor]

@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(INDEX_PAGE)

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant) -> dict:
    try:
        scorer = get_scorer()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model files not found. Run: python -m src.train and python -m src.evaluate",
        )
    return scorer.score(applicant.model_dump())