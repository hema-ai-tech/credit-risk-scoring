"""Load the trained model and score new applicants with explanations."""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import MODELS_DIR
from src.data_prep import (
    DEBT_RATIO_CAP,
    PAST_DUE_COLS,
    PLACEHOLDER_CODES,
    UTILIZATION_CAP,
    UTILIZATION_COL,
)
from src.features import add_features

# API field name -> dataset column name
FIELD_TO_COLUMN = {
    "revolving_utilization": "RevolvingUtilizationOfUnsecuredLines",
    "age": "age",
    "past_due_30_59": "NumberOfTime30-59DaysPastDueNotWorse",
    "debt_ratio": "DebtRatio",
    "monthly_income": "MonthlyIncome",
    "open_credit_lines": "NumberOfOpenCreditLinesAndLoans",
    "times_90_days_late": "NumberOfTimes90DaysLate",
    "real_estate_loans": "NumberRealEstateLoansOrLines",
    "past_due_60_89": "NumberOfTime60-89DaysPastDueNotWorse",
    "dependents": "NumberOfDependents",
}


class CreditRiskScorer:
    def __init__(self) -> None:
        self.pipeline = joblib.load(MODELS_DIR / "best_model.joblib")
        with open(MODELS_DIR / "threshold.json") as f:
            info = json.load(f)
        self.threshold = float(info["threshold"])
        self.model_name = info["model"]
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.explainer = shap.TreeExplainer(self.pipeline.named_steps["model"])

    def _prepare(self, applicant: dict) -> pd.DataFrame:
        """Apply the same cleaning rules and features used in training."""
        row = {col: applicant.get(field) for field, col in FIELD_TO_COLUMN.items()}
        df = pd.DataFrame([row], dtype=float)
        df[PAST_DUE_COLS] = df[PAST_DUE_COLS].replace(PLACEHOLDER_CODES, np.nan)
        df[UTILIZATION_COL] = df[UTILIZATION_COL].clip(upper=UTILIZATION_CAP)
        df["DebtRatio"] = df["DebtRatio"].clip(upper=DEBT_RATIO_CAP)
        df = add_features(df)
        return df[list(self.pipeline.feature_names_in_)]

    def score(self, applicant: dict, top_n: int = 3) -> dict:
        X = self._prepare(applicant)
        risk_score = float(self.pipeline.predict_proba(X)[0, 1])

        X_t = self.preprocessor.transform(X)
        shap_values = self.explainer.shap_values(X_t)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        contributions = np.asarray(shap_values)[0]

        top_idx = np.argsort(-np.abs(contributions))[:top_n]
        top_factors = [
            {
                "feature": str(X_t.columns[i]),
                "value": float(X_t.iloc[0, i]),
                "effect": "increases risk" if contributions[i] > 0 else "decreases risk",
                "impact": round(float(contributions[i]), 4),
            }
            for i in top_idx
        ]
        return {
            "risk_score": round(risk_score, 4),
            "threshold": round(self.threshold, 4),
            "decision": "high_risk" if risk_score >= self.threshold else "low_risk",
            "top_factors": top_factors,
        }


@lru_cache(maxsize=1)
def get_scorer() -> CreditRiskScorer:
    return CreditRiskScorer()


def main() -> None:
    scorer = get_scorer()
    examples = {
        "High-risk example": {
            "revolving_utilization": 1.2, "age": 25, "past_due_30_59": 2,
            "debt_ratio": 0.8, "monthly_income": 2000, "open_credit_lines": 4,
            "times_90_days_late": 2, "real_estate_loans": 0,
            "past_due_60_89": 1, "dependents": 2,
        },
        "Low-risk example": {
            "revolving_utilization": 0.1, "age": 52, "past_due_30_59": 0,
            "debt_ratio": 0.2, "monthly_income": 9000, "open_credit_lines": 8,
            "times_90_days_late": 0, "real_estate_loans": 1,
            "past_due_60_89": 0, "dependents": 1,
        },
    }
    for label, applicant in examples.items():
        result = scorer.score(applicant)
        print(f"\n{label}")
        print(f"  risk_score: {result['risk_score']:.3f} (threshold {result['threshold']:.3f})")
        print(f"  decision:   {result['decision']}")
        for factor in result["top_factors"]:
            print(f"  - {factor['feature']} = {factor['value']:.2f} -> {factor['effect']}")


if __name__ == "__main__":
    main()