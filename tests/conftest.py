"""Shared test helpers."""
import pandas as pd
import pytest

BASE_ROW = {
    "SeriousDlqin2yrs": 0,
    "RevolvingUtilizationOfUnsecuredLines": 0.5,
    "age": 40,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.3,
    "MonthlyIncome": 5000.0,
    "NumberOfOpenCreditLinesAndLoans": 5,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2,
}


@pytest.fixture
def make_df():
    """Build a DataFrame of applicants. Each argument is a dict of values to override."""

    def _make(*overrides: dict) -> pd.DataFrame:
        return pd.DataFrame([{**BASE_ROW, **o} for o in overrides])

    return _make