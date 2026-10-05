"""Tests for feature engineering and the preprocessing pipeline."""
import numpy as np

from src.config import TARGET
from src.features import add_features, build_preprocessor


def test_missing_flags(make_df):
    df = make_df({"NumberOfTimes90DaysLate": np.nan, "MonthlyIncome": np.nan}, {})
    out = add_features(df)
    assert out["PastDueMissing"].tolist() == [1, 0]
    assert out["IncomeMissing"].tolist() == [1, 0]
    assert out["DependentsMissing"].tolist() == [0, 0]


def test_total_past_due(make_df):
    df = make_df(
        {
            "NumberOfTime30-59DaysPastDueNotWorse": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 2,
            "NumberOfTimes90DaysLate": 3,
        },
        {"NumberOfTimes90DaysLate": np.nan},
    )
    out = add_features(df)
    assert out.loc[0, "TotalPastDue"] == 6
    assert np.isnan(out.loc[1, "TotalPastDue"])


def test_income_per_person(make_df):
    df = make_df(
        {"MonthlyIncome": 6000.0, "NumberOfDependents": 2},
        {"MonthlyIncome": 4000.0, "NumberOfDependents": 0},
    )
    out = add_features(df)
    assert out["IncomePerPerson"].tolist() == [2000.0, 4000.0]


def test_preprocessor_fills_all_missing_values(make_df):
    df = make_df(
        {"MonthlyIncome": np.nan, "NumberOfDependents": np.nan,
         "NumberOfTimes90DaysLate": np.nan},
        {},
    )
    X = add_features(df.drop(columns=[TARGET]))
    out = build_preprocessor().fit_transform(X)
    assert out.shape[0] == 2
    assert out.isna().sum().sum() == 0