"""Tests for the data cleaning rules."""
import numpy as np

from src.data_prep import clean_data


def test_removes_applicants_under_18(make_df):
    df = make_df({"age": 0}, {"age": 17}, {"age": 18}, {})
    cleaned = clean_data(df)
    assert len(cleaned) == 2
    assert cleaned["age"].min() >= 18


def test_placeholder_codes_become_missing(make_df):
    df = make_df(
        {"NumberOfTimes90DaysLate": 98, "NumberOfTime30-59DaysPastDueNotWorse": 96},
        {},
    )
    cleaned = clean_data(df)
    assert np.isnan(cleaned.loc[0, "NumberOfTimes90DaysLate"])
    assert np.isnan(cleaned.loc[0, "NumberOfTime30-59DaysPastDueNotWorse"])
    assert cleaned.loc[1, "NumberOfTimes90DaysLate"] == 0


def test_outliers_are_capped(make_df):
    df = make_df({"RevolvingUtilizationOfUnsecuredLines": 50.0, "DebtRatio": 1000.0})
    cleaned = clean_data(df)
    assert cleaned.loc[0, "RevolvingUtilizationOfUnsecuredLines"] == 5.0
    assert cleaned.loc[0, "DebtRatio"] == 10.0


def test_valid_rows_are_kept(make_df):
    cleaned = clean_data(make_df({}))
    assert len(cleaned) == 1
    assert cleaned.isna().sum().sum() == 0