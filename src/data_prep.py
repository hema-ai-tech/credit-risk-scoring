"""Load, clean and split the Give Me Some Credit dataset."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, RANDOM_STATE, RAW_DATA_PATH, TARGET, TEST_SIZE

PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
PLACEHOLDER_CODES = [96, 98]  # data-entry codes, not real counts
UTILIZATION_COL = "RevolvingUtilizationOfUnsecuredLines"
UTILIZATION_CAP = 5.0
DEBT_RATIO_CAP = 10.0


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH, index_col=0)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based cleaning. Nothing here is learned from the data, so no leakage."""
    df = df[df["age"] >= 18].copy()
    df[PAST_DUE_COLS] = df[PAST_DUE_COLS].replace(PLACEHOLDER_CODES, np.nan)
    df[UTILIZATION_COL] = df[UTILIZATION_COL].clip(upper=UTILIZATION_CAP)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=DEBT_RATIO_CAP)
    return df.reset_index(drop=True)


def split_and_save(df: pd.DataFrame) -> None:
    train, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"Train: {train.shape} | default rate: {train[TARGET].mean():.2%}")
    print(f"Test:  {test.shape} | default rate: {test[TARGET].mean():.2%}")


def main() -> None:
    raw = load_raw()
    print(f"Raw data: {raw.shape}")
    clean = clean_data(raw)
    print(f"After cleaning: {clean.shape}")
    print("\nMissing values after cleaning:")
    print(clean.isna().sum()[lambda s: s > 0].to_string())
    print()
    split_and_save(clean)


if __name__ == "__main__":
    main()