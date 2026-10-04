"""Feature engineering and preprocessing pipeline."""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR, TARGET
from src.data_prep import PAST_DUE_COLS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature engineering. Safe to use on training data and at prediction time."""
    df = df.copy()
    df["PastDueMissing"] = df[PAST_DUE_COLS].isna().any(axis=1).astype(int)
    df["IncomeMissing"] = df["MonthlyIncome"].isna().astype(int)
    df["DependentsMissing"] = df["NumberOfDependents"].isna().astype(int)
    df["TotalPastDue"] = df[PAST_DUE_COLS].sum(axis=1, skipna=False)
    df["TotalOpenAccounts"] = (
        df["NumberOfOpenCreditLinesAndLoans"] + df["NumberRealEstateLoansOrLines"]
    )
    df["IncomePerPerson"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    return df


def build_preprocessor(scale: bool = False) -> Pipeline:
    """Median imputation. Set scale=True for Logistic Regression (trees do not need it)."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps).set_output(transform="pandas")


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load 'train' or 'test' and return features X and target y."""
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
    X = add_features(df.drop(columns=[TARGET]))
    y = df[TARGET]
    return X, y


def main() -> None:
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)  # learn medians from train only
    X_test_t = preprocessor.transform(X_test)

    print(f"Train features: {X_train_t.shape} | Test features: {X_test_t.shape}")
    print(f"Missing values left (train): {int(X_train_t.isna().sum().sum())}")
    print(f"Missing values left (test):  {int(X_test_t.isna().sum().sum())}")
    print("\nFeatures:")
    for col in X_train_t.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    main()