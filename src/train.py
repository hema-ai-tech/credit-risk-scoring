"""Train Logistic Regression and XGBoost, compare them with cross-validation, save the models."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features import build_preprocessor, load_split

N_FOLDS = 5


def build_models(scale_pos_weight: float) -> dict[str, Pipeline]:
    """Each model is a full pipeline: preprocessing + classifier."""
    logreg = Pipeline([
        ("preprocessor", build_preprocessor(scale=True)),
        ("model", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])
    xgb = Pipeline([
        ("preprocessor", build_preprocessor(scale=False)),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    return {"logistic_regression": logreg, "xgboost": xgb}


def main() -> None:
    X_train, y_train = load_split("train")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Training rows: {len(X_train)} | scale_pos_weight: {scale_pos_weight:.2f}\n")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = build_models(scale_pos_weight)
    results = []

    for name, pipeline in models.items():
        print(f"Cross-validating {name} ...")
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        )
        results.append({
            "model": name,
            "cv_roc_auc": scores["test_roc_auc"].mean(),
            "cv_roc_auc_std": scores["test_roc_auc"].std(),
            "cv_pr_auc": scores["test_pr_auc"].mean(),
            "cv_pr_auc_std": scores["test_pr_auc"].std(),
        })

    results_df = pd.DataFrame(results).set_index("model").round(4)
    print("\nCross-validation results (training data only):")
    print(results_df.to_string())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv")

    # Fit each model on the full training set and save it
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")
        print(f"Saved models/{name}.joblib")

    best = results_df["cv_roc_auc"].idxmax()
    joblib.dump(models[best], MODELS_DIR / "best_model.joblib")
    print(f"\nBest model by CV ROC-AUC: {best} (saved as models/best_model.joblib)")


if __name__ == "__main__":
    main()