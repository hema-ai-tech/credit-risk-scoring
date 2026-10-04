"""Evaluate the models on the held-out test set and save plots and metrics."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")  # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features import load_split

MODEL_NAMES = ["logistic_regression", "xgboost"]
LABELS = {"logistic_regression": "Logistic Regression", "xgboost": "XGBoost"}


def find_threshold(pipeline, X_train, y_train) -> float:
    """Pick the probability cutoff that maximises F1, using out-of-fold training predictions."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, oof)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1[:-1]))  # last precision/recall point has no threshold
    return float(thresholds[best_idx])


def main() -> None:
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    models = {name: joblib.load(MODELS_DIR / f"{name}.joblib") for name in MODEL_NAMES}
    probas = {name: m.predict_proba(X_test)[:, 1] for name, m in models.items()}

    # 1. Test metrics for both models
    rows = []
    for name, p in probas.items():
        rows.append({
            "model": name,
            "test_roc_auc": roc_auc_score(y_test, p),
            "test_pr_auc": average_precision_score(y_test, p),
        })
    metrics = pd.DataFrame(rows).set_index("model").round(4)
    print("Test set results:")
    print(metrics.to_string())
    metrics.to_csv(REPORTS_DIR / "test_metrics.csv")

    # 2. ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probas.items():
        RocCurveDisplay.from_predictions(y_test, p, name=LABELS[name], ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    fig.savefig(REPORTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Precision-recall curve
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probas.items():
        PrecisionRecallDisplay.from_predictions(y_test, p, name=LABELS[name], ax=ax)
    ax.axhline(y_test.mean(), color="k", linestyle="--", label="Random")
    ax.set_title("Precision-recall curve (test set)")
    ax.legend(loc="upper right")
    fig.savefig(REPORTS_DIR / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Best model (chosen by cross-validation in train.py), threshold, confusion matrix
    comparison = pd.read_csv(REPORTS_DIR / "model_comparison.csv", index_col="model")
    best = comparison["cv_roc_auc"].idxmax()
    print(f"\nBest model (by CV): {LABELS[best]}")
    print("Choosing decision threshold from training data (takes about a minute)...")
    threshold = find_threshold(models[best], X_train, y_train)
    print(f"Decision threshold: {threshold:.3f}")
    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump({"model": best, "threshold": threshold}, f, indent=2)

    y_pred = (probas[best] >= threshold).astype(int)
    print("\nClassification report (test set):")
    print(classification_report(
        y_test, y_pred, target_names=["No default", "Default"], digits=3
    ))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No default", "Default"],
        cmap="Blues", values_format="d", ax=ax,
    )
    ax.set_title(f"Confusion matrix: {LABELS[best]} (threshold {threshold:.2f})")
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved plots and metrics to the reports folder.")


if __name__ == "__main__":
    main()