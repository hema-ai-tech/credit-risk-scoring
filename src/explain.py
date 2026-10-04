"""SHAP explainability: global feature importance and a single-applicant explanation."""
import joblib
import matplotlib

matplotlib.use("Agg")  # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features import load_split

SAMPLE_SIZE = 3000


def main() -> None:
    pipeline = joblib.load(MODELS_DIR / "best_model.joblib")
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_test, _ = load_split("test")
    X_sample = X_test.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    X_t = preprocessor.transform(X_sample)  # DataFrame with the 16 feature columns

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_t)
    if isinstance(shap_values, list):  # some versions return one array per class
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)

    # 1. Global importance table (mean absolute SHAP value per feature)
    importance = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=X_t.columns)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .round(4)
    )
    importance.to_csv(REPORTS_DIR / "shap_importance.csv")
    print("Top 10 features by mean |SHAP|:")
    print(importance.head(10).to_string())

    # 2. Bar chart of global importance
    plt.figure()
    shap.summary_plot(shap_values, X_t, plot_type="bar", show=False, max_display=12)
    plt.title("Feature importance (mean |SHAP value|)")
    plt.savefig(REPORTS_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Beeswarm: direction and size of each feature's effect
    plt.figure()
    shap.summary_plot(shap_values, X_t, show=False, max_display=12)
    plt.title("How each feature pushes the default prediction")
    plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Waterfall for the highest-risk applicant in the sample
    probs = pipeline.predict_proba(X_sample)[:, 1]
    idx = int(np.argmax(probs))
    base_value = float(np.ravel(explainer.expected_value)[0])
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=base_value,
        data=X_t.iloc[idx].values,
        feature_names=list(X_t.columns),
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.savefig(REPORTS_DIR / "shap_waterfall_example.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nExample applicant: predicted default probability {probs[idx]:.1%}")
    print("Saved 3 SHAP plots and shap_importance.csv to the reports folder.")


if __name__ == "__main__":
    main()