import sys
sys.path.insert(0, ".")

import joblib
import pandas as pd
from features import FEATURE_COLUMNS

CUTOFF_DATE = pd.Timestamp("2025-01-01")


def main():
    feat = pd.read_parquet("data/panel_features.parquet")
    test_data = feat[feat["published_date"] >= CUTOFF_DATE].copy()

    model = joblib.load("models/direction_model.joblib")

    X_test = test_data[FEATURE_COLUMNS]
    test_data["predicted_up"] = model.predict(X_test)
    test_data["predicted_proba"] = model.predict_proba(X_test)[:, 1]

    # Label each prediction as correct, a false positive, or a false negative
    def label_error(row):
        if row["target_up"] == row["predicted_up"]:
            return "correct"
        elif row["predicted_up"] == 1 and row["target_up"] == 0:
            return "false_positive"  # predicted up, actually went down
        else:
            return "false_negative"  # predicted down, actually went up

    test_data["error_type"] = test_data.apply(label_error, axis=1)

    print("Overall error breakdown:")
    print(test_data["error_type"].value_counts(normalize=True))

    # Does accuracy differ by company? Find the best and worst predicted tickers
    # (only looking at companies with a reasonable number of test-period rows)
    per_ticker_accuracy = (
        test_data.groupby("ticker")
        .apply(lambda g: pd.Series({
            "n_rows": len(g),
            "accuracy": (g["target_up"] == g["predicted_up"]).mean(),
        }), include_groups=False)
    )
    per_ticker_accuracy = per_ticker_accuracy[per_ticker_accuracy["n_rows"] >= 100]
    per_ticker_accuracy = per_ticker_accuracy.sort_values("accuracy", ascending=False)

    print("\nBest-predicted companies (>=100 test rows):")
    print(per_ticker_accuracy.head(10))
    print("\nWorst-predicted companies (>=100 test rows):")
    print(per_ticker_accuracy.tail(10))

    # Does the model do better on high-volatility or low-volatility days?
    test_data["volatility_bucket"] = pd.qcut(test_data["volatility_20"], q=3, labels=["low", "medium", "high"])
    accuracy_by_volatility = test_data.groupby("volatility_bucket", observed=True).apply(
        lambda g: (g["target_up"] == g["predicted_up"]).mean(), include_groups=False
    )
    print("\nAccuracy by volatility level:")
    print(accuracy_by_volatility)

    per_ticker_accuracy.to_csv("models/error_by_ticker.csv")


if __name__ == "__main__":
    main()