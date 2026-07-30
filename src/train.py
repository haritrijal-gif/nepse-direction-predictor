import sys
sys.path.insert(0, ".")

import json
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS

# A fixed "seed" number so random parts of the model behave the same way
# every time we run this script -- makes results reproducible.
RANDOM_STATE = 42

# Everything before this date is used to TRAIN the model.
# Everything from this date onward is used to TEST it (data the model never saw).
CUTOFF_DATE = pd.Timestamp("2025-01-01")


def evaluate(name, y_true, y_pred, y_proba=None):
    """Print/collect the standard set of classification metrics for one model."""
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        # zero_division=0 just tells sklearn "if this metric can't be computed
        # (e.g. a model that never predicts 'up'), report 0 instead of a warning."
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def main():
    # STEP 1: Load the feature-engineered data we built in the previous step.
    feat = pd.read_parquet("data/panel_features.parquet")

    # STEP 2: Split into "before 2025" (train) and "2025 onward" (test).
    train_data = feat[feat["published_date"] < CUTOFF_DATE]
    test_data = feat[feat["published_date"] >= CUTOFF_DATE]

    print(f"train rows: {len(train_data)}")
    print(f"test rows:  {len(test_data)}")

    # X = the input features (the 14 columns the model learns from)
    # y = the target (1 = price went up next day, 0 = it didn't)
    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["target_up"]
    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data["target_up"]

    results = []

    # ---------------------------------------------------------------
    # BASELINE 1: Majority-class guesser.
    # This model doesn't look at the features at all -- it just always
    # predicts whichever class was more common in the training data.
    # Any "real" model below needs to beat this to be worth anything.
    # ---------------------------------------------------------------
    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)
    results.append(evaluate("Majority-class baseline", y_test, baseline_predictions))

    # ---------------------------------------------------------------
    # BASELINE 2: Naive momentum guesser.
    # "Assume tomorrow repeats whatever direction happened today."
    # We already calculated this column back in features.py.
    # ---------------------------------------------------------------
    naive_predictions = test_data["today_direction"]
    results.append(evaluate("Naive momentum (repeat today)", y_test, naive_predictions))

    # ---------------------------------------------------------------
    # MODEL 1: Logistic Regression.
    # A simple, interpretable model. It works better when all features
    # are on a similar numeric scale, so we scale them first.
    # ---------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # learn the scaling from training data...
    X_test_scaled = scaler.transform(X_test)         # ...then apply that SAME scaling to test data

    logreg_model = LogisticRegression(
        max_iter=1000,             # give it enough iterations to finish training properly
        class_weight="balanced",   # don't let it lazily favor the more common class
        random_state=RANDOM_STATE,
    )
    logreg_model.fit(X_train_scaled, y_train)
    logreg_predictions = logreg_model.predict(X_test_scaled)
    logreg_probabilities = logreg_model.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate("Logistic Regression", y_test, logreg_predictions, logreg_probabilities))

    # ---------------------------------------------------------------
    # MODEL 2: Random Forest.
    # Builds many decision trees and averages their answers. Can pick up
    # more complex patterns than logistic regression. Doesn't need scaling.
    # ---------------------------------------------------------------
    rf_model = RandomForestClassifier(
        n_estimators=300,       # how many individual trees to build
        max_depth=8,            # limit how deep each tree can grow (avoids memorizing noise)
        min_samples_leaf=20,    # each final decision must be based on at least 20 examples
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,               # use all CPU cores to train faster
    )
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_probabilities = rf_model.predict_proba(X_test)[:, 1]
    results.append(evaluate("Random Forest", y_test, rf_predictions, rf_probabilities))

    # Print all 4 models' metrics together so we can compare them
    print("\n" + json.dumps(results, indent=2, default=str))

    # STEP 3: Save whichever real model (logreg or random forest) scored
    # higher on ROC-AUC, so the Streamlit app can load it later.
    logreg_auc = results[2]["roc_auc"]
    rf_auc = results[3]["roc_auc"]

    if rf_auc >= logreg_auc:
        best_name = "Random Forest"
        best_model = rf_model
    else:
        best_name = "Logistic Regression"
        best_model = logreg_model

    joblib.dump(best_model, "models/direction_model.joblib")
    # Random Forest doesn't need a separate scaler, but Logistic Regression does --
    # save the scaler too so the app can apply identical scaling later if needed.
    joblib.dump(scaler, "models/scaler.joblib")

    with open("models/metrics.json", "w") as f:
        json.dump({"results": results, "chosen_model": best_name}, f, indent=2, default=str)

    print(f"\nSaved best model ({best_name}) to models/direction_model.joblib")

    # STEP 4: Show which features the Random Forest relied on most.
    importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)
    print("\nRandom Forest feature importances:")
    print(importances)


if __name__ == "__main__":
    main()