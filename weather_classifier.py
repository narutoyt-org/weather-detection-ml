"""
Weather type detection (sun / rain / drizzle / fog / snow)
from precipitation, temperature and wind measurements.

Usage:
    python weather_classifier.py                      # train, compare models, evaluate
    python weather_classifier.py --predict 10.2 12 6 4 3
        (precipitation temp_max temp_min wind month)
"""
import argparse
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "weather.csv"      # Seattle weather dataset
MODEL_PATH = "weather_model.joblib"
BASE = ["precipitation", "temp_max", "temp_min", "wind", "month"]
FEATURES = BASE + ["temp_range", "temp_avg", "wet"]


def add_features(df):
    df = df.copy()
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["wet"] = (df["precipitation"] > 0).astype(int)
    return df


def load_data(path):
    df = pd.read_csv(path).dropna().drop_duplicates()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    return add_features(df)


def train():
    df = load_data(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Class distribution:\n", df["weather"].value_counts(), "\n")

    X, y = df[FEATURES], df["weather"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": Pipeline([("sc", StandardScaler()),
                                         ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "KNN": Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=7))]),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                class_weight="balanced_subsample", random_state=42),
    }

    # Compare models with 5-fold cross-validation on the TRAINING data only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, m in models.items():
        acc = cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy").mean()
        f1 = cross_val_score(m, X_train, y_train, cv=cv, scoring="f1_macro").mean()
        rows.append((name, acc, f1))
    results = pd.DataFrame(rows, columns=["Model", "CV Accuracy", "CV Macro F1"]).sort_values("CV Macro F1", ascending=False)
    print("5-fold cross-validation (training data):")
    print(results.round(4).to_string(index=False), "\n")

    # Fit the best model on the full training set and test it once
    best_name = results.iloc[0]["Model"]
    best = models[best_name].fit(X_train, y_train)
    pred = best.predict(X_test)
    print(f"Best model: {best_name}")
    print(f"Test accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"Test macro F1: {f1_score(y_test, pred, average='macro'):.4f}\n")
    print(classification_report(y_test, pred, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix ({best_name})")
    plt.savefig("screenshots/confusion_matrix.png", dpi=150, bbox_inches="tight")

    joblib.dump(best, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return best


def predict(model, precipitation, temp_max, temp_min, wind, month):
    row = pd.DataFrame([[precipitation, temp_max, temp_min, wind, month]], columns=BASE)
    return model.predict(add_features(row)[FEATURES])[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", nargs=5, type=float,
                    metavar=("PRECIP", "TMAX", "TMIN", "WIND", "MONTH"))
    args = ap.parse_args()
    if args.predict:
        try:
            model = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            sys.exit("No saved model. Run `python weather_classifier.py` first.")
        print("Predicted weather:", predict(model, *args.predict))
    else:
        model = train()
        print("\n--- Demo predictions ---")
        for vals in [(10.9, 10.6, 2.8, 4.5, 1), (0.0, 28.0, 14.0, 2.5, 7), (0.0, 7.2, 1.1, 1.8, 12), (5.0, 0.5, -3.0, 3.0, 1)]:
            print(vals, "->", predict(model, *vals))
