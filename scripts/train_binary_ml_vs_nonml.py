#!/usr/bin/env python3
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "gpu_util_pct",
    "mem_util_pct",
    "mem_used_mb",
    "power_w",
    "temp_c",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "mem_used_frac",
    "power_per_util",
]

ML_LABELS = {"compute_bound", "mem_stream"}
NONML_LABELS = {"memcpy", "idle"}


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label" not in df.columns:
        raise ValueError("Missing 'label' column in features file.")
    df["y"] = df["label"].isin(ML_LABELS).astype(int)
    # Ensure non-ML labels are 0 (idle/memcpy)
    df.loc[df["label"].isin(NONML_LABELS), "y"] = 0
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml_features", required=True, help="Features parquet from ML run (contains compute_bound/mem_stream/idle)")
    ap.add_argument("--nonml_features", required=True, help="Features parquet from non-ML run (contains memcpy/idle)")
    ap.add_argument("--out", default="data/raw_telemetry/merged_ml_vs_nonml.parquet", help="Merged dataset output parquet")
    args = ap.parse_args()

    ml = pd.read_parquet(args.ml_features)
    nonml = pd.read_parquet(args.nonml_features)

    ml = add_binary_target(ml)
    nonml = add_binary_target(nonml)

    # Merge
    df = pd.concat([ml, nonml], ignore_index=True)

    # Drop missing feature rows
    df = df.dropna(subset=FEATURES + ["y"])

    df.to_parquet(args.out, index=False)
    print("Wrote merged:", args.out)
    print("\nCounts by label:")
    print(df["label"].value_counts().to_string())
    print("\nCounts (y):")
    print(df["y"].value_counts().to_string())

    X = df[FEATURES]
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=0,
        stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print("\nBinary report (ML=1, non-ML=0):")
    print(classification_report(y_test, pred, target_names=["non-ML", "ML"]))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, pred))

    # Feature importance
    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(imp.head(10).to_string())


if __name__ == "__main__":
    main()
