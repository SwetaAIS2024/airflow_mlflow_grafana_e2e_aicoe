import argparse
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--use_date_features", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if args.use_date_features and "crash_date" in df.columns:
        dt = pd.to_datetime(df["crash_date"], errors="coerce", format="mixed")
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["day"] = dt.dt.day
        df = df.drop(columns=["crash_date"])

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "bool"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    mlflow.set_experiment("traffic_anomaly_detection")

    with mlflow.start_run() as run:
        pipe.fit(df)
        preds = pipe.predict(df)
        scores = pipe.named_steps["model"].decision_function(
            pipe.named_steps["preprocess"].transform(df)
        )

        mlflow.log_metric("anomaly_rate", float((preds == -1).mean()))
        mlflow.sklearn.log_model(pipe, "model")

        print(f"MLflow run_id: {run.info.run_id}")


if __name__ == "__main__":
    main()
