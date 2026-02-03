import argparse
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_visualizations(df, output_dir):
    """Create anomaly detection visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    sns.set_style("whitegrid")
    
    # 1. Anomaly Score Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df[df["anomaly_pred"] == 1]["anomaly_score"], bins=50, alpha=0.6, label="Normal", color="green")
    plt.hist(df[df["anomaly_pred"] == -1]["anomaly_score"], bins=50, alpha=0.6, label="Anomaly", color="red")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title("Anomaly Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "anomaly_score_distribution.png", dpi=150)
    plt.close()
    
    # 2. Anomaly Percentage Pie Chart
    plt.figure(figsize=(8, 8))
    anomaly_counts = df["anomaly_pred"].value_counts()
    labels = ["Normal" if x == 1 else "Anomaly" for x in anomaly_counts.index]
    colors = ["#90EE90", "#FF6B6B"]
    plt.pie(anomaly_counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title(f"Anomaly Detection Results\nTotal Records: {len(df)}")
    plt.tight_layout()
    plt.savefig(output_dir / "anomaly_percentage.png", dpi=150)
    plt.close()
    
    # 3. Top Anomalies by Score
    plt.figure(figsize=(12, 6))
    top_anomalies = df[df["anomaly_pred"] == -1].nsmallest(20, "anomaly_score")
    plt.barh(range(len(top_anomalies)), top_anomalies["anomaly_score"].values, color="red", alpha=0.7)
    plt.xlabel("Anomaly Score (lower = more anomalous)")
    plt.ylabel("Record Index")
    plt.title("Top 20 Most Anomalous Records")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "top_anomalies.png", dpi=150)
    plt.close()
    
    # 4. Time-based anomaly trend (if date columns exist)
    date_cols = [col for col in df.columns if 'date' in col.lower() or col in ['year', 'month', 'day']]
    if 'year' in df.columns and 'month' in df.columns:
        df_time = df.copy()
        df_time['year_month'] = df_time['year'].astype(str) + '-' + df_time['month'].astype(str).str.zfill(2)
        anomaly_by_time = df_time.groupby('year_month')['anomaly_pred'].apply(lambda x: (x == -1).sum()).reset_index()
        anomaly_by_time.columns = ['year_month', 'anomaly_count']
        
        plt.figure(figsize=(14, 6))
        plt.plot(anomaly_by_time['year_month'], anomaly_by_time['anomaly_count'], marker='o', color='red', linewidth=2)
        plt.xlabel("Time Period")
        plt.ylabel("Number of Anomalies")
        plt.title("Anomaly Trend Over Time")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "anomaly_trend.png", dpi=150)
        plt.close()
    
    # 5. Anomaly Summary Statistics
    summary_stats = {
        "Total Records": len(df),
        "Normal Records": (df["anomaly_pred"] == 1).sum(),
        "Anomalies Detected": (df["anomaly_pred"] == -1).sum(),
        "Anomaly Rate (%)": round((df["anomaly_pred"] == -1).mean() * 100, 2),
        "Min Anomaly Score": round(df["anomaly_score"].min(), 4),
        "Max Anomaly Score": round(df["anomaly_score"].max(), 4),
        "Mean Anomaly Score": round(df["anomaly_score"].mean(), 4),
    }
    
    # Save summary as text file
    with open(output_dir / "summary_statistics.txt", "w") as f:
        f.write("ANOMALY DETECTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n✓ Visualizations saved to {output_dir}")
    print("\nSummary Statistics:")
    for key, value in summary_stats.items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model_uri", default="latest", help="Model URI or 'latest' to use most recent run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--viz_dir", default="/opt/airflow/dags/visualizations", help="Directory to save visualizations")
    args = parser.parse_args()

    # Auto-detect latest run if model_uri is 'latest'
    if args.model_uri == "latest":
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        exp = mlflow.get_experiment_by_name("traffic_anomaly_detection")
        if not exp:
            raise ValueError("Experiment 'traffic_anomaly_detection' not found")
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id], 
            order_by=["start_time DESC"], 
            max_results=1
        )
        if runs.empty:
            raise ValueError("No runs found in experiment")
        run_id = runs.iloc[0]["run_id"]
        args.model_uri = f"runs:/{run_id}/model"
        print(f"Using latest run: {run_id}")

    df = pd.read_csv(args.data)
    model = mlflow.sklearn.load_model(args.model_uri)

    preds = model.predict(df)
    X = model.named_steps["preprocess"].transform(df)
    scores = model.named_steps["model"].decision_function(X)

    df["anomaly_pred"] = preds
    df["anomaly_score"] = scores

    df.to_csv(args.out, index=False)
    print(f"Wrote scored file to {args.out}")
    
    # Create visualizations
    create_visualizations(df, args.viz_dir)


if __name__ == "__main__":
    main()
