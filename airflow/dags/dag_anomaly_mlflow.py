from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
import datetime as dt

args = {
    "owner": "admin",
    "start_date": dt.datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": dt.timedelta(minutes=1),
    "depends_on_past": False,
}

BASE_DIR = "/opt/airflow/dags"
DATA_PATH = f"{BASE_DIR}/traffic_accidents.csv"
MLFLOW_URI = "http://mlflow:5000"

with DAG(
    dag_id="traffic_anomaly_detection_mlflow",
    default_args=args,
    schedule=None,
    catchup=False,
    tags=["anomaly", "mlflow"],
) as dag:


    #task for data preprocessing

    # data_etl = BashOperator(
    #     task_id="data_etl",
    #     bash_command=(
    #         "pip install -q -r {{ params.base }}/requirements.txt && "
    #         "python {{ params.base }}/data_etl.py "
    #         "--input_path {{ params.input_path }} "
    #         "--output_path {{ params.output_path }}"
    #     ),
    #     params={
    #         "base": BASE_DIR,
    #         "input_path": f"{BASE_DIR}/raw_traffic_accidents.csv",
    #         "output_path": DATA_PATH,
    #     },
    # )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            "pip install -q -r {{ params.base }}/requirements.txt && "
            "export MLFLOW_TRACKING_URI={{ params.mlflow_uri }} && "
            "python {{ params.base }}/train_iforest.py "
            "--data {{ params.data }} "
            "--use_date_features"
        ),
        params={
            "base": BASE_DIR,
            "data": DATA_PATH,
            "mlflow_uri": MLFLOW_URI,
        },
    )

    score_model = BashOperator(
        task_id="score_model",
        bash_command=(
            "pip install -q -r {{ params.base }}/requirements.txt && "
            "export MLFLOW_TRACKING_URI={{ params.mlflow_uri }} && "
            "python {{ params.base }}/score_iforest.py "
            "--data {{ params.data }} "
            "--model_uri latest "
            "--out {{ params.base }}/scored_output.csv"
        ),
        params={
            "base": BASE_DIR,
            "data": DATA_PATH,
            "mlflow_uri": MLFLOW_URI,
        },
    )

    train_model >> score_model
