from datetime import datetime, timedelta
from pathlib import Path
import sys
from airflow import DAG
from airflow.decorators import task

with DAG(
    "Preprocess_Data",
    description="Preprocessing data: removing duplicates, filling missing values, one-hot encoding",
    schedule=timedelta(minutes=5),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["preprocess"]
) as dag:
    
    @task
    def preprocess_data():
        project_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(project_root / "code"))
        from datasets.preprocess import run_preprocess

        run_preprocess(project_root)

    preprocess_data()