from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

from airflow import DAG
from airflow.decorators import task


with DAG(
    "Full_Pipeline",
    description="Preprocess -> Train -> Deploy (docker-compose)",
    schedule=timedelta(minutes=5),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["full_pipeline"],
) as dag:

    @task
    def preprocess():
        project_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(project_root / "code"))
        from datasets.preprocess import run_preprocess

        run_preprocess(project_root)

    @task
    def train():
        project_root = Path(__file__).resolve().parents[3]
        script = project_root / "code" / "models" / "train_model.py"
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(project_root))

    @task
    def deploy():
        project_root = Path(__file__).resolve().parents[3]
        compose_file = project_root / "code" / "deployment" / "docker-compose.yml"
        subprocess.run([
            "docker", "compose", "-f", str(compose_file), "up", "--build", "-d"
        ], check=True, cwd=str(project_root))

    preprocess() >> train() >> deploy()