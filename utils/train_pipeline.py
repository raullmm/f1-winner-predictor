import mlflow, yaml, pathlib, joblib
from mlflow.projects import run

def run_training(params_file: str = "params.yml"):
    cfg = yaml.safe_load(open(params_file))
    mlflow.set_experiment(cfg["experiment"]["name"])

    mlflow.projects.run(
    uri=".",
    entry_point="pipeline",
    parameters={"params_file": params_file},
    env_manager="local"          # ← usa el Python del contenedor
    )


