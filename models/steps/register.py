"""
Step: register
Registra model.pkl en el Model Registry de MLflow y lo pone en 'Production'.
"""

import argparse, pathlib, yaml, mlflow, json
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODEL_NAME = "f1_winner_v1"

def main(params_file: str = "params.yaml"):
    cfg = yaml.safe_load(open(params_file))
    mlflow.set_experiment(cfg["experiment"]["name"])

    model_path = str(pathlib.Path("model/model.pkl").resolve())
    metrics_path = pathlib.Path("model/metrics.json").resolve()
    card_path = pathlib.Path("model_card.json").resolve()

    with mlflow.start_run(run_name="register") as run:
        run_id = run.info.run_id

        # Sube artefactos al run actual
        if metrics_path.exists():
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        if card_path.exists():
            mlflow.log_artifact(str(card_path), artifact_path="model_card")

        # Registra el modelo y obtén su versión
        mv = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model/model.pkl",  # <- importante
            name=MODEL_NAME
        )

    # Transiciona a Production (fuera del contexto del run)
    client = MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"✔ Modelo registrado: {MODEL_NAME} v{mv.version} → Production")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", default="params.yaml")
    args = parser.parse_args()
    main(args.params_file)
