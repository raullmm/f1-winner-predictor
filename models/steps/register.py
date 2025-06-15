"""
Step: register
Registra el modelo entrenado en el MLflow Model Registry y lo pasa a 'Production'.
"""
import argparse, pathlib, yaml, mlflow, joblib
from mlflow.tracking import MlflowClient

MODEL_NAME = "f1_winner_v1"

def main(params_file: str = "params.yaml"):
    cfg = yaml.safe_load(open(params_file))
    mlflow.set_experiment(cfg["experiment"]["name"])

    # ----- rutas locales -----
    model_pkl = pathlib.Path("model.pkl").resolve()
    metrics   = pathlib.Path("model/metrics.json").resolve()
    card      = pathlib.Path("model_card.json").resolve()

    # ---------- RUN ----------
    with mlflow.start_run(run_name="register") as run:
        if metrics.exists():
            mlflow.log_artifact(str(metrics), artifact_path="metrics")
        if card.exists():
            mlflow.log_artifact(str(card), artifact_path="model_card")

        # 1) loggea carpeta-modelo
        model_obj = joblib.load(model_pkl)
        logged = mlflow.sklearn.log_model(model_obj, artifact_path="model")
        model_uri = logged.model_uri          # → runs:/<run_id>/model

        # 2) registra la versión (¡sin “model.pkl” extra!)
        mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME,
                                   await_registration_for=300)

    # ---------- PROMOVER ----------
    client = MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Production",           # o usa alias → ver abajo
        archive_existing_versions=True
    )
    print(f"✔ Modelo registrado: {MODEL_NAME} v{mv.version} → Production")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", default="params.yaml")
    main(parser.parse_args().params_file)
