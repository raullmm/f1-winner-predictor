"""Step: evaluate
Evalúa el modelo en los datos del último año y registra métrica
"""
import argparse, yaml, pathlib, pandas as pd, joblib, mlflow, json
from sklearn.metrics import roc_auc_score

# columnas que NO son features
ID_COLS = {"year", "round", "raceId", "driverId", "constructorId"}
LABEL   = "is_winner"

LEAKAGE_COLS = {
    "positionOrder", "positionText", "points", "laps", "milliseconds",
    "time", "rank", "fastestLap", "fastestLapTime", "fastestLapSpeed",
    "statusId", "finish_pos", "winner"
}

def main(params_file: str):
    cfg = yaml.safe_load(open(params_file))
    mlflow.set_experiment(cfg["experiment"]["name"])

    # -------- leer datos unificados --------
    df = pd.read_csv("data/processed/features.csv")

    # -------- extraer último año como test --------
    latest_year = df["year"].max()
    test_df = df[df["year"] == latest_year].copy()

    # -------- preparar X, y como en train.py --------
    numeric_cols = test_df.select_dtypes(include="number").columns
    feature_cols = [
        c for c in numeric_cols
        if c not in (ID_COLS | LEAKAGE_COLS | {LABEL})
    ]

    X = test_df[feature_cols].fillna(test_df[feature_cols].median())
    y = test_df[LABEL]

    # -------- carga modelo + eval --------
    run_id = pathlib.Path("model/run_id.txt").read_text().strip()
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    proba = model.predict_proba(X)[:, 1]
    auc   = roc_auc_score(y, proba)

    # -------- registra en MLflow --------
    with mlflow.start_run(run_name="evaluate", nested=True):
        mlflow.log_metric("val_auc", auc)

    # -------- persistir métricas --------
    pathlib.Path("model").mkdir(exist_ok=True)
    (pathlib.Path("model") / "metrics.json").write_text(
        json.dumps({"val_auc": auc}, indent=2)
    )
    print(f"✅ Evaluate: val_auc {auc:.3f} on year {latest_year}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", default="params.yml")
    args = parser.parse_args()
    main(args.params_file)
