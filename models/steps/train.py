"""
Step: train
Entrena un modelo GradientBoosting evitando columnas con fugas de información
(leakage) y registra todo en MLflow.
"""
import argparse
import json
import pathlib
import yaml
import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

# ------------------- CONSTANTES -----------------------------------------
ID_COLS = {"year", "round", "raceId", "driverId", "constructorId"}
LABEL   = "is_winner"               # nombre del target tras build_feature_table

LEAKAGE_COLS = {
    # columnas que sólo existen tras la bandera a cuadros
    "positionOrder", "positionText", "points", "laps", "milliseconds",
    "time", "rank", "fastestLap", "fastestLapTime", "fastestLapSpeed",
    "statusId", "finish_pos", "winner"
}

# ------------------------------------------------------------------------
def main(params_file: str = "params.yml") -> None:
    cfg   = yaml.safe_load(open(params_file))
    seed  = cfg["hyperparams"]["seed"]

    # ---------- leer dataset procesado ----------------------------------
    df = pd.read_csv("data/processed/features.csv")

    # ---------- selección de features -----------------------------------
    # 1) columnas numéricas
    numeric_cols = df.select_dtypes("number").columns

    # 2) quitamos IDs, target y leakage
    feature_cols = [
        c for c in numeric_cols
        if c not in (ID_COLS | LEAKAGE_COLS | {LABEL})
    ]

    X = df[feature_cols]
    y = df[LABEL]

    # ---------- selección de features -----------------------------------
    numeric_cols = df.select_dtypes("number").columns
    feature_cols = [
        c for c in numeric_cols
        if c not in (ID_COLS | LEAKAGE_COLS | {LABEL})
    ]

    X = df[feature_cols]
    y = df[LABEL]

    # 🔧---------------------------------------------------------------
    # Imputación: medianas columna-a-columna
    X = X.fillna(X.median())
    # ---------------------------------------------------------------🔧

    # ---------- split reproducible --------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )

    # ---------- split reproducible --------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )

    # ---------- modelo ---------------------------------------------------
    model = GradientBoostingClassifier(
        learning_rate = cfg["hyperparams"].get("learning_rate", 0.1),
        n_estimators  = cfg["hyperparams"].get("n_estimators", 300),
        max_depth     = cfg["hyperparams"].get("max_depth", 3),
        random_state  = seed
    )
    model.fit(X_tr, y_tr)

    # ---------- métricas -------------------------------------------------
    train_auc = roc_auc_score(y_tr,  model.predict_proba(X_tr)[:, 1])
    val_auc   = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


    # justo después de entrenar y calcular métricas:
    out_dir = pathlib.Path("model"); out_dir.mkdir(exist_ok=True)
    (pd.Series(feature_cols)
    .to_csv("model/feature_cols.csv", index=False, header=False))
    mlflow.log_artifact("model/feature_cols.csv", artifact_path="model")

    # ---------- MLflow ---------------------------------------------------
    mlflow.set_experiment(cfg["experiment"]["name"])
    with mlflow.start_run(tags=cfg["experiment"].get("tags", {}), nested=True):
        # params & métricas
        mlflow.log_params(cfg["hyperparams"])
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_metric("train_auc", train_auc)
        mlflow.log_metric("val_auc",   val_auc)

        # artefactos
        
        model_path = out_dir / "model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        (out_dir / "metrics.json").write_text(
            json.dumps({"train_auc": train_auc, "val_auc": val_auc}, indent=2)
        )

        print(
            f"✅  Train AUC: {train_auc:.3f} | Val AUC: {val_auc:.3f} "
            f"({len(feature_cols)} features)"
        )

# ------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", default="params.yml")
    args = parser.parse_args()
    main(args.params_file)
