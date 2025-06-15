"""
Step: train
Entrena un GradientBoostingClassifier, evitando fugas (leakage), y registra
todo en MLflow.
"""
import argparse, json, pathlib, yaml, joblib, mlflow, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

# ------------------- CONSTANTES -----------------------------------------
ID_COLS      = {"year", "round", "raceId", "driverId", "constructorId"}
LABEL        = "is_winner"
LEAKAGE_COLS = {
    # columnas que sólo existen tras la bandera a cuadros
    "positionOrder", "positionText", "points", "laps", "milliseconds",
    "time", "rank", "fastestLap", "fastestLapTime", "fastestLapSpeed",
    "statusId", "finish_pos", "winner",
}

# ------------------------------------------------------------------------
def main(params_file: str = "params.yml") -> None:
    cfg  = yaml.safe_load(open(params_file))
    seed = cfg["hyperparams"]["seed"]

    # ---------- leer dataset procesado ----------------------------------
    df = pd.read_csv("data/processed/features.csv")

    # ---------- seleccionar columnas numéricas --------------------------
    numeric_cols  = df.select_dtypes("number").columns
    feature_cols = [
        c for c in numeric_cols
        if c not in (ID_COLS | LEAKAGE_COLS | {LABEL})
    ]

    # ---------- matriz X / vector y -------------------------------------
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[LABEL]

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
    ).fit(X_tr, y_tr)

    # ---------- métricas -------------------------------------------------
    train_auc = roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1])
    val_auc   = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    # ---------- artefactos & MLflow -------------------------------------
    out_dir = pathlib.Path("model"); out_dir.mkdir(exist_ok=True)
    joblib.dump(model, out_dir / "model.pkl")

    mlflow.set_experiment(cfg["experiment"]["name"])
    mlflow.log_params(cfg["hyperparams"])
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_metrics({"train_auc": train_auc, "val_auc": val_auc})
    logged = mlflow.sklearn.log_model(model, artifact_path="model")
    run_id = mlflow.active_run().info.run_id

    (out_dir / "run_id.txt").write_text(run_id)
    (out_dir / "metrics.json").write_text(
        json.dumps({"train_auc": train_auc, "val_auc": val_auc}, indent=2)
    )

    print(f"✅  Train AUC: {train_auc:.3f} | Val AUC: {val_auc:.3f} "
          f"({len(feature_cols)} features)")

# ------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", default="params.yml")
    main(parser.parse_args().params_file)
