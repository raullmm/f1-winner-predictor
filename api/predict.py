"""
FastAPI – F1 Winner Predictor
Calcula las features “48 h antes” (momentum, circuito, clima medio, etc.)
y devuelve la probabilidad de victoria para un piloto en un GP.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import pandas as pd
import joblib
import mlflow
import os

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------
MODEL_STAGE  = os.getenv("MODEL_STAGE", "Production")
MLFLOW_URI   = os.getenv("MLFLOW_URI")          # si no existe, usa model.pkl local
DATA_DIR     = Path("data/raw")                 # CSV descargados por ingest
MODEL_DIR = Path("model")  # o Path("/app/model") si es más explícito
FEAT_PATH = MODEL_DIR / "feature_cols.csv"
MODEL_PATH = "model.pkl"


# ---------------------------------------------------------------------
# Cargar modelo y columnas de features
# ---------------------------------------------------------------------
model = None
if MLFLOW_URI:
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        model = mlflow.sklearn.load_model(f"models:/f1_winner/{MODEL_STAGE}")
    except Exception as e:
        print(f"⚠️ No se pudo cargar el modelo desde MLflow: {e}")

if model is None:
    try:
        model = joblib.load("model.pkl")
        print("✔ Modelo cargado desde model.pkl")
    except Exception as e:
        raise RuntimeError(f"❌ No se encontró model.pkl y tampoco en MLflow: {e}")


# Cargar lista de columnas esperadas por el modelo
try:
    feat_cols = pd.read_csv(FEAT_PATH, header=None)[0].tolist()
except Exception as e:
    raise RuntimeError(f"No se pudo cargar feature_cols.csv: {e}")

# ---------------------------------------------------------------------
# Cargar CSV para hacer look-ups rápidos
# ---------------------------------------------------------------------
drivers   = pd.read_csv(DATA_DIR / "drivers.csv")
results   = pd.read_csv(DATA_DIR / "results.csv")
races     = pd.read_csv(DATA_DIR / "races.csv")
circuits  = pd.read_csv(DATA_DIR / "circuits.csv")

# Mapeo clima medio (simplificado; completa con tus valores reales)
CLIMATE = {
    1: (28.0, 0.10),
    2: (20.0, 0.35),
    # … añade más …
}

# ---------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------
app = FastAPI(title="F1 Predictor API (48h antes)")

class PredictRequest(BaseModel):
    year: int
    round: int
    driverId: int
    constructorId: int

@app.get("/")
def root():
    return {"status": "ok", "message": "F1 Predictor API (48h antes) running"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        race_row = races.query("year == @req.year and round == @req.round").iloc[0]
    except IndexError:
        raise HTTPException(404, f"No se encontró GP {req.year}-R{req.round}")

    circuit_id = int(race_row["circuitId"])
    race_dt    = pd.to_datetime(race_row["date"])

    hist = results.merge(races[["raceId", "date"]].rename(columns={"date": "race_dt"}),
                         on="raceId").query("race_dt < @race_dt")

    drv_hist = hist[hist["driverId"] == req.driverId].sort_values("race_dt")
    con_hist = hist[hist["constructorId"] == req.constructorId].sort_values("race_dt")

    def last_n(df, col, n=3, f="sum"):
        if df.empty: return None
        subset = df.iloc[-n:]
        return getattr(subset[col], f)()

    try:
        feats = {
            "drv_pts_last3":          last_n(drv_hist, "points"),
            "drv_avg_pos_last3":      last_n(drv_hist, "positionOrder", f="mean"),
            "con_pts_last3":          last_n(con_hist, "points"),
            "season_points_so_far":   drv_hist["points"].sum(),
            "constructor_win_pct_history":
                con_hist.assign(win=lambda d: d["positionOrder"]==1)
                        .groupby("circuitId")["win"]
                        .mean()
                        .reindex([circuit_id]).fillna(0).iloc[0],
            "circuit_type":           int(circuits.set_index("circuitId")
                                                   .loc[circuit_id, "street"]),
            "circuit_length_km":      float(circuits.set_index("circuitId")
                                                     .loc[circuit_id, "length"]),
        }
    except Exception as e:
        raise HTTPException(500, f"Error al construir features: {e}")

    # Añadir clima medio
    temp, rain = CLIMATE.get(circuit_id, (25.0, 0.2))
    feats.update({"avg_temp_month": temp, "avg_rain_month": rain})

    # ---------------- inferencia ----------------
    try:
        X = pd.DataFrame([feats])

        # reordenar columnas y completar faltantes
        for col in feat_cols:
            if col not in X.columns:
                X[col] = None
        X = X[feat_cols]

        # imputar faltantes (median strategy, igual que en train.py)
        X = X.fillna(X.median(numeric_only=True))

        prob = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(500, f"Error en predicción: {e}")

    return {
        "driverId": req.driverId,
        "year": req.year,
        "round": req.round,
        "win_probability": prob
    }

# Entrypoint (para modo script)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
