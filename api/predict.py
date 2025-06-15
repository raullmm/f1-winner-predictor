"""
FastAPI – F1 Winner Predictor
· POST /predict          → probabilidad de victoria para un piloto concreto
· GET  /predict_winner   → piloto(s) con mayor probabilidad de ganar el GP
"""
import os
from pathlib import Path
from datetime import datetime as dt

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MODEL_STAGE   = os.getenv("MODEL_STAGE", "Production")
MLFLOW_URI    = os.getenv("MLFLOW_URI")          # si no existe, usa model.pkl
DATA_DIR      = Path("data/raw")
MODEL_PATH    = "model.pkl"
WIN_THRESHOLD = 0.50                             # 50 % etiqueta booleana

# ----------------------------------------------------------------------
# CARGAR MODELO
# ----------------------------------------------------------------------
model = None
if MLFLOW_URI:
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        model = mlflow.sklearn.load_model(f"models:/f1_winner_v1/{MODEL_STAGE}")
        print(f"✔ Modelo cargado desde MLflow stage '{MODEL_STAGE}'")
    except Exception as e:
        print(f"⚠️  No se pudo cargar el modelo desde MLflow: {e}")

if model is None:
    try:
        model = joblib.load(MODEL_PATH)
        print("✔ Modelo cargado desde model.pkl")
    except Exception as e:
        raise RuntimeError("❌ No se encontró modelo ni en MLflow ni localmente") from e

feat_cols = list(getattr(model, "feature_names_in_", []))

# ----------------------------------------------------------------------
# CARGAR DATA
# ----------------------------------------------------------------------
drivers   = pd.read_csv(DATA_DIR / "drivers.csv")
results   = pd.read_csv(DATA_DIR / "results.csv")
qualifying = pd.read_csv(DATA_DIR / "qualifying.csv")
races     = pd.read_csv(DATA_DIR / "races.csv", parse_dates=["date"])
d_stand   = pd.read_csv(DATA_DIR / "driver_standings.csv")
c_stand   = pd.read_csv(DATA_DIR / "constructor_standings.csv")

driver_lookup = (
    drivers.assign(full=lambda d: d["forename"].str.strip() + " " +
                            d["surname"].str.strip())
           .set_index("driverId")["full"]
)

# unir standings con fecha de la carrera para filtrar por < race_dt
d_stand = d_stand.merge(races[["raceId", "date"]],
                        on="raceId", how="left") \
                 .rename(columns={"position": "driver_champ_rank",
                                  "points":   "driver_season_pts",
                                  "date":     "race_dt"})

c_stand = c_stand.merge(races[["raceId", "date"]],
                        on="raceId", how="left") \
                 .rename(columns={"position": "constructor_champ_rank",
                                  "points":   "constructor_season_pts",
                                  "date":     "race_dt"})

# ----------------------------------------------------------------------
# FASTAPI
# ----------------------------------------------------------------------
app = FastAPI(title="F1 Predictor API")

class PredictRequest(BaseModel):
    year: int
    round: int
    driverId: int
    constructorId: int

# ----------------------------------------------------------------------
# FUNCIÓN AUXILIAR
# ----------------------------------------------------------------------
def _last_k(df: pd.DataFrame, col: str, k: int, agg: str = "sum"):
    """Retorna función agregada sobre las últimas k filas."""
    if df.empty:
        return None
    return getattr(df.iloc[-k:][col], agg)()

def _stand_before(df: pd.DataFrame, entity_col: str, ent_id: int,
                  race_dt: pd.Timestamp, value_col: str):
    """Valor de standings más reciente *antes* de race_dt."""
    sub = df[(df[entity_col] == ent_id) & (df["race_dt"] < race_dt)]
    if sub.empty:
        return None
    return sub.sort_values("race_dt").iloc[-1][value_col]

def _qual_position(race_id: int, driver_id: int):
    row = qualifying.query("raceId == @race_id and driverId == @driver_id")
    if row.empty:
        return None
    return int(row.iloc[0]["position"])

def _predict_single(driver_id: int, constructor_id: int,
                    circuit_id: int, race_dt: pd.Timestamp,
                    race_id: int) -> float:
    """
    Devuelve probabilidad de victoria para un (driver, constructor) dado
    estado 48 h antes de la carrera indicada.
    """
    hist = results.merge(
        races[["raceId", "date", "circuitId"]]
              .rename(columns={"date": "race_dt"}),
        on="raceId"
    ).query("race_dt < @race_dt")

    drv_hist = hist[hist["driverId"] == driver_id].sort_values("race_dt")
    con_hist = hist[hist["constructorId"] == constructor_id].sort_values("race_dt")

    feats = {
        # forma reciente (5 últimas)
        "drv_pts_last5":   _last_k(drv_hist, "points", 5),
        "con_pts_last5":   _last_k(con_hist, "points", 5),
        # forma más corta (3 últimas, retro-compat)
        "drv_pts_last3":   _last_k(drv_hist, "points", 3),
        "con_pts_last3":   _last_k(con_hist, "points", 3),
        "drv_avg_pos_last3": _last_k(drv_hist, "positionOrder", 3, "mean"),
        # puntos temporada
        "season_points_so_far":
            drv_hist.query("race_dt.dt.year == @race_dt.year")["points"].sum(),
        # standings antes de la carrera
        "driver_champ_rank":
            _stand_before(d_stand, "driverId", driver_id, race_dt,
                          "driver_champ_rank"),
        "driver_season_pts":
            _stand_before(d_stand, "driverId", driver_id, race_dt,
                          "driver_season_pts"),
        "constructor_champ_rank":
            _stand_before(c_stand, "constructorId", constructor_id, race_dt,
                          "constructor_champ_rank"),
        "constructor_season_pts":
            _stand_before(c_stand, "constructorId", constructor_id, race_dt,
                          "constructor_season_pts"),
        # histórico de victorias del constructor en el circuito
        "constructor_win_pct_history":
            con_hist.assign(win=lambda d: d["positionOrder"] == 1)
                    .groupby("circuitId")["win"]
                    .mean().reindex([circuit_id]).fillna(0).iloc[0],
        # posición en qualy
        "qual_pos": _qual_position(race_id, driver_id)
    }

    # asegurar mismas columnas que entrenamiento
    X = pd.DataFrame([feats])
    for col in feat_cols:
        if col not in X.columns:
            X[col] = None
    X = X[feat_cols].fillna(X.median(numeric_only=True))

    return float(model.predict_proba(X)[0, 1])

# ----------------------------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "F1 Predictor API running"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        race_row = races.query("year == @req.year and round == @req.round").iloc[0]
    except IndexError:
        raise HTTPException(404, "Gran Premio no encontrado")

    prob = _predict_single(
        driver_id      = req.driverId,
        constructor_id = req.constructorId,
        circuit_id     = int(race_row["circuitId"]),
        race_dt        = pd.to_datetime(race_row["date"]),
        race_id        = int(race_row["raceId"])
    )

    return {
        "driver": driver_lookup.get(req.driverId, f"ID {req.driverId}"),
        "year": req.year,
        "round": req.round,
        "win_probability": prob,
        "predicted_winner": prob >= WIN_THRESHOLD
    }

@app.get("/predict_winner")
def predict_winner(year: int, round: int, top_k: int = 1):
    try:
        race_row = races.query("year == @year and round == @round").iloc[0]
    except IndexError:
        raise HTTPException(404, "Gran Premio no encontrado")

    race_id    = int(race_row["raceId"])
    circuit_id = int(race_row["circuitId"])
    race_dt    = pd.to_datetime(race_row["date"])

    # ⏳ pilotos inscritos en esa carrera
    grid = (
        results.query("raceId == @race_id")[["driverId", "constructorId"]]
               .drop_duplicates()
               .itertuples(index=False)
    )

    preds = []
    for driver_id, constructor_id in grid:
        prob = _predict_single(driver_id, constructor_id, circuit_id, race_dt, race_id)
        preds.append((driver_id, prob))

    # 📏 NORMALIZAR para que las probabilidades sumen 1
    total_prob = sum(p for _, p in preds) or 1.0      # evita división por 0
    preds = [(drv, p / total_prob) for drv, p in preds]

    # elegir Top-k
    preds.sort(key=lambda t: t[1], reverse=True)
    top = preds[: max(1, top_k)]

    return {
        "gp": {"year": year, "round": round},
        "generated_at": dt.utcnow().isoformat(timespec="seconds") + "Z",
        "top": [
            {
                "driverId": drv,
                "driver": driver_lookup.get(drv, f"ID {drv}"),
                "win_probability": prob
            }
            for drv, prob in top
        ]
    }


# Entrypoint local -------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
