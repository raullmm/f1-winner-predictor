"""
FastAPI – F1 Winner Predictor
· POST /predict          → prob. de victoria para un piloto concreto
· GET  /predict_winner   → Top-k pilotos con mayor prob. de ganar el GP
"""
# ───────────────────────── imports ─────────────────────────
import os
from pathlib import Path
from datetime import datetime as dt

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ───────────────────────── config ──────────────────────────
MODEL_STAGE   = os.getenv("MODEL_STAGE", "Production")
MLFLOW_URI    = os.getenv("MLFLOW_URI")          # si no existe, usa model.pkl local
DATA_DIR      = Path("data/raw")
MODEL_PATH    = "model.pkl"
WIN_THRESHOLD = 0.50                             # 50 % = etiqueta booleana

# ───────────────────────── modelo ──────────────────────────
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

# ───────────────────────── datos CSV ───────────────────────
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

# standings con fecha de la carrera
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

# ───────────────────── helpers características ─────────────
def _last_k(df: pd.DataFrame, col: str, k: int, agg: str = "sum"):
    if df.empty:
        return None
    return getattr(df.iloc[-k:][col], agg)()

def _stand_before(df: pd.DataFrame, entity_col: str, ent_id: int,
                  race_dt: pd.Timestamp, value_col: str):
    sub = df[(df[entity_col] == ent_id) & (df["race_dt"] < race_dt)]
    if sub.empty:
        return None
    return sub.sort_values("race_dt").iloc[-1][value_col]

def _qual_position(race_id: int, driver_id: int):
    row = qualifying.query("raceId == @race_id and driverId == @driver_id")
    if row.empty:
        return None
    return int(row.iloc[0]["position"])

# ─────────── NUEVO: lista de inscritos robusta ─────────────
def _entry_list_for_race(race_id: int, race_dt: pd.Timestamp) -> pd.DataFrame:
    """driverId-constructorId del GP, estimado si aún no hay resultados."""
    grid = results.query("raceId == @race_id")[["driverId", "constructorId"]]
    if not grid.empty:
        return grid.drop_duplicates()

    grid = qualifying.query("raceId == @race_id")[["driverId", "constructorId"]]
    if not grid.empty:
        return grid.drop_duplicates()

    prev_races = races[races["date"] < race_dt].sort_values("date", ascending=False)
    for prev_id in prev_races["raceId"]:
        grid = results.query("raceId == @prev_id")[["driverId", "constructorId"]]
        if not grid.empty:
            return grid.drop_duplicates()

    return pd.DataFrame(columns=["driverId", "constructorId"])  # vacío

# ─────────────────── predicción individual ─────────────────
def _predict_single(driver_id: int, constructor_id: int,
                    circuit_id: int, race_dt: pd.Timestamp,
                    race_id: int) -> float:

    hist = results.merge(
        races[["raceId", "date", "circuitId"]]
              .rename(columns={"date": "race_dt"}),
        on="raceId"
    ).query("race_dt < @race_dt")

    drv_hist = hist[hist["driverId"] == driver_id].sort_values("race_dt")
    con_hist = hist[hist["constructorId"] == constructor_id].sort_values("race_dt")

    feats = {
        # forma reciente
        "drv_pts_last5":   _last_k(drv_hist, "points", 5),
        "con_pts_last5":   _last_k(con_hist, "points", 5),
        "drv_pts_last3":   _last_k(drv_hist, "points", 3),
        "con_pts_last3":   _last_k(con_hist, "points", 3),
        "drv_avg_pos_last3": _last_k(drv_hist, "positionOrder", 3, "mean"),
        # temporada
        "season_points_so_far":
            drv_hist.query("race_dt.dt.year == @race_dt.year")["points"].sum(),
        # standings previos
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
        # histórico constructor-circuito
        "constructor_win_pct_history":
            con_hist.assign(win=lambda d: d["positionOrder"] == 1)
                    .groupby("circuitId")["win"]
                    .mean().reindex([circuit_id]).fillna(0).iloc[0],
        # posición de qualy (si existe)
        "qual_pos": _qual_position(race_id, driver_id)
    }

    X = pd.DataFrame([feats])

    # 1) Añadimos cualquier columna faltante como NaN
    for col in feat_cols:
        if col not in X.columns:
            X[col] = pd.NA                # LightGBM maneja missing → OK

    # 2) Orden EXACTO de columnas que espera el modelo
    X = X[feat_cols]

    # 3) Garantizamos float64 (LightGBM lanzará warning si no)
    X = X.astype("float64")

    return float(model.predict_proba(X)[0, 1])

# ────────────────────────── FastAPI ────────────────────────
app = FastAPI(title="F1 Predictor API")

class PredictRequest(BaseModel):
    year: int
    round: int
    driverId: int
    constructorId: int

@app.get("/")
def root():
    return {"status": "ok", "message": "F1 Predictor API running"}

# ------------------------------ /predict ------------------------------------
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

# --------------------------- /predict_winner --------------------------------
@app.get("/predict_winner")
def predict_winner(year: int, round: int, top_k: int = 1):
    try:
        race_row = races.query("year == @year and round == @round").iloc[0]
    except IndexError:
        raise HTTPException(404, "Gran Premio no encontrado")

    race_id    = int(race_row["raceId"])
    circuit_id = int(race_row["circuitId"])
    race_dt    = pd.to_datetime(race_row["date"])

    grid_df = _entry_list_for_race(race_id, race_dt)
    if grid_df.empty:
        raise HTTPException(404, "No existe parrilla estimable para ese GP")

    preds = []
    for driver_id, constructor_id in grid_df.itertuples(index=False):
        prob = _predict_single(driver_id, constructor_id, circuit_id, race_dt, race_id)
        preds.append((driver_id, prob))

    total_prob = sum(p for _, p in preds) or 1.0
    preds = [(drv, p / total_prob) for drv, p in preds]

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

# ───────────────────────── entrypoint local ─────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
