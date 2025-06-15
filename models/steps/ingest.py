"""
Ingesta de datos F1 (dataset único)
-----------------------------------
Descarga los CSV necesarios desde
    https://www.kaggle.com/datasets/jtrotman/formula-1-race-data
si han sido actualizados, y reconstruye la tabla de features.

· Requiere tener configurado el paquete kaggle (KAGGLE_USERNAME / KEY).
· Guarda los ficheros en data/raw/
· Reconstruye features.csv → data/processed/
"""

from __future__ import annotations
import io, json, zipfile, subprocess, pathlib, yaml, pandas as pd

DATASET = "jtrotman/formula-1-race-data"        # <— único dataset

RAW_DIR  = pathlib.Path("data/raw")
PROC_DIR = pathlib.Path("data/processed")
META     = PROC_DIR / "ingest_meta.json"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# helpers                                                            #
# ------------------------------------------------------------------ #
def kaggle_file_list(dataset: str) -> pd.DataFrame:
    """Listing del dataset Kaggle como DataFrame."""
    out = subprocess.check_output(
        ["kaggle", "datasets", "files", "-d", dataset, "--csv"],
        text=True, stderr=subprocess.STDOUT
    )
    clean = "\n".join(l for l in out.splitlines()
                      if not l.lstrip().startswith("Warning"))
    return pd.read_csv(io.StringIO(clean))

def kaggle_download(dataset: str, fname: str) -> None:
    """Descarga <fname> y lo descomprime si viene en .zip."""
    subprocess.check_call([
        "kaggle", "datasets", "download",
        "-d", dataset, "-f", fname, "-p", str(RAW_DIR), "--force"
    ])
    z = RAW_DIR / f"{fname}.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)
        z.unlink()

def pull_if_new(fname: str) -> None:
    """
    Baja fname sólo si la copia remota (en Kaggle) es más nueva
    que la local (según fecha de modificación).
    """
    df_ls = kaggle_file_list(DATASET)
    col_f = next(c for c in df_ls.columns if c.lower() in {"name", "filename", "file"})
    col_d = next(c for c in df_ls.columns if "date" in c.lower())
    row   = df_ls.loc[df_ls[col_f] == fname]
    if row.empty:
        raise FileNotFoundError(f"{fname} no existe en {DATASET}")

    remote_date = pd.to_datetime(row.iloc[0][col_d])
    local_path  = RAW_DIR / fname

    if local_path.exists():
        local_date = pd.to_datetime(local_path.stat().st_mtime, unit="s")
        if local_date >= remote_date:
            print(f"✓ {fname} ya está actualizado")
            return

    print(f"⬇️  Descargando {fname} …")
    kaggle_download(DATASET, fname)

# ------------------------------------------------------------------ #
def main(params_file: str = "params.yml"):
    cfg   = yaml.safe_load(open(params_file))
    files = cfg["data"]["files"]

    # ---------- 1. Descargar / actualizar CSV -------------------------
    for f in files:
        pull_if_new(f)

    # ---------- 2. ¿Necesitamos reconstruir features? -----------------
    last_meta  = json.loads(META.read_text()) if META.exists() else {}
    races_csv  = pd.read_csv(RAW_DIR / "races.csv")
    newest_yr  = int(races_csv["year"].max())
    features_ok = (PROC_DIR / "features.csv").exists()

    if last_meta.get("latest_year") == newest_yr and features_ok:
        print("✓ Datos al día → se omite reconstrucción")
        return

    print("⚙  Reconstruyendo features …")
    df = build_feature_table()
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC_DIR / "features.csv", index=False)

    META.write_text(json.dumps({"latest_year": newest_yr}, indent=2))
    print(f"✔ features.csv ({len(df):,} filas) | Último año = {newest_yr}")

# ------------------------------------------------------------------ #
# ---------- build_feature_table (sin cambios sustanciales) --------- #
NULLS = ["\\N", ""]
# models/steps/ingest.py  (fragmento corregido)
# --------------------------------------------------------------
def build_feature_table() -> pd.DataFrame:
    """
    Devuelve un DataFrame con las mismas features que el notebook
    experimenting.ipynb para cada fila (driver × carrera).
    """
    NULLS = ["\\N", ""]

    # --- Cargar tablas crudas ----------------------------------
    races_df        = pd.read_csv(RAW_DIR / "races.csv",        na_values=NULLS,
                                  parse_dates=["date"])
    drivers_df      = pd.read_csv(RAW_DIR / "drivers.csv",      na_values=NULLS,
                                  parse_dates=["dob"])
    results_df      = pd.read_csv(RAW_DIR / "results.csv",      na_values=NULLS)
    qualifying_df   = pd.read_csv(RAW_DIR / "qualifying.csv",   na_values=NULLS)
    constructors_df = pd.read_csv(RAW_DIR / "constructors.csv", na_values=NULLS)
    d_stand_df      = pd.read_csv(RAW_DIR / "driver_standings.csv",
                                  na_values=NULLS)

    # --- Limpieza + flag ganador ------------------------------
    for col in ("position", "grid"):
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
    results_df["is_winner"] = (results_df["position"] == 1).astype(int)

    # --- Qualy -------------------------------------------------
    q_cols = ["raceId", "driverId", "q1", "q2", "q3", "position"]
    df = (
        results_df.merge(
            qualifying_df[q_cols],
            on=["raceId", "driverId"],
            how="left",
            suffixes=("", "_qual"),
        )
        .rename(columns={"position_qual": "qual_pos", "position": "finish_pos"})
    )

    # --- Metadatos carrera / piloto ---------------------------
    df = (
        df.merge(
            races_df[["raceId", "date", "year", "round", "circuitId"]],
            on="raceId",
            how="left",
        )
        .merge(
            drivers_df[["driverId", "dob", "nationality"]],
            on="driverId",
            how="left",
        )
    )

    df["age"] = ((df["date"] - df["dob"]).dt.days // 365).astype("Int16")

    # --- Puntos acumulados antes de la carrera ----------------
    d_prev = (
        d_stand_df.merge(
            races_df[["raceId", "year", "round"]], on="raceId"
        ).rename(columns={"points": "season_pts"})
    )

    df = df.merge(
        d_prev[["raceId", "driverId", "season_pts"]],
        on=["raceId", "driverId"],
        how="left",
    )

    return df
# --------------------------------------------------------------


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
