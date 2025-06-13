# models/steps/ingest.py
import yaml, pathlib, pandas as pd, zipfile, subprocess, json
from datetime import datetime as dt
import io, csv

RAW_DIR  = pathlib.Path("data/raw")
PROC_DIR = pathlib.Path("data/processed")
META     = PROC_DIR / "ingest_meta.json"

import io, subprocess, zipfile, pathlib, pandas as pd

RAW_DIR  = pathlib.Path("data/raw")

def kaggle_pull_if_new(dataset: str, file: str) -> None:
    # 1) índice de ficheros
    raw_out = subprocess.check_output(
        ["kaggle", "datasets", "files", "-d", dataset, "--csv"],
        text=True
    )
    clean_csv = "\n".join(
        ln for ln in raw_out.splitlines()
        if not ln.lstrip().startswith("Warning:")
    )
    df = pd.read_csv(io.StringIO(clean_csv))

    col_file = next((c for c in df.columns if c.lower() in {"name", "filename", "file"}), None)
    col_date = next((c for c in df.columns if "date" in c.lower()), None)
    if col_file is None:
        raise RuntimeError(f"No se encontró columna de nombre: {df.columns!r}")

    row = df.loc[df[col_file] == file]
    if row.empty:
        raise FileNotFoundError(f"'{file}' no existe en {dataset}")

    remote_date = pd.to_datetime(row.iloc[0][col_date]) if col_date else None

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    local_path = RAW_DIR / file
    if remote_date is not None and local_path.exists():
        if pd.to_datetime(local_path.stat().st_mtime, unit="s") >= remote_date:
            print(f"✓ {file} ya está actualizado")
            return

    print(f"⬇️  Descargando {file} …")
    subprocess.check_call([
        "kaggle", "datasets", "download",
        "-d", dataset, "-f", file, "-p", str(RAW_DIR), "--force"
    ])

    # ────────── NUEVO BLOQUE ──────────
    zip_path = RAW_DIR / f"{file}.zip"
    if zip_path.exists():                       # modo antiguo → extrae zip
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()
    else:                                       # modo nuevo → archivo ya listo
        print(f"✓ {file} descargado sin compresión")


# ----------------------------------------------------------------------
def main(params_file: str = "params.yml"):
    cfg   = yaml.safe_load(open(params_file))
    ds    = cfg["data"]["kaggle_dataset"]
    files = cfg["data"]["files"]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        kaggle_pull_if_new(ds, f)

    # -------- decide si hace falta reconstruir -------------------------
    last_meta      = json.loads(META.read_text()) if META.exists() else {}
    current_races  = pd.read_csv(RAW_DIR / "races.csv")
    newest_year    = current_races["year"].max()
    processed_ok   = (PROC_DIR / "train.csv").exists()

    if last_meta.get("latest_year") == newest_year and processed_ok:
        print("✓ Datos al día → se omite re-ingesta")
        return

    print("⚙ Reconstruyendo features...")

    # -------- build feature table (todas las carreras) --------------------
    df_full = build_feature_table()

    # carpeta de salida
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # 1️⃣  guarda TODAS las filas en un solo fichero
    out_path = PROC_DIR / "features.csv"
    df_full.to_csv(out_path, index=False)

    # 2️⃣  guarda la meta-info con el último año incluido
    latest_year = int(df_full["year"].max())
    META.write_text(json.dumps({"latest_year": latest_year}, indent=2))

    print(f"✔ features.csv actualizado ({len(df_full)}) filas | Último año = {latest_year}")


# ----------------------------------------------------------------------
CLIMATE = {
    # circuitoId : (avg_temp, avg_rain_prob)
    1: (28.0, 0.10),
    2: (30.0, 0.05),
    # … añade los que necesites
}

# ---------- NUEVO build_feature_table -----------------------------------
def build_feature_table():
    """
    Devuelve un DataFrame con exactamente las mismas features que construyes
    en el notebook `experimenting.ipynb` para cada (driver × race) fila.
    Incluye la columna-target `is_winner`.
    """
    NULLS = ["\\N", ""]                           # tokens NA del API Ergast

    # 1) cargar CSVs crudos ------------------------------------------------
    races        = pd.read_csv(RAW_DIR / "races.csv",        na_values=NULLS,
                               parse_dates=["date"])
    drivers      = pd.read_csv(RAW_DIR / "drivers.csv",      na_values=NULLS,
                               parse_dates=["dob"])
    results      = pd.read_csv(RAW_DIR / "results.csv",      na_values=NULLS)
    qualifying   = pd.read_csv(RAW_DIR / "qualifying.csv",   na_values=NULLS)
    constructors = pd.read_csv(RAW_DIR / "constructors.csv", na_values=NULLS)
    d_standings  = pd.read_csv(RAW_DIR / "driver_standings.csv",
                               na_values=NULLS)

    # 2) limpieza básica + flag de ganador --------------------------------
    for col in ["position", "grid"]:
        results[col] = pd.to_numeric(results[col], errors="coerce")

    results["is_winner"] = (results["position"] == 1).astype(int)

    # 3) merge de tiempos de qualy (q1-q3) + posición de qualy ------------
    q_cols = ["raceId", "driverId", "q1", "q2", "q3", "position"]
    df = results.merge(
            qualifying[q_cols],
            on=["raceId", "driverId"],
            how="left",
            suffixes=("", "_qual")           # evita sobre-escribir columnas
         ).rename(columns={
             "position_qual": "qual_pos",    # puesto en clasificación
             "position":      "finish_pos"   # puesto final en carrera
         })

    # 4) metadatos de la carrera + del piloto -----------------------------
    df = (
        df.merge(races[["raceId", "date", "year", "round", "circuitId"]],
                 on="raceId", how="left")
          .merge(drivers[["driverId", "dob", "nationality"]],
                 on="driverId", how="left")
    )

    # edad del piloto en la fecha de la carrera (≈ años cumplidos)
    df["age"] = ((df["date"] - df["dob"]).dt.days // 365).astype("Int16")

    # 5) puntos del piloto acumulados hasta ESA carrera -------------------
    d_st_prev = (
        d_standings
          .merge(races[["raceId", "year", "round"]], on="raceId")
          .rename(columns={"points": "season_pts"})
    )

    df = df.merge(
            d_st_prev[["raceId", "driverId", "season_pts"]],
            on=["raceId", "driverId"],
            how="left"
         )

    return df


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
