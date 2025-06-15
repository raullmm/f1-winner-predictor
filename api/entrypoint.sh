#!/usr/bin/env bash
set -e

MODEL_DIR="/mlruns/1"
echo "⏳ Esperando a que aparezca un modelo en $MODEL_DIR ..."

# espera máx. 180 s
for i in {1..180}; do
  MLFILE=$(find "$MODEL_DIR" -type f -name MLmodel | head -n 1)
  if [[ -n "$MLFILE" ]]; then
    echo "✔ Modelo detectado: $MLFILE"
    break
  fi
  echo "❌ Modelo no detectado"
  sleep 1
done

if [[ -z "$MLFILE" ]]; then
  echo "❌ No se encontró un modelo tras 180 s"; exit 1
fi

exec uvicorn predict:app --host 0.0.0.0 --port 8000
