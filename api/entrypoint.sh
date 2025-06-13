#!/bin/bash

echo "📁 Archivos disponibles:"
ls -l /app

# Esperar a model.pkl
until [ -f "model.pkl" ]; do
  echo "⏳ Esperando a que model.pkl esté disponible..."
  sleep 2
done

# Esperar a feature_cols.csv
until [ -f "model/feature_cols.csv" ]; do
  echo "⏳ Esperando a feature_cols.csv..."
  sleep 2
done

echo "✅ Archivos encontrados, lanzando API"
exec "$@"
