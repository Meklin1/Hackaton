#!/usr/bin/env bash
APP="Exoplanet Server"
MODULE="src.main:app"
PORT=8081
VENV_PATH=".venv/bin/activate"

echo "🚀 Starting ${APP} on port ${PORT} ..."

if [ ! -f "${VENV_PATH}" ]; then
    echo "❌ Virtual Environment not found in $VENV_PATH"
    exit 1
fi

# Activar entorno virtual
source "$VENV_PATH"

# Iniciar servidor
uvicorn "${MODULE}" --reload --port "${PORT}"
