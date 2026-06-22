#!/usr/bin/env bash
# Container entrypoint. Dispatches to a subcommand so the same image can serve
# the API or run one-off jobs (data generation, training, ETL).
#
#   api      -> run the FastAPI service with Uvicorn (default)
#   train    -> train the demand model
#   etl      -> run the ETL pipeline
#   generate -> generate sample raw data
#   bash     -> drop into a shell
#
set -euo pipefail

CMD="${1:-api}"
shift || true

case "$CMD" in
  api)
    exec uvicorn app.api.main:app \
      --host "${API_HOST:-0.0.0.0}" \
      --port "${API_PORT:-8000}" \
      --workers "${UVICORN_WORKERS:-1}"
    ;;
  train)
    exec python -m scripts.train_model "$@"
    ;;
  etl)
    exec python -m scripts.run_etl "$@"
    ;;
  generate)
    exec python -m scripts.generate_sample_data "$@"
    ;;
  predict)
    exec python -m scripts.run_prediction "$@"
    ;;
  bash|sh)
    exec /bin/bash
    ;;
  *)
    # Fall through: treat the argument as an executable command.
    exec "$CMD" "$@"
    ;;
esac
