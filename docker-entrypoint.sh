#!/bin/sh
set -eu

export PYTHONPATH="/app${PYTHONPATH:+:$PYTHONPATH}"

python /app/scripts/download_checkpoint.py
exec "$@"
