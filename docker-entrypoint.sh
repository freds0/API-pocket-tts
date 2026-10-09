#!/bin/sh
set -eu

python /app/scripts/download_checkpoint.py
exec "$@"
