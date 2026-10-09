#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="${1:-$project_root/.conda/api-pocket-tts}"
conda_bin="${CONDA_EXE:-}"
if [[ -z "$conda_bin" ]]; then
    conda_bin="$(type -P conda || true)"
fi
if [[ -z "$conda_bin" && -x /home/fred/anaconda3/bin/conda ]]; then
    conda_bin=/home/fred/anaconda3/bin/conda
fi
if [[ -z "$conda_bin" || ! -x "$conda_bin" ]]; then
    printf 'Conda não encontrado. Defina CONDA_EXE com o caminho do executável.\n' >&2
    exit 1
fi

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$project_root/.conda-pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$project_root/.cache/pip}"
cd "$project_root"
if [[ -f "$env_prefix/conda-meta/history" ]]; then
    "$conda_bin" env update --prefix "$env_prefix" --file environment.yml
else
    "$conda_bin" env create --prefix "$env_prefix" --file environment.yml
fi
"$conda_bin" run --prefix "$env_prefix" python -m pip check
printf '\nAmbiente pronto.\nconda activate "%s"\npython main.py server\n' "$env_prefix"
