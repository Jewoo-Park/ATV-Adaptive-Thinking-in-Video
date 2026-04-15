#!/usr/bin/env bash
set -euo pipefail

# Reproducible benchmark venv setup for GRPO_Video (HF backend compatible).
#
# Run this on a GPU/compute node (after qsub -I), because some nodes require
# module-loaded Python runtime libs (libpython3.11.so) even for venv Python.
#
# Usage:
#   bash src/scripts/setup_bench_venv.sh
# Optional env vars:
#   BENCH_VENV=/home/users/ntu/n2500182/scratch/.venv_bench
#   PYTHON_MODULE=python/3.11.7-gcc11

BENCH_VENV="${BENCH_VENV:-/home/users/ntu/n2500182/scratch/.venv_bench}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.7-gcc11}"

if command -v module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE}" || true
fi

python -V

if [[ ! -d "${BENCH_VENV}" ]]; then
  python -m venv "${BENCH_VENV}"
fi

# shellcheck disable=SC1090
source "${BENCH_VENV}/bin/activate"

python -V
python -m pip install -U pip setuptools wheel

# Pinned to the working HF backend setup.
# - numpy<2: avoid resolver conflicts and downstream packages that pin numpy<2
# - tokenizers<0.23: conservative pin used across this repo's scripts
# - transformers==4.57.6: known to recognize qwen2_5_vl and works with our HF backend path
# - huggingface-hub<1: avoids hub>=1.0 incompat constraints from some local packages
python -m pip install -U \
  "numpy<2" \
  "huggingface-hub<1" \
  "tokenizers<0.23" \
  "transformers==4.57.6" \
  "tqdm>=4.66.0" \
  "Pillow>=10.0.0" \
  "pyyaml>=6.0"

python - <<'PY'
import sys
import numpy, transformers, tokenizers, huggingface_hub
print("OK: python", sys.version.split()[0])
print("OK: numpy", numpy.__version__)
print("OK: transformers", transformers.__version__)
print("OK: tokenizers", tokenizers.__version__)
print("OK: huggingface_hub", huggingface_hub.__version__)
PY

echo "[setup_bench_venv] done: ${BENCH_VENV}"
