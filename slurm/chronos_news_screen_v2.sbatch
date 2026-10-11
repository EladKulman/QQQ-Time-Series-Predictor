#!/usr/bin/env bash
#SBATCH --job-name=qqq-news-screen-v2
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=studentkillable
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:1
#SBATCH --array=0-35%6
#SBATCH --output=slurm-news-v2-%A_%a.out

# The original news-source screen (configs/news_ablation.json), rerun with the corrected
# recipe: all twelve setups x seeds 42/43/44 = 36 fits. Task i -> setup SETUPS[i % 12], seed 42 + i / 12.
#   export CHRONOS_PYTHON=/usr/bin/python3.12; sbatch slurm/chronos_news_screen_v2.sbatch

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch as an array job}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

TASK_PYTHON="${CHRONOS_PYTHON:-}"
if [[ -z "$TASK_PYTHON" ]]; then
  if [[ -x .venv/bin/python && -x .venv/bin/pip ]]; then TASK_PYTHON=.venv/bin/python; else TASK_PYTHON=python3; fi
fi
if [[ -d .pythonlibs ]]; then export PYTHONPATH="$PWD/.pythonlibs${PYTHONPATH:+:$PYTHONPATH}"; fi
if [[ -d .torchlibs ]]; then export PYTHONPATH="$PWD/.torchlibs${PYTHONPATH:+:$PYTHONPATH}"; fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/models/huggingface}"

SETUPS=(control uncertainty fomc_tone sec_disclosures gdelt_all all_external
        gdelt_ai gdelt_semiconductor gdelt_fed gdelt_inflation gdelt_big_tech_earnings gdelt_recession)
SETUP="${SETUPS[$((SLURM_ARRAY_TASK_ID % 12))]}"
SEED=$((42 + SLURM_ARRAY_TASK_ID / 12))
OUTPUT_ROOT="${NEWS_SCREEN_OUTPUT_ROOT:-models/news-screen-v2-${SLURM_ARRAY_JOB_ID}}"
mkdir -p "$OUTPUT_ROOT"

"$TASK_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is visible in the Slurm allocation")
print("GPU:", torch.cuda.get_device_name(0))
PY
"$TASK_PYTHON" -m pip freeze > "$OUTPUT_ROOT/environment-${SETUP}-seed${SEED}.txt"
"$TASK_PYTHON" scripts/check_data_readiness.py --allow-stale-news \
  --as-of 2026-09-04T23:59:59Z \
  --output "$OUTPUT_ROOT/data_readiness-${SETUP}-seed${SEED}.json"
echo "setup=$SETUP seed=$SEED"
"$TASK_PYTHON" -u scripts/run_news_ablation.py \
  --config-name "$SETUP" --seed "$SEED" --output-root "$OUTPUT_ROOT" --device-map cuda
