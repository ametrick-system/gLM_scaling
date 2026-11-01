#!/bin/bash
#SBATCH --job-name=glm-isoflop
#SBATCH --partition=gpu_h100
#SBATCH --account=pi_ac3458
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --constraint=gpu:h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=amy.metrick@yale.edu

set -uo pipefail

GLM_ROOT="${GLM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${GLM_ROOT}" || { echo "ABORT: cannot cd to ${GLM_ROOT}" >&2; exit 1; }
GRID="${GRID:-runs/grid.jsonl}"
CORPUS_ROOT="${CORPUS_ROOT:-data/corpora}"
RUN_ROOT="${RUN_ROOT:-runs/sweep}"
for need in design.py train.py "${GRID}"; do
  [[ -f "${need}" ]] || { echo "ABORT: ${need} not found under ${GLM_ROOT} -- submit from the repo root" >&2; exit 1; }
done

module load miniconda
set +u; conda activate glm; set -u

# All runs must share one torch build
GLM_EXPECT_TORCH="${GLM_EXPECT_TORCH:-2.11.0+cu128}"
GOT_TORCH="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo MISSING)"
[[ "${GOT_TORCH}" == "${GLM_EXPECT_TORCH}" ]] || {
  echo "ABORT: torch ${GOT_TORCH}, expected ${GLM_EXPECT_TORCH} (wrong stack or it changed)." >&2
  echo "       re-pin deliberately: GLM_EXPECT_TORCH=${GOT_TORCH} sbatch ... sweep.slurm" >&2
  exit 1; }

EXPECT_GPU="${EXPECT_GPU:-H100}"
GOT_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[[ "${GOT_GPU}" == *"${EXPECT_GPU}"* ]] || {
  echo "ABORT: expected a GPU matching '${EXPECT_GPU}', got '${GOT_GPU}'." >&2
  echo "       fix --constraint, or set EXPECT_GPU= to mix cards deliberately." >&2
  exit 1; }
echo "[node] $(hostname)  gpu=${GOT_GPU}  torch=${GOT_TORCH}"

mkdir -p logs "${RUN_ROOT}"

# Shared ON-DISK torch.compile cache
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/glm/inductor}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

# Which grid lines to run: a serial BATCH (GLM_LINES) or one ARRAY task
if [[ -n "${GLM_LINES:-}" ]]; then
  LINES=""
  for tok in ${GLM_LINES//,/ }; do
    if [[ "${tok}" == *-* ]]; then LINES+=" $(seq "${tok%-*}" "${tok#*-}")"; else LINES+=" ${tok}"; fi
  done
else
  LINES="${SLURM_ARRAY_TASK_ID:?submit as an array (sbatch --array=...) OR set GLM_LINES=0-14 for a serial batch}"
fi
echo "[batch] lines:${LINES}"

export GLM_NODE="$(hostname)"
export GLM_GPU="${GOT_GPU}"
export GLM_SLURM_JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-na}}_${SLURM_ARRAY_TASK_ID:-na}"

rc=0
for LINE in ${LINES}; do
  # Per line: spread + size_index + the FROZEN per-N peak LR (design.lr_for)
  read -r SPREAD SIZE_INDEX LR < <(GRID="${GRID}" LINE="${LINE}" python - <<'PY'
import json, os, sys
sys.path.insert(0, os.getcwd())
import design
d = json.loads(open(os.environ["GRID"]).read().splitlines()[int(os.environ["LINE"])])
print(d["spread"], d["size_index"], f'{design.lr_for(d["n_non_embedding"]):.6e}')
PY
)
  [[ "${LR}" =~ ^[0-9.eE+-]+$ ]] || { echo "SKIP line ${LINE}: LR='${LR}' not numeric -- design.lr_for failed" >&2; rc=1; continue; }

  # Keep weights only for the Chinchilla-optimal size (index 2 of 5); the isoFLOP fit reads final.json, not checkpoints
  SAVE_CKPT=""; [[ "${SIZE_INDEX}" == "2" ]] && SAVE_CKPT="--save-ckpt"

  echo "[run] line=${LINE} spread=${SPREAD} size_index=${SIZE_INDEX} lr=${LR}"
  python train.py \
    --config "${GRID}" \
    --line "${LINE}" \
    --train-bin "${CORPUS_ROOT}/${SPREAD}.bin" \
    --val-bin   "${CORPUS_ROOT}/val.bin" \
    --out-dir   "${RUN_ROOT}" \
    --lr "${LR}" \
    --micro-batch 16 \
    --tokens-per-step 262144 \
    --compile \
    --device cuda \
    --dtype bfloat16 \
    ${SAVE_CKPT} || { echo "FAILED line ${LINE} (continuing batch)" >&2; rc=1; }
done
exit ${rc}
