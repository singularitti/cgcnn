#!/bin/bash
#SBATCH -J fecos_mptrj_fe
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /scratch/04996/tg842951/fecos_mptrj_formation_energy_%j.out
#SBATCH -e /scratch/04996/tg842951/fecos_mptrj_formation_energy_%j.err

set -euo pipefail

module reset
module load python/3.12.11
module list

REPO_DIR="/work2/04996/tg842951/stampede3/cgcnn"
RUN_NAME="${RUN_NAME:?RUN_NAME must be set before sbatch}"
INITIAL_TARGET_MODE="${INITIAL_TARGET_MODE:-benchmark-predicted}"
CONSTANT_INITIAL_VALUE="${CONSTANT_INITIAL_VALUE:-0.0}"
MPTRJ_BASELINE_RUN_DIR="${MPTRJ_BASELINE_RUN_DIR:-}"
SCRATCH_OUTPUT_DIR="/scratch/04996/tg842951/cgcnn_runs/${RUN_NAME}"
WORK_OUTPUT_DIR="/work2/04996/tg842951/stampede3/run/FeCoS_paper/${RUN_NAME}"
EXTRA_ARGS=(
  --initial-target-mode "${INITIAL_TARGET_MODE}"
  --constant-initial-value "${CONSTANT_INITIAL_VALUE}"
)
if [[ -n "${MPTRJ_BASELINE_RUN_DIR}" ]]; then
  EXTRA_ARGS+=(--mptrj-baseline-run-dir "${MPTRJ_BASELINE_RUN_DIR}")
fi

cd "${REPO_DIR}"

echo "Starting FeCoS MPTrj CGCNN prediction"
echo "Run name: ${RUN_NAME}"
echo "Scratch output: ${SCRATCH_OUTPUT_DIR}"
echo "Work output: ${WORK_OUTPUT_DIR}"
echo "Initial target mode: ${INITIAL_TARGET_MODE}"
echo "Constant initial value: ${CONSTANT_INITIAL_VALUE}"
echo "MPTrj baseline run: ${MPTRJ_BASELINE_RUN_DIR}"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Timestamp: $(date -Is)"

uv run python tools/run_fecos_mptrj_formation_energy_prediction.py \
  --source-root "/work2/04996/tg842951/stampede3/run/FeCoS_paper/formation energy & magnetization predictions using CGCNN" \
  --benchmark-csv "/work2/04996/tg842951/stampede3/run/FeCoS_paper/cgcnn_fecos_formation_energy_20260421_051704/fecos_formation_energy_vs_prediction.csv" \
  --model-path "/work2/04996/tg842951/stampede3/run/mptrj_ef_per_atom_20260705_044309_results/checkpoints/best_validation.pth.tar" \
  --atom-init "/work2/04996/tg842951/stampede3/cgcnn/data/sample-regression/atom_init.json" \
  --previous-run-dir "/work2/04996/tg842951/stampede3/run/FeCoS_paper/cgcnn_fecos_formation_energy_20260421_051704" \
  --scratch-output-dir "${SCRATCH_OUTPUT_DIR}" \
  --work-output-dir "${WORK_OUTPUT_DIR}" \
  --repo-dir "${REPO_DIR}" \
  --batch-size 128 \
  --workers 0 \
  --expected-count 1314 \
  --device cpu \
  "${EXTRA_ARGS[@]}"

mkdir -p "${SCRATCH_OUTPUT_DIR}/logs"
cp "/scratch/04996/tg842951/fecos_mptrj_formation_energy_${SLURM_JOB_ID}.out" "${SCRATCH_OUTPUT_DIR}/logs/slurm-${SLURM_JOB_ID}.out"
cp "/scratch/04996/tg842951/fecos_mptrj_formation_energy_${SLURM_JOB_ID}.err" "${SCRATCH_OUTPUT_DIR}/logs/slurm-${SLURM_JOB_ID}.err"
cp "/scratch/04996/tg842951/fecos_mptrj_formation_energy_${SLURM_JOB_ID}.out" "${WORK_OUTPUT_DIR}/logs/slurm-${SLURM_JOB_ID}.out"
cp "/scratch/04996/tg842951/fecos_mptrj_formation_energy_${SLURM_JOB_ID}.err" "${WORK_OUTPUT_DIR}/logs/slurm-${SLURM_JOB_ID}.err"

echo "Finished at $(date -Is)"
