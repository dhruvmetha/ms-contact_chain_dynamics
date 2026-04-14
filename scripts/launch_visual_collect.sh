#!/usr/bin/env bash
# Launch parallel visual data collection across multiple GPUs via SLURM.
#
# Usage:
#   # 4 workers, 50 episodes each (default zarr mode)
#   bash scripts/launch_visual_collect.sh 4 50
#
#   # 8 workers, 100 episodes, custom output dir
#   bash scripts/launch_visual_collect.sh 8 100 data/shelf_big_run
#
#   # Single worker (no SLURM), useful for debugging
#   bash scripts/launch_visual_collect.sh 1 5 data/debug --local
#
# Each worker gets:
#   - Unique worker_id (0..N-1) → writes to output_dir/worker_NNN.zarr
#   - Unique seed range (base_seed + worker_id * seed_stride)
#   - Resume support: if worker zarr has existing data, continues from there
#
# With default settings (128 envs, 5 pushes, zarr mode):
#   1 episode = 128 envs × 5 pushes = 640 (s,a,s') pairs
#   50 episodes = 32,000 pairs per worker
#   4 workers = 128,000 pairs total

set -euo pipefail

NUM_WORKERS="${1:?Usage: $0 NUM_WORKERS NUM_EPISODES [OUTPUT_DIR] [--local]}"
NUM_EPISODES="${2:?Usage: $0 NUM_WORKERS NUM_EPISODES [OUTPUT_DIR] [--local]}"
TODAY=$(date +%Y%m%d)
OUTPUT_DIR="${3:-/common/users/shared/robot_learning/dm1487/wm/datasets/shelf_push_${TODAY}}"
LOCAL_MODE=false

# Check for --local flag in any position
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        LOCAL_MODE=true
    fi
done

# Tunable parameters
BASE_SEED=42
SEED_STRIDE=10000
NUM_ENVS=512
PUSHES_PER_EPISODE=5
COLLECT_MODE=zarr
PARTITION="unlimited"

# CUDA setup
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export CPATH="${CPATH:-${CUDA_HOME}/targets/x86_64-linux/include}"

PAIRS_PER_WORKER=$((NUM_EPISODES * PUSHES_PER_EPISODE * NUM_ENVS))
TOTAL_PAIRS=$((NUM_WORKERS * PAIRS_PER_WORKER))

echo "=== Visual Data Collection ==="
echo "  Workers:           ${NUM_WORKERS}"
echo "  Episodes/worker:   ${NUM_EPISODES}"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "  Envs/worker:       ${NUM_ENVS}"
echo "  Pushes/episode:    ${PUSHES_PER_EPISODE}"
echo "  Collect mode:      ${COLLECT_MODE}"
echo "  Pairs/worker:      ${PAIRS_PER_WORKER}"
echo "  Total pairs:       ${TOTAL_PAIRS}"
echo "  Mode:              $(if $LOCAL_MODE; then echo 'local'; else echo 'SLURM'; fi)"
echo ""

mkdir -p logs

PIDS=()

for ((i=0; i<NUM_WORKERS; i++)); do
    WORKER_SEED=$((BASE_SEED + i * SEED_STRIDE))

    CMD="uv run python -m taskbench.run \
        task=shelf_panda_stick \
        solver=shelf_visual_collect \
        seed=${WORKER_SEED} \
        run.num_episodes=${NUM_EPISODES} \
        run.solver_kwargs.pushes_per_episode=${PUSHES_PER_EPISODE} \
        run.solver_kwargs.output_dir=${OUTPUT_DIR} \
        run.solver_kwargs.worker_id=${i} \
        run.solver_kwargs.collect_mode=${COLLECT_MODE} \
        runtime.num_envs=${NUM_ENVS}"

    if $LOCAL_MODE; then
        echo "[Worker ${i}] seed=${WORKER_SEED} (local)"
        eval "$CMD" &
        PIDS+=($!)
    else
        echo "[Worker ${i}] seed=${WORKER_SEED} → srun job"
        srun --partition="${PARTITION}" \
             --gres=gpu:1 \
             --job-name="vcollect_${i}" \
             --output="logs/vcollect_worker${i}_%j.out" \
             --error="logs/vcollect_worker${i}_%j.err" \
             bash -c "export CUDA_HOME=${CUDA_HOME} && export CPATH=${CPATH} && ${CMD}" &
        PIDS+=($!)
    fi
done

echo ""
echo "Launched ${NUM_WORKERS} workers. PIDs: ${PIDS[*]}"
echo "Logs: logs/vcollect_worker*"
echo ""
echo "Monitor progress:"
echo "  watch -n10 'for f in ${OUTPUT_DIR}/worker_*.zarr; do python -c \"import zarr; s=zarr.open(\\\"\$f\\\",mode=\\\"r\\\"); print(f\\\"\$f: {s[\\\"rgb_before\\\"].shape[0]} samples\\\")\" 2>/dev/null; done'"
echo ""
echo "Waiting for all workers to finish..."
wait "${PIDS[@]}"
echo "Done. All workers finished."
