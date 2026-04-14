#!/usr/bin/env bash
# Launch data collection with GPU-appropriate env counts.
# Tries best GPUs first, falls back to smaller ones.
#
# Usage:
#   bash scripts/launch_multi_gpu.sh [NUM_EPISODES] [OUTPUT_DIR]
#   bash scripts/launch_multi_gpu.sh 50

set -euo pipefail

NUM_EPISODES="${1:-50}"
TODAY=$(date +%Y%m%d)
OUTPUT_DIR="${2:-/common/users/shared/robot_learning/dm1487/wm/datasets/shelf_push_${TODAY}}"
PUSHES_PER_EPISODE=5
COLLECT_MODE=zarr
PARTITION="unlimited"
BASE_SEED=42
SEED_STRIDE=10000

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export CPATH="${CPATH:-${CUDA_HOME}/targets/x86_64-linux/include}"

mkdir -p logs

# GPU tiers: (gpu_type, node, num_envs, safe with cuRobo)
# Tier 1: Best GPUs first
# Tier 2: Medium
# Tier 3: Fallback
# Note: rlab6 (A5000) has no CUDA 12.6, skip it
WORKERS=(
    # Tier 1: A6000 48GB
    "0  a4500    ilab1  128"
    "1  a4500    ilab1  128"
    "2  a4500    ilab2  128"
    "3  a4500    ilab2  128"
    "4  a4000    ilab4  128"
    "5  a4000    ilab4  128"
    "6  a4000    rlab2  128"
    "7  a4000    rlab2  128"
    "8  a4000    rlab3  128"
    "9  a4000    rlab4  128"
    "10 a4000    rlab4  128"
)

TOTAL_PAIRS=0
PIDS=()

echo "=== Visual Data Collection (multi-GPU) ==="
echo "  Episodes/worker: ${NUM_EPISODES}"
echo "  Pushes/episode:  ${PUSHES_PER_EPISODE}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo ""
printf "  %-4s %-10s %-8s %-6s %-10s\n" "ID" "GPU" "Node" "Envs" "Pairs"
echo "  -----------------------------------------------"

for entry in "${WORKERS[@]}"; do
    read -r WID GPU NODE NENVS <<< "$entry"
    PAIRS=$((NUM_EPISODES * PUSHES_PER_EPISODE * NENVS))
    TOTAL_PAIRS=$((TOTAL_PAIRS + PAIRS))
    WORKER_SEED=$((BASE_SEED + WID * SEED_STRIDE))

    printf "  %-4s %-10s %-8s %-6s %-10s\n" "$WID" "$GPU" "$NODE" "$NENVS" "$PAIRS"

    CMD="uv run python -m taskbench.run \
        task=shelf_panda_stick \
        solver=shelf_visual_collect \
        seed=${WORKER_SEED} \
        run.num_episodes=${NUM_EPISODES} \
        run.solver_kwargs.pushes_per_episode=${PUSHES_PER_EPISODE} \
        run.solver_kwargs.output_dir=${OUTPUT_DIR} \
        run.solver_kwargs.worker_id=${WID} \
        run.solver_kwargs.collect_mode=${COLLECT_MODE} \
        runtime.num_envs=${NENVS}"

    srun --partition="${PARTITION}" \
         --gres=gpu:${GPU}:1 \
         --nodelist="${NODE}" \
         --job-name="vc_${WID}" \
         --output="logs/vc_${WID}_%j.out" \
         --error="logs/vc_${WID}_%j.err" \
         bash -c "export CUDA_HOME=${CUDA_HOME} && export CPATH=${CPATH} && ${CMD}" &
    PIDS+=($!)
done

echo ""
echo "  Total pairs:     ${TOTAL_PAIRS}"
echo ""
echo "Launched ${#PIDS[@]} workers."
echo "Logs: logs/vc_*"
echo ""
echo "Monitor:"
echo "  squeue -u $USER"
echo "  uv run python -c \"import zarr,os;base='${OUTPUT_DIR}';[print(f'  {f}: {zarr.open(os.path.join(base,f),mode=\\\"r\\\")[\\\"rgb_before\\\"].shape[0]}') for f in sorted(os.listdir(base)) if f.endswith('.zarr')]\""
echo ""
echo "Waiting for all workers to finish..."
wait "${PIDS[@]}"
echo "Done."
