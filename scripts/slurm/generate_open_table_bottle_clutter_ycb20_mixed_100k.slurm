#!/bin/bash
#SBATCH --job-name=wm_clutter20_mixed_100k_scenes
#SBATCH --output=/common/users/shared/robot_learning_tmp/tdn39/ms-worldmodel/open_table_bottle_clutter_ycb20_mixed_100k_scenes_algo/slurm_logs/%x_%A_%a.out
#SBATCH --error=/common/users/shared/robot_learning_tmp/tdn39/ms-worldmodel/open_table_bottle_clutter_ycb20_mixed_100k_scenes_algo/slurm_logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-6%7

set -euo pipefail

REPO_ROOT="/common/users/tdn39/Robotics/ms-worldmodel"
OUTPUT_ROOT="/common/users/shared/robot_learning_tmp/tdn39/ms-worldmodel/open_table_bottle_clutter_ycb20_mixed_100k_scenes_algo"

TOTAL_SCENES=100000
SCENES_PER_SHARD=1000
BASE_SEED=100000
NUM_BOTTLES=20
LAYOUT_MODE="mixed"
PARALLEL_JOBS="${SLURM_CPUS_PER_TASK}"
TOTAL_SHARDS=$(( (TOTAL_SCENES + SCENES_PER_SHARD - 1) / SCENES_PER_SHARD ))
SHARDS_PER_BATCH="${PARALLEL_JOBS}"
BATCH_START=$(( SLURM_ARRAY_TASK_ID * SHARDS_PER_BATCH ))
BATCH_END=$(( BATCH_START + SHARDS_PER_BATCH - 1 ))

mkdir -p \
  "$OUTPUT_ROOT" \
  "$OUTPUT_ROOT/shards" \
  "$OUTPUT_ROOT/manifests" \
  "$OUTPUT_ROOT/slurm_logs"

cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

# Keep numerical libraries from oversubscribing CPU cores.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting batch ${SLURM_ARRAY_TASK_ID}"
echo "Repo root: $REPO_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Total shards: $TOTAL_SHARDS"
echo "Batch shard range: $BATCH_START-$BATCH_END"
echo "Parallel jobs: $PARALLEL_JOBS"

if [ "$BATCH_START" -ge "$TOTAL_SHARDS" ]; then
  echo "Batch start $BATCH_START is beyond total shards $TOTAL_SHARDS; exiting."
  exit 0
fi

seq "$BATCH_START" "$BATCH_END" | awk -v total="$TOTAL_SHARDS" '$1 < total' | \
  xargs -P "$PARALLEL_JOBS" -I {} bash -c '
    set -euo pipefail
    echo "Starting shard {}"
    python scripts/generate_clutter_scene_library.py \
      --output-root "'"$OUTPUT_ROOT"'" \
      --total-scenes "'"$TOTAL_SCENES"'" \
      --scenes-per-shard "'"$SCENES_PER_SHARD"'" \
      --shard-index "{}" \
      --base-seed "'"$BASE_SEED"'" \
      --num-bottles "'"$NUM_BOTTLES"'" \
      --layout-mode "'"$LAYOUT_MODE"'" \
      --compression gzip \
      --gzip-level 4 \
      --progress-every 250
    echo "Finished shard {}"
  '

echo "Finished batch ${SLURM_ARRAY_TASK_ID}"
