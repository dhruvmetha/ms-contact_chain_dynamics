#!/bin/bash
#SBATCH --job-name=shelf_push_v2
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab4
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/shelf_push_v2_%j.out
#SBATCH --error=logs/shelf_push_v2_%j.err

set -e
cd /common/home/dm1487/robotics_research/next_world/ms-worldmodel
mkdir -p logs videos/test_shelf_v2 data/success data/failure

# Set CUDA paths (headers are in targets/x86_64-linux/include on this cluster)
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Create a local include symlink so torch can find cuda_runtime.h
# at $CUDA_HOME/include/cuda_runtime.h
if [ ! -f "$CUDA_HOME/include/cuda_runtime.h" ] && [ -f "$CUDA_HOME/targets/x86_64-linux/include/cuda_runtime.h" ]; then
    mkdir -p $HOME/.local/cuda-include
    ln -sf $CUDA_HOME/targets/x86_64-linux/include/* $HOME/.local/cuda-include/
    export CPATH=$HOME/.local/cuda-include:$CPATH
    export C_INCLUDE_PATH=$HOME/.local/cuda-include:$C_INCLUDE_PATH
    export CPLUS_INCLUDE_PATH=$HOME/.local/cuda-include:$CPLUS_INCLUDE_PATH
fi

# Sync venv with cuRobo extra
uv sync --extra curobo

echo "=== GPU check ==="
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
uv run python -c "import curobo; print('cuRobo OK')"

echo "=== Running shelf push (cuRobo + UR5e) ==="
uv run python -m taskbench.run \
    task=shelf \
    solver=shelf_random_push_curobo \
    task.num_objects=5 \
    task.cylinder.radius=0.03 \
    task.cylinder.half_length=0.04 \
    runtime.video_dir=videos/test_shelf_v2 \
    run.num_episodes=5
