#!/bin/bash
#SBATCH --job-name=shelf_push_par
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab4
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/shelf_push_par_%j.out
#SBATCH --error=logs/shelf_push_par_%j.err

set -e
cd /common/home/dm1487/robotics_research/next_world/ms-worldmodel
mkdir -p logs videos/test_shelf_new data/success data/failure

export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
if [ ! -f "$CUDA_HOME/include/cuda_runtime.h" ] && [ -f "$CUDA_HOME/targets/x86_64-linux/include/cuda_runtime.h" ]; then
    mkdir -p $HOME/.local/cuda-include
    ln -sf $CUDA_HOME/targets/x86_64-linux/include/* $HOME/.local/cuda-include/
    export CPATH=$HOME/.local/cuda-include:$CPATH
    export C_INCLUDE_PATH=$HOME/.local/cuda-include:$C_INCLUDE_PATH
    export CPLUS_INCLUDE_PATH=$HOME/.local/cuda-include:$CPLUS_INCLUDE_PATH
fi

uv sync --extra curobo

echo "=== Running shelf push (cuRobo + UR5e) ==="
uv run python -m taskbench.run \
    task=shelf \
    solver=shelf_random_push_curobo \
    task.num_objects=5 \
    task.cylinder.radius=0.03 \
    task.cylinder.half_length=0.04 \
    runtime.video_dir=videos/test_shelf_new \
    run.num_episodes=3
