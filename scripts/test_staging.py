"""Run shelf push solver on multiple shelf configs.

Uses the Hydra entry point for proper video recording.

Run on GPU:
    srun --gres=gpu:a4000:1 --partition=unlimited --nodelist=rlab4 --time=01:00:00 \
        bash -c 'export CUDA_HOME=/usr/local/cuda-12.6 && export PATH=$CUDA_HOME/bin:$PATH && \
        uv run python scripts/test_staging.py'
"""
import subprocess
import os

os.makedirs("videos/shelf_push_demos", exist_ok=True)

CONFIGS = [
    ("shallow_wide", "task.shelf.front_x=0.50 task.shelf.depth=0.20 task.shelf.half_w=0.40 task.shelf.floor_z=0.30 task.shelf.inner_h=0.35"),
    ("bookshelf_low", "task.shelf.front_x=0.50 task.shelf.depth=0.30 task.shelf.half_w=0.30 task.shelf.floor_z=0.30 task.shelf.inner_h=0.35"),
    ("bookshelf_mid", "task.shelf.front_x=0.50 task.shelf.depth=0.30 task.shelf.half_w=0.30 task.shelf.floor_z=0.50 task.shelf.inner_h=0.35"),
    ("wide_low", "task.shelf.front_x=0.50 task.shelf.depth=0.30 task.shelf.half_w=0.40 task.shelf.floor_z=0.30 task.shelf.inner_h=0.35"),
    ("deep_narrow", "task.shelf.front_x=0.50 task.shelf.depth=0.35 task.shelf.half_w=0.20 task.shelf.floor_z=0.30 task.shelf.inner_h=0.35"),
]

for name, overrides in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    cmd = (
        f"uv run python -m taskbench.run "
        f"task=shelf solver=shelf_random_push_curobo "
        f"task.robot_base_pose='[0,0,0,0,0,0,1]' "
        f"task.num_objects=5 "
        f"task.cylinder.radius=0.018 task.cylinder.half_length=0.045 "
        f"{overrides} "
        f"runtime.record_video=true "
        f"runtime.video_dir=videos/shelf_push_demos/{name} "
        f"run.num_episodes=2 "
        f"seed=42"
    )
    print(f"  cmd: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout[-500:] if result.stdout else "(no stdout)")
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        print(result.stderr[-500:] if result.stderr else "")
    else:
        print(f"  OK")

print(f"\n{'='*60}")
print("Done — videos in videos/shelf_push_demos/")
