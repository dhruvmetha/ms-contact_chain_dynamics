"""Extract sample frames from visual trajectory dataset for inspection.

Usage:
    uv run python scripts/extract_visual_frames.py <dataset_dir> [--traj 000000] [--frame 0]

Saves PNG images to <dataset_dir>/preview/ showing all cameras side-by-side.
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taskbench.trajectory_dataset import ManiSkillTrajectoryDataset


def extract_frames(dataset_dir, traj_id="000000", frame_idx=0):
    dataset = ManiSkillTrajectoryDataset(dataset_dir)
    trajs = dataset.list_trajectories()
    print(f"Dataset has {len(trajs)} trajectories: {trajs[:10]}...")

    if traj_id not in trajs:
        traj_id = trajs[0]
        print(f"Using first trajectory: {traj_id}")

    keys = dataset.list_keys(traj_id)
    print(f"Keys for traj {traj_id}: {keys}")

    # Read the trajectory
    traj = dataset.read_trajectory(traj_id)
    print(f"Success: {traj.success}")
    print(f"Video streams: {list(traj.video_streams.keys())}")
    print(f"Metadata keys: {list(traj.metadata.keys())}")

    # Find camera names from RGB keys
    cam_names = [k.replace("_rgb", "") for k in traj.video_streams if k.endswith("_rgb")]
    print(f"Cameras: {cam_names}")

    if not cam_names:
        print("No RGB streams found!")
        return

    # Get frame count
    first_rgb = traj.video_streams[f"{cam_names[0]}_rgb"]
    n_frames = first_rgb.shape[0]
    print(f"Total frames: {n_frames}, resolution: {first_rgb.shape[1]}x{first_rgb.shape[2]}")

    frame_idx = min(frame_idx, n_frames - 1)

    # Save individual camera images and a combined grid
    preview_dir = os.path.join(dataset_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)

    try:
        import cv2
    except ImportError:
        print("cv2 not available, saving raw numpy arrays")
        return

    # Save individual frames per camera
    for cam in cam_names:
        rgb_key = f"{cam}_rgb"
        if rgb_key in traj.video_streams:
            rgb = traj.video_streams[rgb_key][frame_idx]  # (H, W, 3) uint8
            path = os.path.join(preview_dir, f"{cam}_frame{frame_idx:03d}_rgb.png")
            cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"  Saved {path}")

        depth_key = f"{cam}_depth"
        if depth_key in traj.video_streams:
            depth = traj.video_streams[depth_key][frame_idx]  # (H, W) float32 (int16 mm values)
            # Normalize for visualization
            d_valid = depth[depth > 0]
            if len(d_valid) > 0:
                d_min, d_max = d_valid.min(), d_valid.max()
                depth_vis = np.clip((depth - d_min) / max(d_max - d_min, 1), 0, 1)
                depth_vis = (depth_vis * 255).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
            else:
                depth_vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
            path = os.path.join(preview_dir, f"{cam}_frame{frame_idx:03d}_depth.png")
            cv2.imwrite(path, depth_vis)
            print(f"  Saved {path}")

    # Create combined grid: cameras side by side
    row_rgb = []
    row_depth = []
    for cam in cam_names:
        rgb_key = f"{cam}_rgb"
        if rgb_key in traj.video_streams:
            row_rgb.append(traj.video_streams[rgb_key][frame_idx])

        depth_key = f"{cam}_depth"
        if depth_key in traj.video_streams:
            depth = traj.video_streams[depth_key][frame_idx]
            d_valid = depth[depth > 0]
            if len(d_valid) > 0:
                d_min, d_max = d_valid.min(), d_valid.max()
                depth_vis = np.clip((depth - d_min) / max(d_max - d_min, 1), 0, 1)
                depth_vis = (depth_vis * 255).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
            else:
                depth_vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
            row_depth.append(depth_vis)

    if row_rgb:
        grid_rgb = np.concatenate(row_rgb, axis=1)
        path = os.path.join(preview_dir, f"all_cameras_frame{frame_idx:03d}_rgb.png")
        cv2.imwrite(path, cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR))
        print(f"\n  Combined RGB grid: {path}")

    if row_depth:
        grid_depth = np.concatenate(row_depth, axis=1)
        path = os.path.join(preview_dir, f"all_cameras_frame{frame_idx:03d}_depth.png")
        cv2.imwrite(path, cv2.cvtColor(grid_depth, cv2.COLOR_RGB2BGR))
        print(f"  Combined depth grid: {path}")

    # Also extract a few frames across the trajectory for a temporal montage
    sample_frames = np.linspace(0, n_frames - 1, min(8, n_frames), dtype=int)
    for cam in cam_names[:1]:  # just first camera for temporal montage
        temporal = []
        for fi in sample_frames:
            rgb = traj.video_streams[f"{cam}_rgb"][fi]
            temporal.append(rgb)
        if temporal:
            montage = np.concatenate(temporal, axis=1)
            path = os.path.join(preview_dir, f"{cam}_temporal_montage.png")
            cv2.imwrite(path, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
            print(f"  Temporal montage ({cam}, {len(sample_frames)} frames): {path}")

    print(f"\nAll previews saved to: {preview_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Auto-detect most recent visual dataset
        import glob
        dirs = sorted(glob.glob("data/visual_trajectories/*"))
        if dirs:
            dataset_dir = dirs[-1]
            print(f"Auto-detected: {dataset_dir}")
        else:
            print("Usage: python scripts/extract_visual_frames.py <dataset_dir>")
            sys.exit(1)
    else:
        dataset_dir = sys.argv[1]

    traj_id = "000000"
    frame_idx = 0
    for arg in sys.argv[2:]:
        if arg.startswith("--traj"):
            traj_id = sys.argv[sys.argv.index(arg) + 1]
        elif arg.startswith("--frame"):
            frame_idx = int(sys.argv[sys.argv.index(arg) + 1])

    extract_frames(dataset_dir, traj_id, frame_idx)
