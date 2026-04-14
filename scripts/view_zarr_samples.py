"""Interactive viewer for zarr push dataset.

Saves a grid of samples showing: before | before+action_mask | after | depth

Usage:
    # View 8 random samples
    uv run python scripts/view_zarr_samples.py data/visual_trajectories/worker_099.zarr

    # View specific sample indices
    uv run python scripts/view_zarr_samples.py data/visual_trajectories/worker_099.zarr --indices 0 5 10 15

    # View N random samples
    uv run python scripts/view_zarr_samples.py data/visual_trajectories/worker_099.zarr --n 16

    # Change number of action dots
    uv run python scripts/view_zarr_samples.py data/visual_trajectories/worker_099.zarr --n_dots 15

    # Save individual sample images
    uv run python scripts/view_zarr_samples.py data/visual_trajectories/worker_099.zarr --save_individual --out_dir /tmp/samples
"""

import argparse
import os

import cv2
import numpy as np
import zarr


def project(pt3d, K, E):
    """Project a 3D point to 2D pixel coordinates."""
    p = E @ np.append(pt3d, 1.0)
    px = K @ p
    return int(round(px[0] / px[2])), int(round(px[1] / px[2]))


def make_action_mask(p1, p2, intrinsics, extrinsics, H, W, n_dots=10):
    """Render action as dots along the p1->p2 sweep line."""
    mask = np.zeros((H, W), dtype=np.uint8)
    dot_radius = max(2, H // 50)
    for i in range(n_dots):
        t = i / max(1, n_dots - 1)
        pt = p1 * (1 - t) + p2 * t
        x, y = project(pt, intrinsics, extrinsics)
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(mask, (x, y), dot_radius, 255, -1)
    return mask


def render_sample(store, idx, n_dots=10, size=256):
    """Render a single sample as a dict of images."""
    rgb_b = store["rgb_before"][idx]
    rgb_a = store["rgb_after"][idx]
    depth_b = store["depth_before"][idx]
    depth_a = store["depth_after"][idx]
    p1 = store["entry_position"][idx]
    p2 = store["sweep_position"][idx]
    intr = store["intrinsics"][idx]
    extr = store["extrinsics"][idx]

    H, W = rgb_b.shape[:2]

    # Action mask
    action_mask = make_action_mask(p1, p2, intr, extr, H, W, n_dots=n_dots)

    # Before + action overlay
    before_action = rgb_b.copy()
    # Green dots for action
    before_action[action_mask > 0] = [0, 255, 0]
    # Red circle at start, green at end
    px1 = project(p1, intr, extr)
    px2 = project(p2, intr, extr)
    cv2.circle(before_action, px1, max(3, H // 40), (255, 50, 50), -1)
    cv2.circle(before_action, px2, max(3, H // 40), (50, 255, 50), -1)

    # Depth colormaps
    def depth_to_color(d):
        valid = d[d > 0]
        if len(valid) == 0:
            return np.zeros((*d.shape, 3), dtype=np.uint8)
        lo, hi = valid.min(), valid.max()
        norm = ((d - lo) / (hi - lo + 1e-6) * 255).clip(0, 255).astype(np.uint8)
        norm[d <= 0] = 0
        return cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)

    depth_b_color = depth_to_color(depth_b)
    depth_a_color = depth_to_color(depth_a)

    # Resize all to target size
    def up(img):
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)

    return {
        "before": up(rgb_b),
        "action": up(before_action),
        "action_mask": up(np.stack([action_mask] * 3, axis=-1)),
        "after": up(rgb_a),
        "depth_before": cv2.cvtColor(up(depth_b_color), cv2.COLOR_BGR2RGB),
        "depth_after": cv2.cvtColor(up(depth_a_color), cv2.COLOR_BGR2RGB),
        "p1": p1,
        "p2": p2,
    }


def main():
    parser = argparse.ArgumentParser(description="View zarr push dataset samples")
    parser.add_argument("zarr_path", help="Path to .zarr dataset")
    parser.add_argument("--indices", nargs="+", type=int, default=None,
                        help="Specific sample indices to view")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of random samples (ignored if --indices given)")
    parser.add_argument("--n_dots", type=int, default=10,
                        help="Number of dots in action mask")
    parser.add_argument("--size", type=int, default=256,
                        help="Display size per image")
    parser.add_argument("--out", type=str, default="/tmp/zarr_viewer.png",
                        help="Output path for grid image")
    parser.add_argument("--save_individual", action="store_true",
                        help="Save each sample as separate image")
    parser.add_argument("--out_dir", type=str, default="/tmp/zarr_samples",
                        help="Directory for individual sample images")
    args = parser.parse_args()

    store = zarr.open(args.zarr_path, mode="r")
    total = store["rgb_before"].shape[0]
    H, W = store["rgb_before"].shape[1], store["rgb_before"].shape[2]

    print(f"Dataset: {args.zarr_path}")
    print(f"  Samples: {total}")
    print(f"  Resolution: {H}x{W}")
    print(f"  Keys: {list(store.keys())}")
    print()

    # Select indices
    if args.indices is not None:
        indices = [i for i in args.indices if i < total]
    else:
        indices = sorted(np.random.default_rng(42).choice(total, size=min(args.n, total), replace=False))

    print(f"Viewing samples: {indices}")
    print(f"Columns: before | before+action | action_mask | after | depth_before | depth_after")
    print()

    rows = []
    for idx in indices:
        s = render_sample(store, idx, n_dots=args.n_dots, size=args.size)
        print(f"  [{idx:>5d}] p1=[{s['p1'][0]:.3f},{s['p1'][1]:.3f},{s['p1'][2]:.3f}] "
              f"-> p2=[{s['p2'][0]:.3f},{s['p2'][1]:.3f},{s['p2'][2]:.3f}]")

        row = np.concatenate([
            cv2.cvtColor(s["before"], cv2.COLOR_RGB2BGR),
            cv2.cvtColor(s["action"], cv2.COLOR_RGB2BGR),
            cv2.cvtColor(s["action_mask"], cv2.COLOR_RGB2BGR),
            cv2.cvtColor(s["after"], cv2.COLOR_RGB2BGR),
            cv2.cvtColor(s["depth_before"], cv2.COLOR_RGB2BGR),
            cv2.cvtColor(s["depth_after"], cv2.COLOR_RGB2BGR),
        ], axis=1)
        rows.append(row)

        if args.save_individual:
            os.makedirs(args.out_dir, exist_ok=True)
            for name, img in s.items():
                if isinstance(img, np.ndarray) and img.ndim == 3:
                    path = os.path.join(args.out_dir, f"sample_{idx:05d}_{name}.png")
                    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(args.out, grid)
    print(f"\nSaved grid: {args.out} ({grid.shape[1]}x{grid.shape[0]})")

    if args.save_individual:
        print(f"Saved individual images: {args.out_dir}/")


if __name__ == "__main__":
    main()
