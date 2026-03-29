"""Fit tight collision spheres for UR5e + 2F-85 from URDF collision meshes.

Uses cuRobo's sphere_fit to voxelize each link's collision mesh and
generate spheres that closely match the actual geometry.

Run on GPU:
    srun --gres=gpu:a4000:1 --partition=unlimited --nodelist=rlab4 --time=00:15:00 --mem=32G \
        bash -c 'export CUDA_HOME=/usr/local/cuda-12.6 && export PATH=$CUDA_HOME/bin:$PATH && \
        uv run python scripts/fit_collision_spheres.py'
"""
import trimesh
import numpy as np
import yaml
import os
from pathlib import Path
from curobo.geom.sphere_fit import SphereFitType, voxel_fit_mesh, sample_even_fit_mesh

URDF_DIR = Path("data/robots/ur5e_robotiq_2f_85")

# Links to fit spheres for (arm only — gripper spheres are disabled)
ARM_LINKS = {
    "shoulder_link": {
        "mesh": "meshes/ur5e/collision/shoulder.stl",
        "n_spheres": 8,
        "radius": 0.02,
    },
    "upper_arm_link": {
        "mesh": "meshes/ur5e/collision/upperarm.stl",
        "n_spheres": 12,
        "radius": 0.02,
    },
    "forearm_link": {
        "mesh": "meshes/ur5e/collision/forearm.stl",
        "n_spheres": 12,
        "radius": 0.02,
    },
    "wrist_1_link": {
        "mesh": "meshes/ur5e/collision/wrist1.stl",
        "n_spheres": 6,
        "radius": 0.015,
    },
    "wrist_2_link": {
        "mesh": "meshes/ur5e/collision/wrist2.stl",
        "n_spheres": 6,
        "radius": 0.015,
    },
    "wrist_3_link": {
        "mesh": "meshes/ur5e/collision/wrist3.stl",
        "n_spheres": 4,
        "radius": 0.015,
    },
}

print("Fitting collision spheres from URDF meshes\n")

all_spheres = {}

for link_name, cfg in ARM_LINKS.items():
    mesh_path = URDF_DIR / cfg["mesh"]
    if not mesh_path.exists():
        print(f"  {link_name}: MESH NOT FOUND at {mesh_path}")
        continue

    mesh = trimesh.load(str(mesh_path))
    print(f"{link_name}:")
    print(f"  mesh: {mesh_path.name} ({len(mesh.vertices)} verts, "
          f"extents={[round(x,4) for x in mesh.extents]})")

    # Fit spheres using voxelization
    try:
        pts, radii = voxel_fit_mesh(
            mesh,
            n_spheres=cfg["n_spheres"],
            surface_sphere_radius=cfg["radius"],
            voxelize_method="ray",
        )
    except Exception as e:
        print(f"  voxel_fit failed ({e}), trying surface sampling")
        pts, radii = sample_even_fit_mesh(mesh, cfg["n_spheres"], cfg["radius"])

    if pts is None or len(pts) == 0:
        print(f"  FAILED to fit spheres")
        continue

    spheres = []
    for i in range(len(pts)):
        center = [round(float(pts[i][0]), 4),
                   round(float(pts[i][1]), 4),
                   round(float(pts[i][2]), 4)]
        radius = round(float(radii[i]), 4)
        spheres.append({"center": center, "radius": radius})
        print(f"  sphere {i}: center={center} radius={radius}")

    all_spheres[link_name] = spheres
    print(f"  -> {len(spheres)} spheres fitted\n")

# Compare with current (2F-140) sphere radii
print("\n=== Comparison: old (2F-140) vs new (fitted) ===")
old_max_radii = {
    "shoulder_link": 0.10,
    "upper_arm_link": 0.08,
    "forearm_link": 0.072,
    "wrist_1_link": 0.047,
    "wrist_2_link": 0.047,
    "wrist_3_link": 0.043,
}
for link, spheres in all_spheres.items():
    old_r = old_max_radii.get(link, 0)
    new_max_r = max(s["radius"] for s in spheres) if spheres else 0
    print(f"  {link:20s}: old max={old_r:.3f}  new max={new_max_r:.3f}  "
          f"reduction={1-new_max_r/old_r:.0%}" if old_r > 0 else "")

# Output YAML format for cuRobo config
print("\n\n=== YAML for cuRobo config (copy to ur5e_robotiq_2f_85.yml) ===")
print("    collision_spheres:")
for link, spheres in all_spheres.items():
    print(f"      {link}:")
    for s in spheres:
        print(f'        - "center": {s["center"]}')
        print(f'          "radius": {s["radius"]}')

print("\nDone")
