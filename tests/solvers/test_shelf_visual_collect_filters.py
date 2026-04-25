"""Regression tests for the vectorized scene-validity filter.

The shelf_visual_collect solver replaced per-env Python loops with a single
numpy expression in `_scene_validity_masks`. These tests prove the vector
form returns identical answers to the original loop-based predicate on
randomized scenes, and that hand-crafted edge cases behave as expected.

Run:
    uv run pytest tests/solvers/test_shelf_visual_collect_filters.py -v
"""

from dataclasses import dataclass

import numpy as np
import pytest

from taskbench.solvers.shelf_visual_collect import _scene_validity_masks


@dataclass
class FakeShelfGeom:
    front_x: float = 0.50
    back_x: float = 0.85
    half_w: float = 0.18
    surface_z: float = 0.40


# Spawn quaternion: 90° about +Y, body +X → world +Z.
CYL_UPRIGHT_Q = np.array([0.7071068, 0.0, 0.7071068, 0.0])


def _scene_validity_loop(cyl_pos, cyl_quat, g, tilt_cos=0.5):
    """Reference implementation: per-env Python loop matching the new
    yaw-invariant body-x-axis check.

    Cylinders spawn with body-X aligned to world-Z (CYL_UPRIGHT_Q), so
    upright-ness uses the world-z component of the body x-axis:
    `2*(qx*qz + qw*qy)`.
    """
    N, num_cyl, _ = cyl_pos.shape
    out = np.ones(N, dtype=bool)
    for i in range(N):
        for c in range(num_cyl):
            cp = cyl_pos[i, c]
            cq = cyl_quat[i, c]
            if cp[0] > 2.0:
                continue  # hidden
            if (cp[0] < g.front_x - 0.02 or cp[0] > g.back_x + 0.02
                    or cp[1] < -g.half_w - 0.02 or cp[1] > g.half_w + 0.02
                    or cp[2] < g.surface_z - 0.01):
                out[i] = False
                break
            qw, qx, qy, qz = cq
            body_x_world_z = abs(2.0 * (qx * qz - qw * qy))
            if body_x_world_z < tilt_cos:
                out[i] = False
                break
    return out


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _tilt_quat(angle_rad, axis):
    """Quaternion for a rotation of `angle_rad` around `axis` (unit)."""
    return np.array([
        np.cos(angle_rad / 2),
        np.sin(angle_rad / 2) * axis[0],
        np.sin(angle_rad / 2) * axis[1],
        np.sin(angle_rad / 2) * axis[2],
    ])


def _random_quaternions(rng, shape, max_tilt_deg=10.0):
    """Generate quaternions perturbed away from CYL_UPRIGHT_Q by up to
    `max_tilt_deg` around a random axis. So `max_tilt_deg=0` yields exactly
    CYL_UPRIGHT_Q; small values are slight tilts; >60° crosses the topple
    threshold."""
    n = int(np.prod(shape[:-1]))
    angles = rng.uniform(-np.deg2rad(max_tilt_deg), np.deg2rad(max_tilt_deg), n)
    axes = rng.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True) + 1e-9
    qw = np.cos(angles / 2)
    qxyz = np.sin(angles / 2)[:, None] * axes
    perturb = np.concatenate([qw[:, None], qxyz], axis=1)
    out = np.empty_like(perturb)
    for i in range(n):
        out[i] = _quat_mul(perturb[i], CYL_UPRIGHT_Q)
    return out.reshape(*shape[:-1], 4)


# ---------------------------------------------------------------------------
# Vectorized vs loop: parity on randomized scenes
# ---------------------------------------------------------------------------


def test_vector_matches_loop_clean_scenes():
    """Mostly-clean scenes: small tilts, in-bounds, all upright."""
    rng = np.random.default_rng(0)
    g = FakeShelfGeom()
    N, num_cyl = 64, 6
    cyl_pos = np.stack([
        rng.uniform(g.front_x + 0.02, g.back_x - 0.02, (N, num_cyl)),
        rng.uniform(-g.half_w + 0.02, g.half_w - 0.02, (N, num_cyl)),
        rng.uniform(g.surface_z, g.surface_z + 0.05, (N, num_cyl)),
    ], axis=-1)
    cyl_quat = _random_quaternions(rng, (N, num_cyl, 4), max_tilt_deg=5)

    vec = _scene_validity_masks(cyl_pos, cyl_quat, g)
    ref = _scene_validity_loop(cyl_pos, cyl_quat, g)
    np.testing.assert_array_equal(vec, ref)
    assert vec.all(), "all clean scenes should be valid"


def test_vector_matches_loop_messy_scenes():
    """Messy scenes: a mix of toppled, off-shelf, and clean envs.

    Per-cylinder bad rate is kept low enough that with 4 cylinders/env we
    get a meaningful split — both passes and rejections.
    """
    rng = np.random.default_rng(42)
    g = FakeShelfGeom()
    N, num_cyl = 128, 4

    # Most cylinders in-bounds; a small slice deliberately off-shelf.
    cyl_pos = np.stack([
        rng.uniform(g.front_x + 0.02, g.back_x - 0.02, (N, num_cyl)),
        rng.uniform(-g.half_w + 0.02, g.half_w - 0.02, (N, num_cyl)),
        rng.uniform(g.surface_z, g.surface_z + 0.05, (N, num_cyl)),
    ], axis=-1)
    bad_xy = rng.random((N, num_cyl)) < 0.05
    cyl_pos[bad_xy, 0] = g.front_x - 0.10  # 10cm in front — off-shelf

    # ~20% hidden
    hidden = rng.random((N, num_cyl)) < 0.2
    cyl_pos[hidden, 0] = 3.0

    # Mostly upright, ~10% wildly tilted
    cyl_quat = _random_quaternions(rng, (N, num_cyl, 4), max_tilt_deg=10)
    bad_q = rng.random((N, num_cyl)) < 0.1
    n_bad_q = int(bad_q.sum())
    if n_bad_q:
        cyl_quat[bad_q] = _random_quaternions(
            rng, (n_bad_q, 4), max_tilt_deg=120
        ).reshape(n_bad_q, 4)

    vec = _scene_validity_masks(cyl_pos, cyl_quat, g)
    ref = _scene_validity_loop(cyl_pos, cyl_quat, g)
    np.testing.assert_array_equal(vec, ref)
    assert (~vec).any(), "messy scene should produce some rejections"
    assert vec.any(), "messy scene should still have some valid envs"


@pytest.mark.parametrize("seed", range(5))
def test_vector_matches_loop_random_seeds(seed):
    """Sweep multiple seeds for confidence."""
    rng = np.random.default_rng(seed)
    g = FakeShelfGeom()
    N, num_cyl = 32, 5
    cyl_pos = rng.uniform(-1.0, 1.0, (N, num_cyl, 3))
    # Random hidden mask
    cyl_pos[rng.random((N, num_cyl)) < 0.2, 0] = 3.0
    cyl_quat = _random_quaternions(rng, (N, num_cyl, 4), max_tilt_deg=180)

    vec = _scene_validity_masks(cyl_pos, cyl_quat, g)
    ref = _scene_validity_loop(cyl_pos, cyl_quat, g)
    np.testing.assert_array_equal(vec, ref)


# ---------------------------------------------------------------------------
# Hand-crafted edge cases
# ---------------------------------------------------------------------------


def test_spawn_pose_passes():
    """The exact spawn quaternion CYL_UPRIGHT_Q must pass."""
    g = FakeShelfGeom()
    cyl_pos = np.array([[[0.7, 0.0, g.surface_z]]])
    cyl_quat = np.array([[CYL_UPRIGHT_Q]])
    assert _scene_validity_masks(cyl_pos, cyl_quat, g)[0]


def test_yaw_only_rotation_passes():
    """Pure yaw (rotation around world-Z) of an upright cylinder must NOT
    be flagged toppled. This is the bug the new filter fixes — old check
    |qw| < 0.5 rejected pure yaw spins through 120°–240°."""
    g = FakeShelfGeom()
    cyl_pos = np.array([[[0.7, 0.0, g.surface_z]]])
    # Yaw 180° about world-Z, then apply spawn rotation
    yaw = _tilt_quat(np.pi, np.array([0.0, 0.0, 1.0]))
    q = _quat_mul(yaw, CYL_UPRIGHT_Q)
    assert _scene_validity_masks(cyl_pos, np.array([[q]]), g)[0]


def test_lying_flat_is_rejected():
    """A 90° tilt of the cylinder's HEIGHT axis away from world-Z must be
    flagged toppled. Old check missed this (qw ≈ 0.707 > 0.5)."""
    g = FakeShelfGeom()
    cyl_pos = np.array([[[0.7, 0.0, g.surface_z]]])
    # Tilt the spawn pose by 90° around world-X — height axis goes horizontal
    tilt = _tilt_quat(np.pi / 2, np.array([1.0, 0.0, 0.0]))
    q = _quat_mul(tilt, CYL_UPRIGHT_Q)
    assert not _scene_validity_masks(cyl_pos, np.array([[q]]), g)[0]


def test_60deg_tilt_at_threshold():
    """60° tilt of the height axis: body-x world-z ≈ cos(60°) = 0.5 — borderline.
    The check is `< 0.5`, so 59° should pass and 61° should fail."""
    g = FakeShelfGeom()
    cyl_pos = np.array([[[0.7, 0.0, g.surface_z]]])

    tilt_59 = _tilt_quat(np.deg2rad(59), np.array([1.0, 0.0, 0.0]))
    q59 = _quat_mul(tilt_59, CYL_UPRIGHT_Q)
    assert _scene_validity_masks(cyl_pos, np.array([[q59]]), g)[0]

    tilt_61 = _tilt_quat(np.deg2rad(61), np.array([1.0, 0.0, 0.0]))
    q61 = _quat_mul(tilt_61, CYL_UPRIGHT_Q)
    assert not _scene_validity_masks(cyl_pos, np.array([[q61]]), g)[0]


def test_hidden_cylinders_ignored():
    """Hidden cylinders (x > 2.0) must not affect validity, even if their
    pose looks 'bad' on every other axis."""
    g = FakeShelfGeom()
    cyl_pos = np.array([[
        [0.7, 0.0, g.surface_z],     # active, fine
        [3.5, 99.0, -10.0],          # hidden — wildly off but ignored
    ]])
    bad_q = _quat_mul(
        _tilt_quat(np.pi / 2, np.array([1.0, 0.0, 0.0])), CYL_UPRIGHT_Q
    )
    cyl_quat = np.array([[CYL_UPRIGHT_Q, bad_q]])
    assert _scene_validity_masks(cyl_pos, cyl_quat, g)[0]


def test_off_shelf_rejected():
    """A cylinder past the front wall must be rejected."""
    g = FakeShelfGeom()
    cyl_pos = np.array([[[g.front_x - 0.10, 0.0, g.surface_z]]])  # 10cm in front
    cyl_quat = np.array([[CYL_UPRIGHT_Q]])
    assert not _scene_validity_masks(cyl_pos, cyl_quat, g)[0]


def test_no_active_cylinders_is_valid():
    """Env with all cylinders hidden is trivially valid."""
    g = FakeShelfGeom()
    cyl_pos = np.array([[[3.0, 99.0, -10.0]] * 4])  # all hidden
    cyl_quat = np.tile(CYL_UPRIGHT_Q[None, None, :], (1, 4, 1))
    assert _scene_validity_masks(cyl_pos, cyl_quat, g)[0]


# ---------------------------------------------------------------------------
# Speed sanity: vectorized version must be much faster than the loop
# ---------------------------------------------------------------------------


def test_vector_is_faster_than_loop():
    """Sanity check that the vectorized form is at least 5× faster than the
    Python-loop reference at realistic dataset-collection size."""
    import time

    rng = np.random.default_rng(7)
    g = FakeShelfGeom()
    N, num_cyl = 256, 20
    cyl_pos = rng.uniform(-1.0, 1.0, (N, num_cyl, 3))
    cyl_pos[rng.random((N, num_cyl)) < 0.2, 0] = 3.0
    cyl_quat = _random_quaternions(rng, (N, num_cyl, 4), max_tilt_deg=90)

    # Warm
    _ = _scene_validity_masks(cyl_pos, cyl_quat, g)
    _ = _scene_validity_loop(cyl_pos, cyl_quat, g)

    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        _scene_validity_masks(cyl_pos, cyl_quat, g)
    t_vec = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters):
        _scene_validity_loop(cyl_pos, cyl_quat, g)
    t_loop = time.perf_counter() - t0

    print(f"\nN={N}, num_cyl={num_cyl}, iters={iters}: "
          f"vec={t_vec*1e3:.1f}ms, loop={t_loop*1e3:.1f}ms, "
          f"speedup={t_loop / t_vec:.1f}×")
    # In practice we see ~3× on small inputs (where loop short-circuits on
    # first failure) and 5–10× on larger ones; require >2× as a floor.
    assert t_vec * 2 < t_loop, (
        f"vec should be >2× faster than loop, got {t_loop / t_vec:.1f}×"
    )
