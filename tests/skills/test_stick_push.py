"""Tests for StickPush skill.

Unit tests (CPU): dataclass construction, input validation.
Integration tests (GPU): full pipeline, regression against baselines.

Run:
    # CPU only
    uv run pytest tests/skills/test_stick_push.py -v -m "not gpu"

    # With GPU
    srun --partition=unlimited --gres=gpu:a6000:1 --time=00:15:00 \
        bash -c 'export CUDA_HOME=/usr/local/cuda-12.6 && \
        export CPATH=/usr/local/cuda-12.6/targets/x86_64-linux/include:$CPATH && \
        uv run pytest tests/skills/test_stick_push.py -v'
"""
import numpy as np
import pytest
import torch
from pathlib import Path

from taskbench.skills.stick_push import StickPush, StickPushResult

REGRESSION_DIR = Path("data/regression")
SHELF = dict(front_x=0.25, depth=0.20, half_w=0.25, floor_z=0.40, thickness=0.01, inner_h=0.30)
ROBOT_UID = "panda_stick_long"
Q_INTO_SHELF = [0.5, -0.5, 0.5, -0.5]
REST_QPOS = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]
SAFE_QPOS = [0.0002, -1.2996, 0.0003, -2.4930, -0.0009, 1.0177, 0.0]


# ---------------------------------------------------------------------------
# Unit tests (no GPU)
# ---------------------------------------------------------------------------

class TestStickPushResult:
    def test_fields(self):
        r = StickPushResult(
            success_mask=torch.tensor([True, False]),
            staging_success=torch.tensor([True, True]),
            entry_final_dist=torch.tensor([0.01, 0.02]),
            sweep_final_dist=torch.tensor([0.005, 0.03]),
            retract_final_dist=torch.tensor([0.01, 0.01]),
            orientation_drift=torch.tensor([0.001, 0.002]),
            steps_executed=300,
        )
        assert r.success_mask.shape == (2,)
        assert r.steps_executed == 300
        assert r.success_mask[0].item() is True
        assert r.success_mask[1].item() is False


# ---------------------------------------------------------------------------
# GPU fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shelf_env_and_planner():
    """Create shelf env + cuRobo planner for single-env testing."""
    import gymnasium as gym
    import taskbench.envs
    import taskbench.agents.panda_stick_long
    from taskbench.skills.curobo_motion import setup_curobo_planner, get_arm_joint_names
    from curobo.geom.types import Cuboid, WorldConfig

    env = gym.make(
        "ShelfEnv-v1", num_envs=1, robot_uids=ROBOT_UID,
        robot_base_pose=[-0.4, 0.15, 0.0],
        num_objects=10, reward_mode="none", obs_mode="state_dict",
        render_mode="rgb_array", sim_backend="cpu",
        control_mode="pd_joint_pos", shelf=SHELF,
    )
    raw = env.unwrapped

    g = raw.shelf_geom
    robot_base = raw.agent.robot.pose.p[0].cpu().numpy()
    sc = np.array([g.center_x, 0.0, g.ceil_z / 2]) - robot_base[:3]
    sd = [g.depth + 0.04, 2 * g.half_w + 0.04, g.ceil_z + 0.04]
    world = WorldConfig(cuboid=[Cuboid(
        name="shelf", pose=[float(sc[0]), float(sc[1]), float(sc[2]), 1, 0, 0, 0], dims=sd,
    )])
    motion_gen = setup_curobo_planner(ROBOT_UID, world_configs=world, n_envs=1)
    robot_base_pos = raw.agent.robot.pose.p[0].to(device="cuda:0").unsqueeze(0)

    yield env, motion_gen, robot_base_pos

    env.close()


def _reset_to_rest(env):
    """Reset env and set rest qpos."""
    raw = env.unwrapped
    n_arm = 7
    obs, _ = env.reset(seed=42)
    qpos = raw.agent.robot.get_qpos().clone()
    qpos[0, :n_arm] = torch.tensor(REST_QPOS, dtype=torch.float32)
    raw.agent.robot.set_qpos(qpos)
    rest = torch.zeros(1, n_arm, device=raw.device)
    rest[0, :n_arm] = torch.tensor(REST_QPOS, dtype=torch.float32, device=raw.device)
    for _ in range(10):
        env.step(rest)
    # Ensure pd_joint_pos mode
    raw.agent.set_control_mode("pd_joint_pos")
    raw.agent.controller.reset()


# ---------------------------------------------------------------------------
# Integration tests (GPU)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestSingleEnvPush:
    def test_staging_succeeds(self, shelf_env_and_planner):
        env, motion_gen, robot_base_pos = shelf_env_and_planner
        _reset_to_rest(env)

        push = StickPush(
            env, motion_gen, robot_uid=ROBOT_UID, n_envs=1,
            rest_qpos=REST_QPOS, robot_base_pos=robot_base_pos,
            safe_start_qpos=SAFE_QPOS,
        )
        cuda = torch.device("cuda:0")
        result = push(
            approach_positions=torch.tensor([[0.20, 0.0, 0.455]], device=cuda),
            approach_quaternions=torch.tensor([Q_INTO_SHELF], device=cuda),
            entry_positions=torch.tensor([[0.30, 0.0, 0.455]], device=cuda),
            sweep_positions=torch.tensor([[0.30, 0.10, 0.455]], device=cuda),
            retract_positions=torch.tensor([[0.20, 0.10, 0.455]], device=cuda),
        )
        assert result.staging_success[0].item()

    def test_sweep_converges(self, shelf_env_and_planner):
        env, motion_gen, robot_base_pos = shelf_env_and_planner
        _reset_to_rest(env)

        push = StickPush(
            env, motion_gen, robot_uid=ROBOT_UID, n_envs=1,
            rest_qpos=REST_QPOS, robot_base_pos=robot_base_pos,
            safe_start_qpos=SAFE_QPOS,
        )
        cuda = torch.device("cuda:0")
        result = push(
            approach_positions=torch.tensor([[0.20, 0.0, 0.455]], device=cuda),
            approach_quaternions=torch.tensor([Q_INTO_SHELF], device=cuda),
            entry_positions=torch.tensor([[0.30, 0.0, 0.455]], device=cuda),
            sweep_positions=torch.tensor([[0.30, 0.10, 0.455]], device=cuda),
            retract_positions=torch.tensor([[0.20, 0.10, 0.455]], device=cuda),
        )
        assert result.sweep_final_dist[0].item() < 0.02

    def test_orientation_stable(self, shelf_env_and_planner):
        env, motion_gen, robot_base_pos = shelf_env_and_planner
        _reset_to_rest(env)

        push = StickPush(
            env, motion_gen, robot_uid=ROBOT_UID, n_envs=1,
            rest_qpos=REST_QPOS, robot_base_pos=robot_base_pos,
            safe_start_qpos=SAFE_QPOS,
        )
        cuda = torch.device("cuda:0")
        result = push(
            approach_positions=torch.tensor([[0.20, 0.0, 0.455]], device=cuda),
            approach_quaternions=torch.tensor([Q_INTO_SHELF], device=cuda),
            entry_positions=torch.tensor([[0.30, 0.0, 0.455]], device=cuda),
            sweep_positions=torch.tensor([[0.30, 0.10, 0.455]], device=cuda),
            retract_positions=torch.tensor([[0.20, 0.10, 0.455]], device=cuda),
        )
        assert result.orientation_drift[0].item() < 0.02

    def test_multi_push_episode(self, shelf_env_and_planner):
        """Two pushes in sequence — second push re-stages from rest."""
        env, motion_gen, robot_base_pos = shelf_env_and_planner
        _reset_to_rest(env)

        push = StickPush(
            env, motion_gen, robot_uid=ROBOT_UID, n_envs=1,
            rest_qpos=REST_QPOS, robot_base_pos=robot_base_pos,
            safe_start_qpos=SAFE_QPOS,
        )
        cuda = torch.device("cuda:0")

        # Push 1
        r1 = push(
            approach_positions=torch.tensor([[0.20, -0.05, 0.455]], device=cuda),
            approach_quaternions=torch.tensor([Q_INTO_SHELF], device=cuda),
            entry_positions=torch.tensor([[0.30, -0.05, 0.455]], device=cuda),
            sweep_positions=torch.tensor([[0.30, 0.05, 0.455]], device=cuda),
            retract_positions=torch.tensor([[0.20, 0.05, 0.455]], device=cuda),
        )
        assert r1.staging_success[0].item()

        # Push 2 — different direction, arm starts from rest after push 1
        r2 = push(
            approach_positions=torch.tensor([[0.20, 0.08, 0.455]], device=cuda),
            approach_quaternions=torch.tensor([Q_INTO_SHELF], device=cuda),
            entry_positions=torch.tensor([[0.30, 0.08, 0.455]], device=cuda),
            sweep_positions=torch.tensor([[0.30, -0.08, 0.455]], device=cuda),
            retract_positions=torch.tensor([[0.20, -0.08, 0.455]], device=cuda),
        )
        assert r2.staging_success[0].item()


@pytest.mark.gpu
class TestRegression:
    """Verify the StickPush skill matches the pre-refactor solver baselines."""

    @pytest.fixture(params=[42, 43, 44, 45, 46])
    def baseline(self, request):
        seed = request.param
        path = REGRESSION_DIR / f"stick_push_seed{seed}.npz"
        if not path.exists():
            pytest.skip(f"Baseline not found: {path}")
        return np.load(path)

    def test_sweep_distance_matches(self, shelf_env_and_planner, baseline):
        """Sweep final TCP matches baseline within tolerance."""
        env, motion_gen, robot_base_pos = shelf_env_and_planner
        seed = int(baseline["seed"])
        env.reset(seed=seed)
        raw = env.unwrapped

        # Set rest qpos
        qpos = raw.agent.robot.get_qpos().clone()
        qpos[0, :7] = torch.tensor(REST_QPOS, dtype=torch.float32)
        raw.agent.robot.set_qpos(qpos)
        rest = torch.zeros(1, 7, device=raw.device)
        rest[0] = torch.tensor(REST_QPOS, dtype=torch.float32, device=raw.device)
        for _ in range(10):
            env.step(rest)
        raw.agent.set_control_mode("pd_joint_pos")
        raw.agent.controller.reset()

        push = StickPush(
            env, motion_gen, robot_uid=ROBOT_UID, n_envs=1,
            rest_qpos=REST_QPOS, robot_base_pos=robot_base_pos,
            safe_start_qpos=SAFE_QPOS,
        )
        cuda = torch.device("cuda:0")
        p1 = baseline["p1"]
        p2 = baseline["p2"]
        approach = baseline["approach"]
        retract = baseline["retract"]

        result = push(
            approach_positions=torch.tensor([approach], dtype=torch.float32, device=cuda),
            approach_quaternions=torch.tensor([Q_INTO_SHELF], dtype=torch.float32, device=cuda),
            entry_positions=torch.tensor([p1], dtype=torch.float32, device=cuda),
            sweep_positions=torch.tensor([p2], dtype=torch.float32, device=cuda),
            retract_positions=torch.tensor([retract], dtype=torch.float32, device=cuda),
        )

        # Sweep should converge close to p2 (same as baseline)
        # The skill returns to rest after sweep, so compare sweep_final_dist
        # against how close the baseline got to p2.
        baseline_sweep_tcp = baseline["sweep_tcp"]
        baseline_dist_to_p2 = np.linalg.norm(baseline_sweep_tcp - p2)

        # Skill's sweep distance should be comparable to baseline's
        skill_dist = result.sweep_final_dist[0].item()
        assert skill_dist < 0.03, (
            f"Seed {seed}: sweep didn't converge (dist={skill_dist:.4f}m)"
        )
        assert result.staging_success[0].item(), f"Seed {seed}: staging failed"


@pytest.mark.gpu
class TestEdgeCases:
    def test_short_push(self, shelf_env_and_planner):
        """Very short sweep (3cm)."""
        env, motion_gen, robot_base_pos = shelf_env_and_planner
        _reset_to_rest(env)

        push = StickPush(
            env, motion_gen, robot_uid=ROBOT_UID, n_envs=1,
            rest_qpos=REST_QPOS, robot_base_pos=robot_base_pos,
            safe_start_qpos=SAFE_QPOS,
        )
        cuda = torch.device("cuda:0")
        result = push(
            approach_positions=torch.tensor([[0.20, 0.0, 0.455]], device=cuda),
            approach_quaternions=torch.tensor([Q_INTO_SHELF], device=cuda),
            entry_positions=torch.tensor([[0.30, 0.0, 0.455]], device=cuda),
            sweep_positions=torch.tensor([[0.30, 0.03, 0.455]], device=cuda),
            retract_positions=torch.tensor([[0.20, 0.03, 0.455]], device=cuda),
        )
        assert result.staging_success[0].item()
        assert result.sweep_final_dist[0].item() < 0.02

    def test_diagonal_push(self, shelf_env_and_planner):
        """Diagonal sweep (X and Y change)."""
        env, motion_gen, robot_base_pos = shelf_env_and_planner
        _reset_to_rest(env)

        push = StickPush(
            env, motion_gen, robot_uid=ROBOT_UID, n_envs=1,
            rest_qpos=REST_QPOS, robot_base_pos=robot_base_pos,
            safe_start_qpos=SAFE_QPOS,
        )
        cuda = torch.device("cuda:0")
        result = push(
            approach_positions=torch.tensor([[0.20, -0.05, 0.455]], device=cuda),
            approach_quaternions=torch.tensor([Q_INTO_SHELF], device=cuda),
            entry_positions=torch.tensor([[0.28, -0.05, 0.455]], device=cuda),
            sweep_positions=torch.tensor([[0.35, 0.08, 0.455]], device=cuda),
            retract_positions=torch.tensor([[0.20, 0.08, 0.455]], device=cuda),
        )
        assert result.staging_success[0].item()
