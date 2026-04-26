"""CPU tests for StickPush trace visualization helpers."""

import numpy as np
import torch

from taskbench.skills.stick_push import (
    StickPushPhaseLineMetrics,
    StickPushPhaseTrace,
    StickPushTraceDebug,
)
from taskbench.skills.stick_push_trace_viz import (
    build_trace_viz_spec,
    render_trace_viz_image,
)
from taskbench.solvers.shelf_panda_stick_push_batched import (
    ShelfPandaStickPushBatchedSolver,
)


def _make_trace_debug() -> StickPushTraceDebug:
    p_start_entry = torch.tensor(
        [[0.20, -0.05, 0.45], [0.20, 0.00, 0.45], [0.20, 0.05, 0.45]],
        dtype=torch.float32,
    )
    p_end_entry = torch.tensor(
        [[0.30, -0.05, 0.45], [0.30, 0.00, 0.45], [0.30, 0.05, 0.45]],
        dtype=torch.float32,
    )
    p_start_sweep = p_end_entry.clone()
    p_end_sweep = torch.tensor(
        [[0.34, 0.02, 0.45], [0.34, 0.10, 0.45], [0.34, -0.08, 0.45]],
        dtype=torch.float32,
    )
    p_start_retract = p_end_sweep.clone()
    p_end_retract = torch.tensor(
        [[0.05, 0.02, 0.45], [0.05, 0.10, 0.45], [0.05, -0.08, 0.45]],
        dtype=torch.float32,
    )

    traces_entry = [
        np.array([[0.20, -0.05, 0.45], [0.25, -0.05, 0.45], [0.30, -0.05, 0.45]], dtype=np.float32),
        np.array([[0.20, 0.00, 0.45], [0.25, 0.02, 0.45], [0.30, 0.04, 0.45]], dtype=np.float32),
        np.array([[0.20, 0.05, 0.45], [0.25, 0.05, 0.45], [0.30, 0.05, 0.45]], dtype=np.float32),
    ]
    traces_sweep = [
        np.array([[0.30, -0.05, 0.45], [0.32, -0.01, 0.45], [0.34, 0.02, 0.45]], dtype=np.float32),
        np.array([[0.30, 0.00, 0.45], [0.32, 0.05, 0.45], [0.34, 0.10, 0.45]], dtype=np.float32),
        np.array([[0.30, 0.05, 0.45], [0.32, -0.01, 0.45], [0.34, -0.08, 0.45]], dtype=np.float32),
    ]
    traces_retract = [
        np.array([[0.34, 0.02, 0.45], [0.20, 0.02, 0.45], [0.05, 0.02, 0.45]], dtype=np.float32),
        np.array([[0.34, 0.10, 0.45], [0.20, 0.12, 0.45], [0.05, 0.13, 0.45]], dtype=np.float32),
        np.array([[0.34, -0.08, 0.45], [0.20, -0.07, 0.45], [0.05, -0.08, 0.45]], dtype=np.float32),
    ]

    entry_metrics = StickPushPhaseLineMetrics(
        max_offtrack=torch.tensor([0.005, 0.030, 0.010], dtype=torch.float32),
        final_offtrack=torch.tensor([0.002, 0.018, 0.005], dtype=torch.float32),
        final_along_error=torch.tensor([0.001, 0.002, 0.001], dtype=torch.float32),
        offtrack_violation=torch.tensor([False, True, False]),
    )
    sweep_metrics = StickPushPhaseLineMetrics(
        max_offtrack=torch.tensor([0.008, 0.040, 0.012], dtype=torch.float32),
        final_offtrack=torch.tensor([0.003, 0.022, 0.006], dtype=torch.float32),
        final_along_error=torch.tensor([0.002, 0.001, 0.002], dtype=torch.float32),
        offtrack_violation=torch.tensor([False, True, False]),
    )
    retract_metrics = StickPushPhaseLineMetrics(
        max_offtrack=torch.tensor([0.010, 0.012, 0.050], dtype=torch.float32),
        final_offtrack=torch.tensor([0.002, 0.011, 0.020], dtype=torch.float32),
        final_along_error=torch.tensor([0.001, 0.003, 0.001], dtype=torch.float32),
        offtrack_violation=torch.tensor([False, False, True]),
    )

    return StickPushTraceDebug(
        entry=StickPushPhaseTrace(
            phase="entry",
            planned_start_xyz=p_start_entry,
            planned_end_xyz=p_end_entry,
            actual_trace_xyz=traces_entry,
            metrics=entry_metrics,
        ),
        sweep=StickPushPhaseTrace(
            phase="sweep",
            planned_start_xyz=p_start_sweep,
            planned_end_xyz=p_end_sweep,
            actual_trace_xyz=traces_sweep,
            metrics=sweep_metrics,
        ),
        retract=StickPushPhaseTrace(
            phase="retract",
            planned_start_xyz=p_start_retract,
            planned_end_xyz=p_end_retract,
            actual_trace_xyz=traces_retract,
            metrics=retract_metrics,
        ),
    )


def test_render_trace_viz_image_smoke():
    debug = _make_trace_debug()
    spec = build_trace_viz_spec(
        env_idx=1,
        trace_debug=debug,
        shelf_front_x=0.25,
        shelf_back_x=0.45,
        shelf_half_w=0.25,
        blocker_disks_xy_r=np.array([[0.33, 0.0, 0.018]], dtype=np.float32),
    )
    image = render_trace_viz_image(spec, width=960, height=720)
    assert isinstance(image, np.ndarray)
    assert image.shape == (720, 960, 3)
    assert image.dtype == np.uint8


def test_select_trace_viz_envs_prefers_violations():
    debug = _make_trace_debug()
    solver = ShelfPandaStickPushBatchedSolver(
        pushes_per_episode=1,
        trace_viz_enabled=True,
        trace_viz_max_envs_per_push=2,
        trace_viz_save_only_violations=True,
    )
    selected = solver._select_trace_viz_envs(debug)
    # violating envs are 1 and 2; env 2 has higher max offtrack than env 1
    assert selected == [2, 1]


def test_select_trace_viz_envs_falls_back_to_topk_without_violations():
    debug = _make_trace_debug()
    debug.entry.metrics.offtrack_violation[:] = False
    debug.sweep.metrics.offtrack_violation[:] = False
    debug.retract.metrics.offtrack_violation[:] = False

    solver = ShelfPandaStickPushBatchedSolver(
        pushes_per_episode=1,
        trace_viz_enabled=True,
        trace_viz_max_envs_per_push=2,
        trace_viz_save_only_violations=True,
    )
    selected = solver._select_trace_viz_envs(debug)
    # scores: env0=0.010, env1=0.040, env2=0.050
    assert selected == [2, 1]
