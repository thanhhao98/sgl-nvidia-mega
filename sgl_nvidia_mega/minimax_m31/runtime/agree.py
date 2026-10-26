# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/kernel_plugins/nvidia/binding.py (megakernel_agree, _gather_tp)
"""Build-time agreement over the TP group: every rank's value (``gather_tp``), and a
refusal on every rank when the ranks differ (``agree_tp``)."""

from __future__ import annotations

from typing import Any

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel", '"tp_group"'),
    "srt/distributed/parallel_state.py": ("world_size", "cpu_group"),
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": ("FLAG",),
}


def gather_tp(obj: Any) -> list[Any]:
    """``obj`` from every TP rank, through the TP group's CPU group (a collective)."""
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    if group.world_size == 1:
        return [obj]
    out: list[Any] = [None] * group.world_size
    dist.all_gather_object(out, obj, group=group.cpu_group)
    return out


def agree_tp(value: Any, *, what: str) -> None:
    """Raise on every rank unless every TP rank holds ``value``."""
    try:
        from sglang.kernels.plugin_kernels.nvidia.spec import FLAG
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.megakernels.spec import FLAG

    values = gather_tp(value)
    if any(v != value for v in values):
        raise RuntimeError(
            f"{FLAG}: the ranks' {what} differ: "
            + "; ".join(f"rank {rank}: {v}" for rank, v in enumerate(values))
        )
