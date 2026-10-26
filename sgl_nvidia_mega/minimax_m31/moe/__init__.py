# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The FFN sublayer of a covered step: the dense MLP of layers 0-2, and the MoE of
layers 3-59 (the shared expert on the main stream; the router, the top-4 and the
NVFP4 routed experts on the alternate stream, their finalize deferred into the next
boundary).

Stock: SGLang's ``MiniMaxM3MLP`` / ``MiniMaxM3MoE`` modules.

Slot ``moe_nofill`` (``moe_nofill.py``, bitwise): an MoE layer runs SGLang's own calls
in SGLang's order, but quantizes the routed experts' NVFP4 input into buffers built
once, without the chain's fill launch and allocations. A step that misses one of the slot's
preconditions runs SGLang's module for that layer.

Slot ``hop_ffn`` (``boundary/hop_ffn.py``, bitwise; needs moe_nofill): the FFN input
is an ``hop_ffn.FfnInput``, the bf16 normed rows with the quantized inputs the
boundary already wrote. The gate_up of the dense MLP or the shared expert takes their
MXFP8 pair (``hop_ffn.mlp_input``) and moe_nofill's routed experts their per-token
NVFP4 (``moe_nofill.Prequantized``), so neither quantizes; everything else reads the
bf16 rows, and a layer that falls back to SGLang's module gets them too.
"""

from __future__ import annotations

from typing import Any

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/minimax_m3.py": (
        "is_layer_sparse",
        "layer_id",
        "mlp",
        "shared_experts",
        "forward(forward_batch)",
    ),
}


def step(layer, x, st) -> Any:
    """``layer``'s FFN output from its normed input ``x`` (bf16 rows, or an
    ``FfnInput`` from slot hop_ffn): an unreduced partial, or the MoE's
    deferred-finalize handoff."""
    # imported here, not at the top: boundary.hop_ffn imports this package's moe_nofill
    from sgl_nvidia_mega.minimax_m31.boundary.hop_ffn import FfnInput, mlp_input

    prepared = x if isinstance(x, FfnInput) else None
    if prepared is not None:
        x = prepared.x
    if st.slots.moe_nofill and layer.is_layer_sparse:
        from sgl_nvidia_mega.minimax_m31.moe import moe_nofill

        bound = st.bound.get("moe")
        why = moe_nofill.fallback_reason(bound, layer, x)
        if not why:
            mlp, quant, shared_input = layer.mlp, bound.quant, x
            if prepared is not None:
                shared_input = mlp_input(mlp.shared_experts, prepared)
                if prepared.nvfp4 is not None:
                    quant = moe_nofill.Prequantized(x, prepared.nvfp4, quant)
            return moe_nofill.forward(mlp, x, quant, shared_input=shared_input)
        moe_nofill.log_fallback(layer.layer_id, why)
    elif prepared is not None and not layer.is_layer_sparse:
        return layer.mlp(mlp_input(layer.mlp, prepared), forward_batch=st.fb)
    return layer.mlp(x, forward_batch=st.fb)


def build(model, widths: tuple[int, ...], slots: frozenset):
    """The slot's build (weights bound, buffers allocated, kernels compiled), or None
    when its slot is not requested."""
    if "moe_nofill" not in slots:
        return None
    from sgl_nvidia_mega.minimax_m31.moe import moe_nofill

    return moe_nofill.build(model, widths)


def serves(bound, rows: int) -> bool:
    """Whether the slot's kernels serve a step of this many rows."""
    return bound is not None and bound.serves(rows)
