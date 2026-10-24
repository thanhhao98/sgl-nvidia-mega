# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/step5.py (Step5DecoderLayer._forward_ffn, Step5MoE.forward).
"""The FFN of a covered step, over SGLang's loaded Step5MLP / Step5MoE.

Stock: ``Step5DecoderLayer._forward_ffn``'s TP arm with ``Step5MoE.forward``'s TP arm
(attention DP and MegaMoE are refused at load). Dense layers 0-2 and 91: the layer's
Step5MLP (gate_up, SwiGLU-step, down; a TP partial). MoE layers 3-90: fork the
alternate stream, the router (BF16 GEMM with FP32 logits; FP8 layers hand the logits
and the selection bias to FlashInfer's in-kernel routing, BF16 layers route with
Step5RouterTopKOp), the routed experts (FP8 layers 3-87:
``FusedMoE.forward_deferred_finalize``, the FlashInfer TRT-LLM block-scale MoE with
its finalize left out; BF16 layers 88-90: ``FusedMoE.forward``, finalized inside
FlashInfer), the shared Step5MLP on the alternate stream, the join. With ``defer`` and
an output Step5BoundaryFusions can fuse, a FP8 layer returns SGLang's
``Step5DeferredMoE`` (gemm2_out, FP32 expert weights, the permutation, the shared
partial) for the next layer's input kernel to finalize; otherwise the finalize +
shared add (``round_routed``), or ``routed + shared`` for the BF16 tails. Returns the
unreduced output; the exit completes or carries it.

Seams: ``dense``, ``router``, ``experts``, ``shared``.
"""

from __future__ import annotations

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/step5.py": (
        "use_moe",
        "mlp",
        "moe",
        "boundary_fusions",
        "alt_stream",
        "overlap_max_rows",
        "share_expert",
        "router",
        "experts",
        "fuse_shared_finalize",
        "forward(num_token_non_padded)",
    ),
    "srt/model_executor/forward_batch_info.py": ("num_token_non_padded",),
    "srt/layers/moe/fused_moe_triton/layer.py": ("forward_deferred_finalize",),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "FlashInferTrtllmDeferredFinalizeOutput",
        "finalize_flashinfer_trtllm_deferred_output",
        "finalize_flashinfer_trtllm_deferred_output(output_dtype)",
        "finalize_flashinfer_trtllm_deferred_output(round_routed)",
    ),
    "srt/layers/step5_allreduce.py": ("Step5DeferredMoE", "can_fuse_moe"),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/step5.py:Step5DecoderLayer._forward_ffn": "1afaeb75691f5144",
    "srt/models/step5.py:Step5MoE.forward": "eaf2f75672a3ce29",
}


def step(layer, hidden_states, st, defer: bool):
    """``layer``'s FFN on the normed input [rows, 4096]: a TP partial, or with
    ``defer`` an FP8 layer's ``Step5DeferredMoE`` handoff (_forward_ffn, TP arm).
    ``defer`` is the exit's ``defer_moe_finalize``, the value SGLang reads from
    ``get_forward()`` inside the same scope."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
        finalize_flashinfer_trtllm_deferred_output,
    )
    from sglang.srt.layers.step5_allreduce import Step5DeferredMoE

    if not layer.use_moe:
        return dense(layer, hidden_states, st)
    shared, routed = moe(layer, hidden_states, st)
    if isinstance(routed, FlashInferTrtllmDeferredFinalizeOutput):
        if defer and layer.boundary_fusions.can_fuse_moe(shared, routed):
            return Step5DeferredMoE(routed, shared)
        return finalize_flashinfer_trtllm_deferred_output(
            routed, shared, output_dtype=torch.bfloat16, round_routed=True
        )
    # MegaMoE's deferred combine is refused at load (moe.use_mega_moe)
    return routed + shared


def moe(layer, hidden_states, st):
    """``Step5MoE.forward``, TP arm: (shared partial, routed output)."""
    m = layer.moe
    rows = hidden_states.shape[0]
    # forward_batch is the step's, its mode target verify (plan)
    overlap = m.alt_stream is not None and 0 < rows <= m.overlap_max_rows
    if overlap:
        current_stream = torch.cuda.current_stream()
        m.alt_stream.wait_stream(current_stream)
        # keep collectives and the routed workspace on the main stream
        hidden_states.record_stream(m.alt_stream)
    else:
        shared_output = shared(layer, hidden_states, st)
    topk = router(layer, hidden_states, st)
    routed_output = experts(layer, hidden_states, topk, st)
    if overlap:
        with torch.cuda.stream(m.alt_stream):
            shared_output = shared(layer, hidden_states, st)
        current_stream.wait_stream(m.alt_stream)
        shared_output.record_stream(current_stream)
    return shared_output, routed_output


# --- seams: each is SGLang's own call in the stock path ---------------------------


def dense(layer, hidden_states, st):
    """The dense Step5MLP (layers 0-2, 91): gate_up, SwiGLU-step, down."""
    return layer.mlp(hidden_states)


def router(layer, hidden_states, st):
    """Step5Router: the BF16 gate GEMM with FP32 logits, then the TopK (BYPASSED for
    the FP8 layers' in-kernel routing, Step5RouterTopKOp for the BF16 tails)."""
    return layer.moe.router(
        hidden_states, num_token_non_padded=st.fb.num_token_non_padded
    )


def experts(layer, hidden_states, topk, st):
    """The routed experts: FP8 layers leave their finalize to the caller."""
    m = layer.moe
    if (
        m.fuse_shared_finalize
        and hidden_states.dtype == torch.bfloat16
        and st.bound["ffn.runner_defers"]
    ):
        return m.experts.forward_deferred_finalize(hidden_states, topk)
    return m.experts(hidden_states, topk)


def shared(layer, hidden_states, st):
    """The shared expert's Step5MLP (BF16, 768 / 384 per rank)."""
    return layer.moe.share_expert(hidden_states)
