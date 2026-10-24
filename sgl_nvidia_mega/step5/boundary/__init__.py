# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/step5.py (Step5DecoderLayer.forward, Step5Model.forward).
"""The residual stream between sublayers.

``open_stream`` starts it (``residual_batch.start``); per layer ``attn_input`` (the
input norm, completing what the previous layer left: the embedding as the stack's
start, the deferred FP8 MoE finalize, or an unreduced partial), ``attn_output`` (the
attention partial recorded as an ATTN_TP sum), ``ffn_input`` (that sum's all-reduce +
residual + post-attention norm), ``ffn_exit`` (the FFN's exit scope: its
``defer_moe_finalize`` and its ``finish``); ``close_stream`` is
``Step5Model.forward``'s tail.

Stock: SGLang's own StageBoundary objects of each layer, in
``Step5DecoderLayer.forward``'s order. With the Step-5 multicast communicator and up to
32 rows they run sglang::moe_finalize_all_reduce_kernel at both boundaries
(Step5BoundaryFusions._ffn_input and ._attn_input), the dense layers' FFN output and
the BF16 tails through FlashInfer's fused all-reduce + residual + RMSNorm, and L91's
through ``complete_now``'s all-reduce; without the communicator every boundary is
FlashInfer's fusion.
"""

from __future__ import annotations

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # residual_batch: the module alias step5.py imports too
    "srt/models/step5.py": ("residual_batch", "attn_boundary", "ffn_boundary", "norm"),
    "srt/layers/layer_boundary/stage.py": (
        "prepare",
        "_prepare_attention(capture_gathered)",
        "finish",
        "exit",
    ),
    "srt/layers/layer_boundary/exit.py": ("defer_moe_finalize", "finish"),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "fold",
        "take_output",
    ),
    "kernels/spec.py": ("KernelBackend", "TORCH"),
    # the final norm's per-call backend (GemmaRMSNorm is a BaseFusedOp)
    "kernels/fused_op.py": ("forward(backend)",),
}

# The SGLang functions whose body this module mirrors, and the bodies close_stream's
# equivalence to Step5Model.forward's tail rests on (it calls fold and take_output)
SGLANG_MIRRORS = {
    "srt/models/step5.py:Step5DecoderLayer.forward": "493ca877ada2f7a7",
    "srt/models/step5.py:Step5Model.forward": "b7846fb61d679b85",
    "srt/layers/layer_boundary/residual/batch.py:fold": "f9b719619fefc572",
    "srt/layers/layer_boundary/residual/batch.py:take_output": "58292f564c5ac14a",
}


def open_stream(st) -> None:
    """``Step5Model.forward``'s ``residual_batch.start``."""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

    residual_batch.start(st.fb)


def attn_input(layer, hidden, st):
    """``layer``'s attention input [rows, 4096] from the stream (its input norm)."""
    return layer.attn_boundary.prepare(hidden, st.fb, capture_gathered=None)


def attn_output(layer, part, st):
    """The attention's TP partial into the stream."""
    return layer.attn_boundary.finish(part, st.fb)


def ffn_input(layer, hidden, st):
    """The FFN input [rows, 4096]: the attention sum, residual add and the
    post-attention norm."""
    return layer.ffn_boundary.prepare(hidden, st.fb)


def ffn_exit(layer, st):
    """The FFN's exit scope; ``defer_moe_finalize`` tells the FFN whether to hand its
    finalize to the next input, ``finish(out)`` completes or carries the output."""
    return layer.ffn_boundary.exit(st.fb)


def close_stream(st, hidden):
    """``Step5Model.forward`` after the layers: the stream's (output, residual) pair,
    their BF16 sum, the final norm on the TORCH backend; returns (normed, the FP32
    pre-norm sum the MTP draft consumes).

    SGLang exports the pair, sets ``forward_batch.residual_stream = None`` and adds
    ``hidden_states + residual``. ``residual_batch.fold`` exports the same pair and
    runs that same add (the pending update is PLAIN_ADD on Step-5);
    ``residual_batch.take_output`` then drops the stream, so the plugin assigns no
    ForwardBatch attribute itself. Same kernels, same bits."""
    from sglang.kernels.spec import KernelBackend
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

    summed = residual_batch.take_output(residual_batch.fold(hidden, st.fb), st.fb)
    normalized = st.model.norm(summed, backend=KernelBackend.TORCH)
    return normalized, summed.float()
