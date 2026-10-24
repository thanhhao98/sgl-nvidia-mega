# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot ``moe_quant_alt``: an FP8 MoE layer's (3-87) routed-expert input quantization
runs on the layer's alternate stream, beside the router GEMM, instead of on the main
stream between the router and the experts.

The kernel and its input are unchanged: SGLang's FlashInfer TRT-LLM DeepSeekFp8 runner
calls ``per_token_group_quant_fp8(hidden_states, 128, column_major_scales=True)`` on the
experts' input, and this slot makes the same call on the same tensor, so the experts
read the same bits (bitwise). The runner takes the pair through
``FusedMoE.forward_deferred_finalize(pre_quant_input=)`` (SGLang's
``deepseek_fp8_moe_input`` checks its layout and refuses any other). The main stream
waits for the quantization's event just before the experts; the shared expert follows
the quantization on the alternate stream, as late as before (it joins the main stream
where SGLang's forward joins it). The router GEMM, the quantization and the shared
expert are independent, so only the quantization's time leaves the main stream.
"""

from __future__ import annotations

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # the name the runner calls; on CUDA it is sglang_per_token_group_quant_fp8
    "kernels/ops/quantization/fp8_kernel.py": (
        "per_token_group_quant_fp8",
        "sglang_per_token_group_quant_fp8(column_major_scales)",
    ),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": ("deepseek_fp8_moe_input",),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "forward_deferred_finalize(pre_quant_input)",
        "quant_method",
    ),
    "srt/layers/quantization/fp8.py": (
        "quant_config",
        "block_quant",
        "weight_block_size",
        "use_mxfp8",
    ),
}

BLOCK = (
    128  # the DeepSeekFp8 activation group: the experts' weight_block_size [128, 128]
)


def unsupported(model, bound: dict) -> list[str]:
    """Why this rank's model cannot run the slot ([] when it can)."""
    from sgl_nvidia_mega.step5 import FP8_EXPERT_LAYERS

    reasons = []
    try:
        # without it the runner ignores the pair and quantizes again on the main stream
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (  # noqa: F401
            deepseek_fp8_moe_input,
        )
    except ImportError:
        reasons.append(
            "slot moe_quant_alt: this SGLang's FlashInfer TRT-LLM FP8 MoE runner takes no "
            "pre-quantized input (deepseek_fp8_moe_input); update SGLang"
        )
    if not bound["ffn.runner_defers"]:
        reasons.append(
            "slot moe_quant_alt: the MoE runner backend is not FlashInfer TRT-LLM"
        )
    layers = model.language_model.model.layers
    bad = []
    for i in FP8_EXPERT_LAYERS:
        method = layers[i].moe.experts.quant_method
        config = getattr(method, "quant_config", None)
        if not (
            getattr(method, "block_quant", False)
            and getattr(config, "weight_block_size", None) == [BLOCK, BLOCK]
            and not getattr(config, "use_mxfp8", False)
        ):
            bad.append(i)
    if bad:
        reasons.append(
            f"slot moe_quant_alt: the experts of layers {bad} are not DeepSeek FP8 with "
            f"{BLOCK} x {BLOCK} blocks"
        )
    return reasons


def quantize(
    hidden_states: torch.Tensor, alt: torch.cuda.Stream, main: torch.cuda.Stream
):
    """The experts' input pair (E4M3 [rows, 4096], FP32 scales [rows, 32] column-major),
    computed on ``alt``; ``main`` waits for it before it returns, so the next kernel on
    ``main`` (the experts) reads it complete. The caller has already forked ``alt`` from
    ``main`` with ``hidden_states`` ready."""
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8

    with torch.cuda.stream(alt):
        a_q, a_sf = per_token_group_quant_fp8(
            hidden_states, BLOCK, column_major_scales=True
        )
        ready = torch.cuda.Event()
        ready.record(alt)
    a_q.record_stream(main)
    a_sf.record_stream(main)
    main.wait_event(ready)
    return a_q, a_sf
