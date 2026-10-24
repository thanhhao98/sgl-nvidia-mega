# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot ``swa_attn``: a covered step's SWA layers (69) run their attention core as the
page-split kernel and its merge (sm103/attention/swa_split.py) instead of FA4 through
RadixAttention, which spends one CTA per request on the 515-key window.

It replaces exactly SGLang's FA4 call with the sigmoid head gate in its epilogue, after
the q/k-norm kernel stored this step's K / V rows (``kv_write``): the slot reads those
rows, the layer's SWA page table and sequence lengths, and applies the gate itself, so
any other path (no fused store, no gate epilogue) stops the step instead of running the
slot on inputs it was not built for. Numerics: tolerance (runtime.TOLERANCES).
"""

from __future__ import annotations

from sgl_nvidia_mega.step5.plan import TOKENS_PER_REQUEST

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/kernel_plugins/megakernel/binding.py|srt/megakernels/binding.py": (
        "swa_metadata",
    ),
    "srt/layers/attention/flashattention_backend.py": (
        "swa_page_table",
        "cache_seqlens_int32",
        "swa_spec_metadata",
    ),
    "srt/layers/radix_attention.py": (
        "tp_q_head_num",
        "tp_k_head_num",
        "head_dim",
        "sliding_window_size",
        "scaling",
    ),
    "srt/models/step5.py": ("self_attn", "attn", "is_dsa"),
    "srt/runtime_context.py": ("get_exec",),
    "srt/arg_groups/fields/*.py": ("enable_deterministic_inference",),
}


def unsupported(model) -> list[str]:
    """Why this rank's model cannot run the slot ([] when it can): every SWA layer has
    the shape the kernels are compiled for."""
    from sglang.srt.runtime_context import get_exec

    from sgl_nvidia_mega.step5.sm103.attention import swa_split as k

    reasons = []
    if get_exec().deterministic.enable_deterministic_inference:
        reasons.append(
            "slot swa_attn: deterministic inference keeps SGLang's FA4 reduction order"
        )
    want = {
        "tp_q_head_num": k.H,
        "tp_k_head_num": 1,
        "head_dim": k.D,
        "sliding_window_size": k.WINDOW_LEFT,
    }
    for layer in model.language_model.model.layers:
        a = layer.self_attn
        if a.is_dsa:
            continue
        got = {name: getattr(a.attn, name, None) for name in want}
        bad = {n: v for n, v in got.items() if v != want[n]}
        if abs(float(getattr(a.attn, "scaling", 0.0)) - k.D**-0.5) > 1e-12:
            bad["scaling"] = getattr(a.attn, "scaling", None)
        if bad:
            reasons.append(
                f"slot swa_attn: SWA layer {layer.layer_id} has {bad}, the kernels "
                f"take {want} and scaling {k.D}**-0.5"
            )
            break
    return reasons


def build(widths: tuple[int, ...]):
    """The kernels for the covered widths, compiled and warmed up (no capture yet)."""
    import torch

    from sgl_nvidia_mega.step5.sm103.attention.swa_split import SwaVerifyAttention

    kernels = SwaVerifyAttention(
        max(widths) // TOKENS_PER_REQUEST,
        torch.device("cuda", torch.cuda.current_device()),
    )
    kernels.warmup()
    return kernels


def attend(layer, q, gate, kv_write, st, *, gate_epilogue: bool):
    """The gated SWA attention output [rows, 16 * 192] of ``layer`` for this step."""
    if not gate_epilogue or kv_write is None:
        raise RuntimeError(
            f"NVIDIA megakernel: slot swa_attn at SWA layer {layer.layer_id}: SGLang "
            f"would not run FA4 with the gate epilogue after the fused K / V store here "
            f"(gate epilogue {gate_epilogue}, fused store {kv_write is not None})"
        )
    md = st.step5.swa_metadata
    if md is None or md.swa_spec_metadata is not None or md.swa_page_table is None:
        raise RuntimeError(
            f"NVIDIA megakernel: slot swa_attn at SWA layer {layer.layer_id}: FA4 would "
            "not read the SWA page table of the forward metadata here"
        )
    bs = st.rows // TOKENS_PER_REQUEST
    k_rows, v_rows, _ = kv_write
    return st.bound["swa_attn.kernels"](
        q,
        gate,
        k_rows,
        v_rows,
        md.swa_page_table[:bs],
        md.cache_seqlens_int32[:bs],
    )
