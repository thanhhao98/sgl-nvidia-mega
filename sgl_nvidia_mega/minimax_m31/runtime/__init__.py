# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel runtime: the kernel slots, the per-rank checks of the loaded model
(``model_disable_reasons``) and the build (``build``): the widths the megakernel
serves, each step package's build result and the slot plan per width. The executor of
a covered step is ``executor``, over its per-step state (``state``); ``agree`` holds
the TP ranks' build-time agreement.

The kernel slots: which of the megakernel's kernels replace SGLang's in a covered step.
With every slot off the step functions run SGLang's own modules, so the covered step
is bit for bit SGLang's forward. A slot is requested by name (``SLOTS_ENV``) and built
per width; the plan (``plan.Slots``) records which slots serve each width. This module
stays light (``build`` imports the rest at build): the spec, which SGLang's launcher
imports, reads the slots here.
"""

from __future__ import annotations

import os
from typing import Any

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "widths",
        "tokens_per_request",
        "binding",
    ),
    "srt/models/minimax_m3.py": (
        "MiniMaxM3DecoderLayer",
        "MiniMaxM3Model",
        "pp_group",
        "start_layer",
        "end_layer",
        "layers",
    ),
    "srt/models/minimax_m3_vl.py": ("model", "is_mrope_enabled", "logits_processor"),
    "srt/distributed/parallel_state.py": ("is_first_rank", "is_last_rank"),
}

# The slots this package ships; unset SLOTS_ENV runs all of them. Both are bitwise,
# checked by their GPU tests (test/kernels/test_minimax_m31_moe_nofill.py; hop_mxfp8's
# kernel: SGLang's test/registered/kernels/ops/communication/
# test_all_reduce_residual_norm_quant.py) and by the model's composition check at
# every covered width (GB300, 2026-10-20):
# - moe_nofill: the routed experts' per-token NVFP4 input quantized into buffers
#   built once, without the two fill launches per MoE layer
# - hop_mxfp8: the attention-input boundary kernel also writes the qkv|index GEMM's
#   MXFP8 input, replacing the separate quantize launch per layer
SHIPPED_SLOTS: tuple[str, ...] = ("moe_nofill", "hop_mxfp8")
# Slots that run only when listed in SLOTS_ENV:
# - attn_ll: the qkv|index GEMM on FlashInfer's low-latency MXFP8 GEMM at the widths
#   it takes (1-8 rows; o_proj too only under its test knob, attention.GEMMS_ENV);
#   with hop_mxfp8 the boundary kernel before it signals its launch at entry, so its
#   weight loads overlap the all-reduce. Not bitwise: its split-K order. Locally
#   within one bf16 ulp, but the model amplifies such differences like a change of
#   GEMM tactic, so it ships after an accuracy comparison, not on parity
# - qk_store: the attention's q/k/index norm + RoPE (FP8 q epilogue) and its NVFP4
#   K/V/index-K page store in one kernel per layer, which SGLang's KV pool launches
#   (attention.qk_store), bitwise (its GPU test,
#   test/kernels/test_minimax_m31_qk_store.py, and a check against SGLang's two
#   kernels at build); opt-in until the model's composition check has run it on GPU
# - hop_ffn: the FFN-input boundary kernel also writes the FFN's quantized inputs
#   (the MXFP8 of the dense MLP's / shared expert's gate_up, a MoE layer's per-token
#   NVFP4 for its routed experts), replacing their quantize launches. Bitwise by
#   construction; ships once its GPU test and the composition check pass on GB300
OPT_IN_SLOTS: tuple[str, ...] = ("attn_ll", "qk_store", "hop_ffn")
# slot -> the slot it needs
NEEDS: dict[str, str] = {
    # the routed experts take hop_ffn's NVFP4 through moe_nofill's chain
    "hop_ffn": "moe_nofill",
}
# The slots whose kernels agree with SGLang's bit for bit; any other requested slot
# makes the spec's numerics TOLERANCE with that slot's tolerances (TOLERANCES).
BITWISE_SLOTS: frozenset[str] = frozenset(
    {"moe_nofill", "hop_mxfp8", "qk_store", "hop_ffn"}
)
# slot -> {quantity: the largest difference against SGLang's op on the same GPU}.
# attn_ll's are a-priori bounds until its GPU test
# (test/kernels/test_minimax_m31_attn_ll.py) measures them; that test asserts each one
# and prints the measured values.
TOLERANCES: dict[str, dict[str, float]] = {
    "attn_ll": {
        # the GEMMs' BF16 outputs against SGLang's MXFP8 GEMM on the same MXFP8
        # input (the FP32 summation order only): the largest |difference| over the
        # largest |value|, and the difference's norm over the output's norm (o_proj's
        # where its test knob runs it on the low-latency kernel)
        "attn.qkv_index.rel": 1e-2,
        "attn.o_proj.rel": 1e-2,
        "attn.qkv_index.norm_rel": 5e-4,
        "attn.o_proj.norm_rel": 5e-4,
        # what SGLang's kernels then write from the q|k|v|index output, where a BF16
        # difference moves a code only when it crosses a rounding boundary: the
        # fraction of FP8 q / index-q codes that move and the largest move (e4m3
        # steps); the fraction of NVFP4 K / V / index-K codes and of their E4M3
        # block scales that move, and the largest dequantized move over the block's
        # coarsest E2M1 step (2 x its scale)
        "attn.fp8_q.moved_fraction": 1e-2,
        "attn.fp8_q.max_e4m3_steps": 1.0,
        "attn.kv.nvfp4.moved_fraction": 1e-2,
        "attn.kv.scale.moved_fraction": 2e-2,
        "attn.kv.nvfp4.block_top_steps": 1.0,
        # the fraction of query rows whose top-16 index pages move (the MSA-v1
        # indexer's page scores from the FP8 index q, over a synthetic context)
        "attn.index.topk_moved_fraction": 5e-2,
    },
}

SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"


def requested_slots(environ=None) -> frozenset[str]:
    """``SLOTS_ENV`` (comma separated; "" turns every slot off, for tests and
    bisecting), else every shipped slot."""
    raw = (os.environ if environ is None else environ).get(SLOTS_ENV)
    if raw is None:
        return frozenset(SHIPPED_SLOTS)
    return frozenset(name.strip() for name in raw.split(",")) - {""}


SLOTS = requested_slots()


def slot_reasons(slots: frozenset[str] = SLOTS) -> list[str]:
    """Why the requested slots cannot be served ([] when they can)."""
    known = (*SHIPPED_SLOTS, *OPT_IN_SLOTS)
    reasons = [
        f"{SLOTS_ENV}: unknown slot {name!r} (known: {', '.join(known) or 'none'})"
        for name in sorted(slots - set(known))
    ]
    reasons += [
        f"{SLOTS_ENV}: slot {slot!r} needs slot {NEEDS[slot]!r}"
        for slot in sorted(slots)
        if slot in NEEDS and NEEDS[slot] not in slots
    ]
    return reasons


def tolerances(slots: frozenset[str] = SLOTS) -> dict[str, float]:
    """The tolerances of the requested slots that are not bitwise."""
    out: dict[str, float] = {}
    for slot in sorted(slots - BITWISE_SLOTS):
        out.update(TOLERANCES.get(slot, {}))
    return out


def model_disable_reasons(model: Any) -> list[str]:
    """What the server arguments do not show: the loaded classes the executor
    follows, per rank."""
    from sglang.srt.models.minimax_m3 import MiniMaxM3DecoderLayer, MiniMaxM3Model

    m = getattr(model, "model", None)
    if not isinstance(m, MiniMaxM3Model):
        return [f"the language model is {type(m).__name__}, not MiniMaxM3Model"]
    reasons = []
    if getattr(model, "is_mrope_enabled", False):
        reasons.append("the text model uses mrope positions")
    if not (m.pp_group.is_first_rank and m.pp_group.is_last_rank):
        reasons.append("the model is split across pipeline ranks")
    if getattr(model, "logits_processor", None) is None:
        reasons.append("the model has no logits processor (encoder only)")
    other = [
        i
        for i in range(m.start_layer, m.end_layer)
        if not isinstance(m.layers[i], MiniMaxM3DecoderLayer)
    ]
    if other:
        reasons.append(f"layers {other} are not MiniMaxM3DecoderLayer")
    return reasons


def _unserved_reasons(bound: dict) -> str:
    """Why each package's slot serves no width, for the refusal of a requested slot
    that serves none: "built" when it serves, "not requested" when it is off."""
    moe, attention, hop = bound["moe"], bound["attention"], bound["boundary"]
    qk, ffn_hop = bound["qk_store"], bound["hop_ffn"]
    reasons = (
        ("moe_nofill", moe, getattr(moe, "reason", "")),
        ("attn_ll", attention, "; ".join(getattr(attention, "reasons", ()))),
        # why it refused each layer, when it takes none
        (
            "hop_mxfp8",
            hop,
            ""
            if getattr(hop, "hops", None)
            else "; ".join(sorted(set(getattr(hop, "refused", {}).values()))),
        ),
        ("qk_store", qk, getattr(qk, "reason", "")),
        (
            "hop_ffn",
            ffn_hop,
            ""
            if getattr(ffn_hop, "hops", None)
            else "; ".join(sorted(set(getattr(ffn_hop, "refused", {}).values()))),
        ),
    )
    return "; ".join(
        f"{slot}: {'not requested' if built is None else why or 'built'}"
        for slot, built, why in reasons
    )


def build(model: Any, ctx: Any):
    """The executor of the covered steps; ``ctx`` is SGLang's BuildContext."""
    from sgl_nvidia_mega.minimax_m31 import plan as slot_plan
    from sgl_nvidia_mega.minimax_m31.plan import MAX_REQUESTS, Slots, served_widths
    from sgl_nvidia_mega.minimax_m31.runtime.agree import agree_tp
    from sgl_nvidia_mega.minimax_m31.runtime.executor import MegakernelDecode

    widths = served_widths(tuple(ctx.widths), ctx.tokens_per_request)
    if not widths:
        raise RuntimeError(
            f"NVIDIA megakernel: no width of {list(ctx.widths)} has at most "
            f"{MAX_REQUESTS} requests"
        )
    from sgl_nvidia_mega.minimax_m31 import attention, boundary, moe
    from sgl_nvidia_mega.minimax_m31.attention import qk_store
    from sgl_nvidia_mega.minimax_m31.boundary import hop_ffn

    # each slot module's build result (None with its slots off) and the widths it serves
    bound: dict[str, Any] = {
        "boundary": boundary.build(model, widths, SLOTS),
        "hop_ffn": hop_ffn.build(model, widths, SLOTS),
        "attention": attention.build(model, widths, SLOTS),
        "qk_store": qk_store.build(model, widths, SLOTS),
        "moe": moe.build(model, widths, SLOTS),
    }
    # the widths each slot's build serves
    facts = slot_plan.Facts(
        moe_nofill=frozenset(w for w in widths if moe.serves(bound["moe"], w)),
        attn_ll=frozenset(w for w in widths if attention.serves(bound["attention"], w)),
        hop_mxfp8=frozenset(w for w in widths if boundary.serves(bound["boundary"], w)),
        qk_store=frozenset(w for w in widths if qk_store.serves(bound["qk_store"], w)),
        hop_ffn=frozenset(w for w in widths if hop_ffn.serves(bound["hop_ffn"], w)),
    )
    # every rank must launch the same kernels at each width, and each build decides
    # from its own rank (hop_mxfp8's build has no agreement of its own, hop_ffn's
    # agrees only on its NVFP4 probe)
    agree_tp(facts, what="slot plans (widths per slot)")
    plan = slot_plan.plan(widths, facts)
    served = {
        name: [w for w, s in plan.items() if getattr(s, name)]
        for name in Slots.__struct_fields__
    }
    missing = sorted(name for name in SLOTS if not served.get(name))
    if missing:
        raise RuntimeError(
            f"NVIDIA megakernel: requested slots {missing} serve no width of "
            f"{list(widths)} ({_unserved_reasons(bound)})"
        )
    summary = (
        f"slots {sorted(SLOTS) or 'none (stock modules)'} "
        f"({', '.join(f'{n} at {served[n]} rows' for n in sorted(SLOTS)) or 'stock'}); "
        f"{'verify' if ctx.tokens_per_request > 1 else 'decode'} steps of "
        f"{widths[0]}-{widths[-1]} rows ({ctx.tokens_per_request} a request)"
    )
    return MegakernelDecode(
        model,
        widths=widths,
        tokens_per_request=ctx.tokens_per_request,
        plan=plan,
        bound=bound,
        binding=ctx.binding,
        summary=summary,
    )
