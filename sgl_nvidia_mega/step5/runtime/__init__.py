# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel runtime: the slot registry (``SLOT_TABLE`` and the lists derived from
it), the test-only attention switch, the executor of a covered target-verify step
(``executor``), its per-step state (``state``), the embedding (``embedding``), the
test-only parity check (``parity``), and ``build``, which SGLang runs through the spec
once the model is loaded and before the KV cache is sized (so no pool or attention
backend exists yet)."""

from __future__ import annotations

import os
from typing import Any, Mapping

import msgspec

from sgl_nvidia_mega.step5.plan import MAX_ROWS


class SlotInfo(msgspec.Struct, frozen=True):
    """One slot: the seam it fills, its numerics class, its row bound, whether it is
    on by default, the slots it runs on, and (tolerance slots) the bounds its GPU test
    asserts."""

    seam: str  # the stock step function it replaces, e.g. "attention.core"
    numerics: str  # "bitwise" or "tolerance"
    max_rows: int  # the widest step its kernel serves (plan.MAX_ROWS at most)
    shipped: bool = False  # on when SGLANG_NVIDIA_MEGAKERNEL_SLOTS is unset
    needs: tuple[str, ...] = ()
    tolerances: Mapping[str, float] = {}

    def __post_init__(self):
        if self.numerics not in ("bitwise", "tolerance"):
            raise ValueError(f"slot numerics {self.numerics!r}: bitwise or tolerance")
        if (self.numerics == "tolerance") != bool(self.tolerances):
            raise ValueError(
                "a tolerance slot states its tolerances, a bitwise one none"
            )
        if not 0 < self.max_rows <= MAX_ROWS:
            raise ValueError(f"slot max_rows {self.max_rows} is outside 1-{MAX_ROWS}")


# One row per slot, each with its kernel test at every width it serves (eager and graph
# replay), its unit tests of where it serves and declines, and its numerics key. Both
# are opt-in until their gates pass (bitwise: served identity + parity; tolerance: the
# accuracy matrix against the switch-off arm).
SLOT_TABLE: dict[str, SlotInfo] = {
    # ffn/moe_quant_alt.py: the unchanged quantization kernel on the alternate stream
    "moe_quant_alt": SlotInfo(
        seam="ffn.experts", numerics="bitwise", max_rows=MAX_ROWS
    ),
    # attention/swa_attn.py: test/kernels/test_step5_swa_attn.py asserts these bounds
    # against an FP64 reference and against SGLang's FA4 call on the same inputs
    "swa_attn": SlotInfo(
        seam="attention.core",
        numerics="tolerance",
        max_rows=MAX_ROWS,
        tolerances={
            "swa_attn.head_ulp_vs_fp64": 1.5,
            "swa_attn.rel_fro_vs_fp64": 3e-3,
            "swa_attn.head_ulp_vs_fa4": 2.0,
            "swa_attn.rel_fro_vs_fa4": 4e-3,
        },
    ),
}

# The lists the harness and the other packages' conventions name (derived)
SHIPPED_SLOTS = tuple(n for n, s in SLOT_TABLE.items() if s.shipped)
OPT_IN_SLOTS = tuple(n for n, s in SLOT_TABLE.items() if not s.shipped)
NEEDS = {n: s.needs for n, s in SLOT_TABLE.items() if s.needs}
BITWISE_SLOTS = frozenset(n for n, s in SLOT_TABLE.items() if s.numerics == "bitwise")
TOLERANCES = {
    n: dict(s.tolerances) for n, s in SLOT_TABLE.items() if s.numerics == "tolerance"
}
SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"


def requested_slots() -> frozenset:
    """``SLOTS_ENV`` (comma-separated; "" runs the stock step), else the shipped
    slots."""
    raw = os.environ.get(SLOTS_ENV)
    if raw is None:
        return frozenset(SHIPPED_SLOTS)
    return frozenset(name.strip() for name in raw.split(",")) - {""}


SLOTS = requested_slots()

# Test only: SGLANG_NVIDIA_MEGAKERNEL_STEP5_ATTN=module runs each layer's attention as
# SGLang's module (layer.self_attn: Step5Attention.forward whole) instead of the
# attention step code, to bisect a parity failure of that code; it reads no Step-5
# view. Unset or empty runs the step code.
ATTN_ENV = "SGLANG_NVIDIA_MEGAKERNEL_STEP5_ATTN"
ATTN_MODE = os.environ.get(ATTN_ENV, "")


def slot_reasons(slots: frozenset | None = None) -> list[str]:
    """Why the requested slots (default: SLOTS) cannot be served ([] when they can)."""
    slots = SLOTS if slots is None else slots
    known = ", ".join(SLOT_TABLE) or "none"
    reasons = [
        f"{SLOTS_ENV}: unknown slot {name!r} (known: {known})"
        for name in sorted(slots - SLOT_TABLE.keys())
    ]
    for name in sorted(slots & SLOT_TABLE.keys()):
        missing = [n for n in SLOT_TABLE[name].needs if n not in slots]
        if missing:
            reasons.append(f"{SLOTS_ENV}: slot {name!r} runs on {missing}")
    return reasons


def attention_mode_reasons(slots: frozenset | None = None) -> list[str]:
    """Why ``ATTN_ENV`` cannot run with the requested slots ([] when it can)."""
    slots = SLOTS if slots is None else slots
    if ATTN_MODE not in ("", "module"):
        return [f"{ATTN_ENV} {ATTN_MODE!r} is not 'module' (unset runs the step code)"]
    if ATTN_MODE != "module":
        return []
    return [
        f"{ATTN_ENV}=module runs SGLang's attention module, which leaves out slot "
        f"{name!r} ({SLOT_TABLE[name].seam})"
        for name in sorted(slots & SLOT_TABLE.keys())
        if SLOT_TABLE[name].seam.startswith("attention.")
    ]


def bitwise(slots: frozenset | None = None) -> bool:
    """Whether every requested slot (default: SLOTS) is bitwise; an unknown name is
    slot_reasons' refusal."""
    slots = SLOTS if slots is None else slots
    return all(SLOT_TABLE[n].numerics == "bitwise" for n in slots if n in SLOT_TABLE)


def tolerances(slots: frozenset | None = None) -> dict[str, float]:
    """The union of the requested tolerance slots' bounds."""
    slots = SLOTS if slots is None else slots
    return {
        key: value
        for name in sorted(slots)
        if name in SLOT_TABLE
        for key, value in SLOT_TABLE[name].tolerances.items()
    }


# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # BuildContext's fields
    "kernels/plugin_kernels/megakernel/spec.py|srt/megakernels/spec.py": (
        "widths",
        "tokens_per_request",
        "binding",
    ),
    "srt/models/step3p7.py": ("Step3p7ForConditionalGeneration", "language_model"),
    "srt/models/step5.py": (
        "Step5ForCausalLM",
        "model",
        "layers",
        "start_layer",
        "end_layer",
        "layer_id",
        "use_dp_attention_reduce",
        "boundary_fusions",
        "self_attn",
        "is_dsa",
        "qkv_gate_weight",
        "_qkv_gate_layer",
        "alt_stream",
        "use_moe",
        "moe",
        "attn_dp_size",
        "use_mega_moe",
        "use_nvfp4_experts",
        "keep_bf16_experts",
        "bf16_trtllm",
        "kernel_routing",
        "fuse_shared_finalize",
    ),
    "srt/layers/step5_allreduce.py": ("allreduce_comm", "max_fused_rows"),
    "srt/layers/moe/utils.py": ("get_moe_runner_backend",),
}


def model_disable_reasons(model: Any) -> list[str]:
    """Why this rank's loaded model cannot run the stock step ([] when it can): the
    branches of SGLang's forward the step code mirrors, and no other."""
    from sglang.srt.models.step3p7 import Step3p7ForConditionalGeneration
    from sglang.srt.models.step5 import Step5ForCausalLM

    from sgl_nvidia_mega.step5 import BF16_EXPERT_LAYERS, FP8_EXPERT_LAYERS

    if not isinstance(model, Step3p7ForConditionalGeneration) or not isinstance(
        getattr(model, "language_model", None), Step5ForCausalLM
    ):
        return [
            f"{type(model).__name__} is not the Step-5 VL model with a Step-5 backbone"
        ]
    m = model.language_model.model
    reasons = []
    if (m.start_layer, m.end_layer, len(m.layers)) != (0, 92, 92):
        reasons.append(
            f"the model holds layers {m.start_layer}-{m.end_layer - 1}, not all 92"
        )
    for layer in m.layers:
        i, a = layer.layer_id, layer.self_attn
        if layer.use_dp_attention_reduce:
            reasons.append(f"layer {i}: attention DP (use_dp_attention_reduce)")
        if a.qkv_gate_weight is None or a._qkv_gate_layer is None:
            reasons.append(f"layer {i}: q/k/v and the gate are not one fused GEMM")
        if a.is_dsa and a.alt_stream is None:
            reasons.append(f"layer {i}: no alternate stream for the indexer")
        if not layer.use_moe:
            continue
        moe = layer.moe
        if moe.use_mega_moe or moe.use_nvfp4_experts or moe.attn_dp_size != 1:
            reasons.append(f"layer {i}: MegaMoE, NVFP4 experts or attention DP")
        if moe.alt_stream is None:
            reasons.append(f"layer {i}: no alternate stream for the shared expert")
        if i in FP8_EXPERT_LAYERS and not (
            moe.kernel_routing and moe.fuse_shared_finalize
        ):
            reasons.append(
                f"layer {i}: FP8 experts without in-kernel routing + deferred finalize"
            )
        if i in BF16_EXPERT_LAYERS and not (moe.keep_bf16_experts and moe.bf16_trtllm):
            reasons.append(f"layer {i}: BF16 experts not on FlashInfer TRT-LLM")
    return reasons


def summary(widths: tuple[int, ...], tokens_per_request: int, bound: dict) -> str:
    """The startup line's Step-5 part: the slots, the covered rows and the boundary
    tier SGLang built. Without the Step-5 multicast communicator SGLang's whole fused
    tier is off: every boundary runs FlashInfer's all-reduce fusion and the FP8 MoE
    finalizes inside the FFN (step5_allreduce.py: _get_comm, can_defer_finalize)."""
    rows = (
        f"{widths[0]}-{widths[-1]} (batch 1-{widths[-1] // tokens_per_request})"
        if widths
        else "none"
    )
    if bound["boundary.multicast"]:
        tier = (
            f"boundaries Step-5 multicast at <= {bound['boundary.fused_rows']} rows; "
            "experts FP8 3-87 deferred finalize, BF16 88-90"
        )
    else:
        tier = (
            "FlashInfer boundaries, no multicast communicator (finalize in the FFN); "
            "experts FP8 3-87, BF16 88-90"
        )
    slots = ", ".join(sorted(SLOTS)) or "none (stock)"
    text = f"slots {slots}; verify rows {rows}; {tier}"
    if ATTN_MODE == "module":
        text += f"; attention Step5Attention.forward whole ({ATTN_ENV}=module)"
    return text


def build(model: Any, ctx: Any):
    """Every package's build, the slot plan per width and the executor; ``ctx`` is
    SGLang's BuildContext (widths of 4-row multiples up to the spec's max_rows)."""
    from sglang.srt.layers.moe.utils import get_moe_runner_backend

    from sgl_nvidia_mega.step5 import plan as slot_plan
    from sgl_nvidia_mega.step5.runtime import parity
    from sgl_nvidia_mega.step5.runtime.executor import Step5Verify

    if ctx.tokens_per_request != slot_plan.TOKENS_PER_REQUEST:
        raise RuntimeError(
            f"NVIDIA megakernel: Step-5 is built for {slot_plan.TOKENS_PER_REQUEST} "
            f"verify rows a request, not {ctx.tokens_per_request}"
        )
    widths = slot_plan.covered(tuple(ctx.widths), ctx.tokens_per_request)
    fusions = model.language_model.model.layers[0].boundary_fusions
    bound = {
        # Step5MoE.forward re-reads this per call; it is fixed for the process
        "ffn.runner_defers": get_moe_runner_backend().value
        in ("flashinfer_trtllm_routed", "flashinfer_trtllm"),
        # the fused boundary tier SGLang built: 32 rows with the multicast
        # communicator; without it every boundary is FlashInfer's
        "boundary.fused_rows": fusions.max_fused_rows,
        "boundary.multicast": fusions.allreduce_comm is not None,
    }
    from sgl_nvidia_mega.step5.attention import swa_attn
    from sgl_nvidia_mega.step5.ffn import moe_quant_alt

    # each requested slot's own checks of the loaded model; any reason stops every
    # rank alike, before a kernel is built (no silent fallback to the stock code)
    reasons = []
    if "moe_quant_alt" in SLOTS:
        reasons += moe_quant_alt.unsupported(model, bound)
    if "swa_attn" in SLOTS:
        reasons += swa_attn.unsupported(model)
    if reasons:
        raise RuntimeError("NVIDIA megakernel: " + "; ".join(reasons))
    if "swa_attn" in SLOTS:
        bound["swa_attn.kernels"] = swa_attn.build(widths)
    facts = slot_plan.Facts(
        moe_quant_alt="moe_quant_alt" in SLOTS, swa_attn="swa_attn" in SLOTS
    )
    slots = slot_plan.plan(widths, facts)
    # a requested slot that serves no width stops every rank alike
    built = {
        name
        for width_slots in slots.values()
        for name, on in msgspec.structs.asdict(width_slots).items()
        if on
    }
    missing = sorted(SLOTS - built)
    if missing:
        raise RuntimeError(
            f"NVIDIA megakernel: requested slots {missing} built no width"
        )
    return Step5Verify(
        model,
        slots=slots,
        bound=bound,
        widths=widths,
        binding=ctx.binding,
        summary=summary(widths, ctx.tokens_per_request, bound),
        parity=parity.Parity() if parity.ENABLED else None,
    )
