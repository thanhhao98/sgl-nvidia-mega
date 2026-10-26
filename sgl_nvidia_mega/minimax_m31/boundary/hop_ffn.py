# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# This file has been modified from its original version (the configuration FlashInfer's
# per-token NVFP4 quantize reads, and the build's probe rows for its arithmetic).
# Adapted from FlashInfer's flashinfer/quantization/kernels/nvfp4_quantize.py
# (nvfp4_quantize_per_token_cute_dsl, NVFP4QuantizePerTokenKernel;
# flashinfer-ai/flashinfer@946200de, v0.7.0.post1).
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/layer_boundary
# (StageBoundary.prepare of an FFN input: _prepare_input, _prepare, _run_entry,
# _reduce_update_read; CustomAllReduceFusion._ffn_input_add_norm) and
# python/sglang/srt/models/minimax_m3.py (MiniMaxM3MLP.forward,
# MiniMaxM3MoE._forward_shared_experts).
"""Slot ``hop_ffn`` (bitwise): the FFN-input boundary also writes the FFN's quantized
inputs.

Stock, every layer's FFN input is SGLang's push-plane kernel (the plain all-reduce of
the attention's partial sum + residual + Gemma norm,
``CustomAllReduceFusion._ffn_input_add_norm``), and the FFN then quantizes the bf16
row it wrote: the gate_up of the dense MLP (layers 0-2) and of a MoE layer's shared
expert each launch FlashInfer's MXFP8 quantize, and a MoE layer's routed experts
FlashInfer's per-token NVFP4 quantize (slot moe_nofill: into its buffers).

With the slot the boundary is that kernel in its quantizing forms
(``all_reduce_residual_norm_quant[_nvfp4]``): the same bf16 normed row and updated
residual, bit for bit, plus the normed row's MXFP8 pair (every layer) and, for a MoE
layer, its per-token NVFP4, each as FlashInfer's quantize writes it. ``ffn_input``
returns them as an ``FfnInput``. ``moe.step`` hands the pair to the gate_up of the
dense MLP or the shared expert (``Fp8LinearMethod`` takes an ``Mxfp8SwizzledInput`` as
it is, without its quantize; ``mlp_input``) and the NVFP4 to slot moe_nofill's routed
experts in place of their quantize (``moe_nofill.Prequantized``); the router and
everything else read the bf16 row. Everything else ``ffn_boundary.prepare`` does is
done the same way. A layer whose stock FFN input does not run the push-plane kernel,
or whose FFN does not take the inputs as they are (checked at build), and a step that
misses one of the kernel's preconditions, run the stock ``prepare``; the first step of
each width logs the layers it left to it and why (``HopFfn.report``).

The NVFP4 is FlashInfer's NVFP4QuantizePerTokenKernel in the configuration SGLang's
routed MoE runs it in (fast math, no 4over6, linear scale layout), redone instruction
for instruction in the kernel's epilogue (all_reduce_fusion.cuh, ``nvfp4_quant_row``),
which exchanges the row's amax across the cluster that holds the row, as the norm
does its sum of squares. The build runs that epilogue on probe rows against
FlashInfer's own kernel on this GPU (``probe_nvfp4``), at the served global scale and
its f32 neighbours a few ulps either side: rows on which the two subnormal modes of
the few f32 operations FlashInfer leaves to CuTe-DSL differ (the probe settles the
mode), and rows on e4m3 rounding midpoints at which another association of the block
scale's three multiplies, or a one-ulp change in rcp(6), would change some byte
(whichever value rcp.approx returns within an ulp, as a model of the arithmetic
shows). Without a bit-for-bit match on every rank, or under another FlashInfer
configuration (also checked at each step), a MoE layer takes the MXFP8 pair only and
its routed experts quantize as before.

Buffers: the MXFP8 pair (codes [rows, hidden], one 128-row tile of 128x4-swizzled
scales) and the NVFP4 triple (codes [rows, hidden / 2], linear E4M3 scales, per-token
scales) are sized for the largest width (at most 128 rows) and sliced per step; the
bf16 normed row and the updated residual are allocated per step, as the stock kernel's
are. Reusing them is safe:

- the MXFP8 pair is written by layer L's boundary kernel and read only by layer L's
  gate_up GEMM, both on the main stream (the GEMM is the kernel's PDL dependent and
  waits for it before reading). The next writer, layer L + 1's boundary kernel, starts
  after the GEMM has finished: layer L + 1's MSA indexer and top-k run between them,
  launched without PDL, also where SGLang launches its MSA-v1 dequant kernels with PDL
  (each waits for every earlier kernel on the stream; ``boundary.HopMxfp8``);
- the NVFP4 triple is written by layer L's boundary kernel on the main stream and read
  by layer L's routed experts on ``mlp.alt_stream`` after its ``wait_stream`` on the
  main stream (an event: the kernel has completed). Its readers (the routing and GEMM1
  of FlashInfer's routed MoE) finish before the routed chain's last kernel, which
  layer L's join (``current_stream.wait_stream(alt_stream)``) waits for, and the join
  precedes layer L + 1's boundary kernel on the main stream. A captured graph keeps
  both joins as full dependencies; eager steps (the parity test) run every launch on
  one stream, in order;
- the decode graphs of different widths, which share them, never run at the same
  time, and nothing else touches them (slot moe_nofill keeps its own buffers).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from functools import partial
from typing import Any, NamedTuple, Optional

import torch

from sgl_nvidia_mega.minimax_m31 import TEXT_CONFIG
from sgl_nvidia_mega.minimax_m31.moe import moe_nofill

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/minimax_m3.py": (
        "MiniMaxM3MLP",
        "MiniMaxM3MoE",
        "ffn_boundary",
        "is_layer_sparse",
        "mlp",
        "gate_up_proj",
        "shared_experts",
        "num_fused_shared_experts",
        "_forward_shared_experts",
        "forward",
        "start_layer",
        "end_layer",
        "layers",
    ),
    "srt/models/minimax_m3_vl.py": ("model",),
    "srt/layers/layer_boundary/contracts.py": ("BatchVariant", "ORDINARY"),
    "srt/layers/layer_boundary/layout.py": ("SumGroup", "ATTN_TP"),
    "srt/layers/layer_boundary/fusions/custom_all_reduce_fusion.py": (
        "CustomAllReduceFusion",
        "_bf16_norm_gamma",
        "hidden_size",
        "_ffn_input_add_norm",
        "_eligible",
    ),
    # the FFN input's ordinary entry: _run_entry with the _reduce_update_read step
    # whose first fusion is the push-plane kernel (ffn_hop reads their keywords)
    "srt/layers/layer_boundary/prepare.py": (
        "_reduce_update_read",
        "_run_entry",
        "_run_entry(step)",
        "_run_entry(is_plain_add)",
        "_run_entry(expected_sum)",
        "_run_entry(written_step)",
        "_reduce_update_read(fusions)",
        "_reduce_update_read(gathers_residual)",
        "_reduce_update_read(read_fusions)",
        "_reduce_update_read(read_fusion_indices)",
        "_reduce_update_read(read)",
        "_reduce_update_read(group)",
    ),
    # a read bound per batch (ffn_hop declines it)
    "srt/layers/layer_boundary/residual/__init__.py": ("bind",),
    "srt/layers/layer_boundary/*.py": (
        "plan",
        "fusions",
        "norm",
        "_publish_lora_layout",
        "paths",
        "entry",
        "prepare",
        "input_move",
        "attn_input_adapter",
        "declared_sum",
    ),
    # pending.update: the stream's ResidualUpdate (PLAIN_ADD in this model)
    "srt/layers/layer_boundary/residual/__init__.py": ("is_plain_add",),
    "srt/layers/layer_boundary/residual/stream.py": (
        "DeclaredSum",
        "group",
        "pending",
        "residual",
        "check",
        "write",
        "owed",
        "update",
        "value",
    ),
    "srt/model_executor/forward_batch_info.py": ("residual_stream",),
    "srt/layers/linear.py": (
        "ColumnParallelLinear",
        "forward",
        "quant_method",
        "weight",
    ),
    "srt/layers/quantization/fp8.py": (
        "Fp8LinearMethod",
        "apply",
        "use_mxfp8",
        "block_fp8_as_mxfp8",
        "w8a8_mxfp8_linear",
    ),
    "srt/layers/quantization/fp8_utils.py": ("flashinfer_mxfp8_blockscaled_linear",),
    "srt/layers/quantization/mxfp8_input.py": (
        "accepts_mxfp8_swizzled_input",
        "Mxfp8SwizzledInput",
        # the linear's fields accepts_mxfp8_swizzled_input reads (the gate_up's)
        '"quant_method"',
        '"block_fp8_mxfp8_ready"',
    ),
    "srt/runtime_context.py": (
        "get_forward",
        "get_parallel",
        '"sp_active"',
        '"tp_size"',
    ),
    "srt/layers/layernorm.py": ("variance_epsilon", '"gemma_weight"'),
    # the boundary kernel's quantizing forms, its NVFP4 epilogue on rows from memory
    # (the build's probe) and their builds
    "kernels/ops/communication/all_reduce_fusion.py": (
        "all_reduce_residual_norm_quant",
        "all_reduce_residual_norm_quant(world_size)",
        "all_reduce_residual_norm_quant(residual_out)",
        "all_reduce_residual_norm_quant(quantized)",
        "all_reduce_residual_norm_quant(scales)",
        "all_reduce_residual_norm_quant_nvfp4",
        "all_reduce_residual_norm_quant_nvfp4(world_size)",
        "all_reduce_residual_norm_quant_nvfp4(ftz)",
        "all_reduce_residual_norm_quant_nvfp4(fp4)",
        "all_reduce_residual_norm_quant_nvfp4(fp4_scales)",
        "all_reduce_residual_norm_quant_nvfp4(token_scales)",
        "all_reduce_residual_norm_quant_nvfp4(residual_out)",
        "all_reduce_residual_norm_quant_nvfp4(quantized)",
        "all_reduce_residual_norm_quant_nvfp4(scales)",
        "nvfp4_row_quant",
        "nvfp4_row_quant(ftz)",
        "nvfp4_row_quant(cluster_size)",
        "nvfp4_row_quant(vecs_per_thread)",
        "NVFP4_SF_VEC",
        "mxfp8_scale_bytes",
        "MXFP8_QUANT_MAX_TOKENS",
        "load_residual_norm_quant",
        "load_residual_norm_quant_nvfp4",
        "load_residual_norm_quant_nvfp4(ftz)",
        "_norm_launch",
    ),
}

logger = logging.getLogger(__name__)

SLOT = "hop_ffn"
HIDDEN = TEXT_CONFIG["hidden_size"]
# st.scratch[TALLY]: a step's outcome at each layer slot hop_ffn was built for, TAKEN
# or why it left the layer to the stock prepare
TALLY = "hop_ffn"
TAKEN = "taken"
# The NVFP4 outputs, in FlashInfer's order
NVFP4_OUTPUTS = ("fp4 codes", "block scales", "per-token scales")
# The inverse global scales the probe runs at, in f32 ulps from the served one
# (``probe_scales``)
PROBE_SCALE_ULPS = (0, -1, 1, -2, 2, -3, 3)
# ``_midpoint_rows``: a row's amax is e4m3_max * c, so its per-token scale is c / 6 up
# to the f32 roundings, never a power of two (c = 3/2 would make it 1/4)
MIDPOINT_ROW_C = (1.0, 17 / 16, 9 / 8, 5 / 4, 11 / 8, 13 / 8, 7 / 4, 15 / 8)


class FfnInput(NamedTuple):
    """The FFN input slot hop_ffn's boundary wrote: the stock bf16 normed rows ``x``
    (the router reads them, and any stock call), their MXFP8 pair for the gate_up of
    the dense MLP or shared expert, and for a MoE layer their per-token NVFP4 as
    ``moe_nofill.PerTokenQuant`` returns it (``hs_fp4 [m, K / 2]`` uint8, ``hs_scale
    [m, K / 16]`` float8_e4m3fn, ``per_token_scale [m]`` fp32), else None."""

    x: torch.Tensor
    mxfp8: Any
    nvfp4: Optional[tuple]


class _FfnHop(NamedTuple):
    """A layer whose FFN input the slot takes: the layer, its FFN stage's plan, the
    plan's ordinary batch path, its push-plane fusion, and whether the FFN is a MoE."""

    layer: Any
    plan: Any
    entry: Any
    fusion: Any
    moe: bool


def ffn_hop(layer, hidden: int = HIDDEN) -> tuple[Optional[_FfnHop], Optional[str]]:
    """``(hop, None)`` when ``layer``'s stock FFN input completes the attention's
    partial sum with the push-plane kernel (CustomAllReduceFusion, tried first, for an
    ordinary batch), else ``(None, why not)``."""
    from sglang.srt.layers.layer_boundary.contracts import BatchVariant
    from sglang.srt.layers.layer_boundary.fusions.custom_all_reduce_fusion import (
        CustomAllReduceFusion,
        _bf16_norm_gamma,
    )
    from sglang.srt.layers.layer_boundary.layout import SumGroup
    from sglang.srt.layers.layer_boundary.prepare import (
        _reduce_update_read,
        _run_entry,
    )

    plan = getattr(getattr(layer, "ffn_boundary", None), "plan", None)
    if plan is None:
        return None, "no FFN stage boundary"
    fusion = plan.fusions
    if not isinstance(fusion, CustomAllReduceFusion) or fusion.hidden_size != hidden:
        return None, f"no push-plane fusion of hidden {hidden}"
    if _bf16_norm_gamma(plan.norm) is None:
        return None, "the post-attention norm has no bf16 gamma"
    if plan._publish_lora_layout:
        return None, "the FFN input publishes LoRA rows"
    entry = getattr(plan.paths.get(BatchVariant.ORDINARY), "entry", None)
    if entry is None:
        return None, "no ordinary batch path"
    if (
        entry.input_move is not None
        or entry.attn_input_adapter is not None
        or entry.declared_sum is not SumGroup.ATTN_TP
    ):
        return None, "the FFN input moves rows or does not owe the attention-TP sum"
    prepare = entry.prepare
    step = (
        prepare.keywords.get("step")
        if isinstance(prepare, partial)
        and prepare.func is _run_entry
        and prepare.keywords.get("is_plain_add") is True
        and prepare.keywords.get("expected_sum") is SumGroup.ATTN_TP
        and prepare.keywords.get("written_step") is None
        else None
    )
    fused = (
        step.keywords.get("fusions", ())
        if isinstance(step, partial)
        and step.func is _reduce_update_read
        and not step.keywords.get("gathers_residual")
        and not step.keywords.get("read_fusions")
        and not step.keywords.get("read_fusion_indices")
        and getattr(step.keywords.get("read"), "bind", None) is None
        and step.keywords.get("group") is SumGroup.ATTN_TP
        else ()
    )
    first = fused[0] if fused else None
    if not (
        isinstance(first, partial)
        and first.func == fusion._ffn_input_add_norm
        and first.args == (plan,)
        and not first.keywords
    ):
        return (
            None,
            "the push-plane kernel is not the FFN input's first fused completion",
        )
    moe = bool(getattr(layer, "is_layer_sparse", False))
    return _FfnHop(layer, plan, entry, fusion, moe), None


def mlp_reasons(mlp, hidden: int = HIDDEN) -> list[str]:
    """Why ``mlp`` (a dense MLP or a shared expert) does not hand its input straight to
    a gate_up that takes the MXFP8 pair as it is ([] when it does): MiniMaxM3MLP's
    forward, no hooks, and a column-parallel MXFP8 linear on FlashInfer's MXFP8 GEMM
    (``Fp8LinearMethod.apply`` passes the pair to ``flashinfer_mxfp8_blockscaled_linear``
    as its quantized input)."""
    from sglang.srt.layers.linear import ColumnParallelLinear
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )
    from sglang.srt.layers.quantization.mxfp8_input import (
        accepts_mxfp8_swizzled_input,
    )
    from sglang.srt.models.minimax_m3 import MiniMaxM3MLP

    if not isinstance(mlp, MiniMaxM3MLP):
        return [f"the MLP is a {type(mlp).__name__}, not MiniMaxM3MLP"]
    reasons = moe_nofill._overridden(mlp, MiniMaxM3MLP, ("forward",))
    if why := moe_nofill.call_reason(mlp):
        reasons.append(why)
    gate_up = getattr(mlp, "gate_up_proj", None)
    if not isinstance(gate_up, ColumnParallelLinear):
        return reasons + [f"the gate_up is a {type(gate_up).__name__}"]
    reasons += moe_nofill._overridden(gate_up, ColumnParallelLinear, ("forward",))
    if why := moe_nofill.call_reason(gate_up):
        reasons.append(f"gate_up: {why}")
    qm = getattr(gate_up, "quant_method", None)
    if not isinstance(qm, Fp8LinearMethod):
        return reasons + [
            f"the gate_up's quant method is a {type(qm).__name__}, not Fp8LinearMethod"
        ]
    reasons += [
        f"the gate_up's quant method: {why}"
        for why in moe_nofill._overridden(qm, Fp8LinearMethod, ("apply",))
    ]
    if not (
        getattr(qm, "use_mxfp8", False)
        and not getattr(qm, "block_fp8_as_mxfp8", False)
        and accepts_mxfp8_swizzled_input(gate_up)
    ):
        reasons.append("the gate_up takes no MXFP8 128x4 input")
    fn = getattr(qm, "w8a8_mxfp8_linear", None)
    if not (isinstance(fn, partial) and fn.func is flashinfer_mxfp8_blockscaled_linear):
        reasons.append("the gate_up's MXFP8 linear is not FlashInfer's")
    weight = getattr(gate_up, "weight", None)
    if weight is None or weight.shape[-1] != hidden:
        reasons.append(f"the gate_up does not take {hidden} columns")
    return reasons


def consumer_reasons(layer, hidden: int = HIDDEN) -> list[str]:
    """Why ``layer``'s FFN does not take the inputs the slot writes as they are ([]
    when it does): the dense MLP's gate_up the MXFP8 pair; a MoE's shared expert the
    MXFP8 pair and its routed experts slot moe_nofill's chain (moe_nofill.layer_reasons)
    with the NVFP4."""
    from sglang.srt.models.minimax_m3 import MiniMaxM3MoE

    mlp = getattr(layer, "mlp", None)
    if not getattr(layer, "is_layer_sparse", False):
        return [f"dense MLP: {why}" for why in mlp_reasons(mlp, hidden)]
    reasons = moe_nofill.layer_reasons(layer)
    if not isinstance(mlp, MiniMaxM3MoE):
        return reasons
    reasons += moe_nofill._overridden(mlp, MiniMaxM3MoE, ("_forward_shared_experts",))
    if getattr(mlp, "shared_experts", None) is None or mlp.num_fused_shared_experts:
        return reasons + ["the MoE has no separate shared expert"]
    return reasons + [
        f"shared expert: {why}" for why in mlp_reasons(mlp.shared_experts, hidden)
    ]


def layer_hop(layer, hidden: int = HIDDEN) -> tuple[Optional[_FfnHop], Optional[str]]:
    """``ffn_hop`` for a layer whose FFN also takes the slot's inputs
    (``consumer_reasons``)."""
    hop, why = ffn_hop(layer, hidden)
    if hop is None:
        return None, why
    reasons = consumer_reasons(layer, hidden)
    if reasons:
        return None, "; ".join(reasons)
    return hop, None


_WARNED: set = set()


def mlp_input(mlp, prepared: FfnInput):
    """What ``mlp`` (a dense MLP or a shared expert the build checked, ``mlp_reasons``)
    takes from the slot's FFN input at this step: the MXFP8 pair, or the bf16 rows when
    its gate_up would not hand the pair to the GEMM as it is (sequence parallelism,
    hooks added since the build), logged once per reason. Host state only."""
    from sglang.srt.runtime_context import get_forward

    why = ""
    if get_forward().sp_active:
        why = "sequence parallelism is active"
    elif reason := (
        moe_nofill.call_reason(mlp) or moe_nofill.call_reason(mlp.gate_up_proj)
    ):
        why = reason
    if not why:
        return prepared.mxfp8
    if why not in _WARNED:
        _WARNED.add(why)
        logger.warning(
            "NVIDIA megakernel slot %s: a gate_up quantizes the bf16 rows itself "
            "(and any other for the same reason): %s",
            SLOT,
            why,
        )
    return prepared.x


def config_reason() -> str:
    """Why FlashInfer's per-token NVFP4 quantize would not run the configuration the
    kernel's epilogue reproduces ("" when it would): fast math, no 4over6; read as
    ``nvfp4_quantize_per_token_cute_dsl`` reads it, at each call."""
    try:
        from flashinfer.quantization.kernels import nvfp4_quantize as nq
    except ImportError as e:
        return f"FlashInfer's CuTe-DSL NVFP4 quantize does not import: {e}"
    if nq._env_flag_enabled("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"):
        return "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH selects the exact-math quantize"
    if nq.resolve_nvfp4_4over6() is not None:
        return "FLASHINFER_NVFP4_4OVER6 selects the 4over6 quantize"
    return ""


def _e4m3_values_and_midpoints(e4m3_max: float):
    """The positive finite e4m3 values up to ``e4m3_max``, and the midpoints between
    neighbouring ones (and between 0 and the smallest) below it, ascending."""
    e4m3 = torch.arange(1, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    mids = torch.cat((e4m3[:1] / 2, (e4m3[:-1] + e4m3[1:]) / 2))
    return e4m3[e4m3 <= e4m3_max], mids[mids < e4m3_max]


def _tie_rows(hidden: int, e4m3_max: float = 448.0) -> torch.Tensor:
    """Rows whose scales and codes land on rounding ties at a power-of-two encode
    scale: with the row's amax 6 * e4m3_max, the per-token scale at the served global
    scale is 1 (a power of two at the other magnitudes), so a block's scale is its
    max times rcp(6) alone, and a one-ulp difference in rcp(6) moves a byte: blocks
    whose max is 6x each midpoint between neighbouring e4m3 values (the block scale
    is that midpoint, up to an ulp), at three magnitudes; and blocks of max 6x an
    e4m3 value holding that value times each midpoint between neighbouring e2m1
    values. A power-of-two encode scale multiplies exactly, so at the served global
    scale the order of the multiplies cannot show here; it does on
    ``_midpoint_rows``."""
    e4m3, mids = _e4m3_values_and_midpoints(e4m3_max)
    blocks = hidden // 16
    rows = []
    for magnitude in (1.0, 2.0**-20, 2.0**10):
        x = torch.zeros(blocks, 16)
        x[0, 0] = 6.0 * e4m3_max
        for b in range(1, blocks):
            mid = mids[(b - 1) % len(mids)]
            # past the first pass, just off the ties
            x[b, 0] = 6 * mid * (1 + ((b - 1) // len(mids)) * 2.0**-7)
            x[b, 1] = -3 * mid
        rows.append(x.reshape(-1) * magnitude)
    e2m1_mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    x = torch.zeros(blocks, 16)
    x[0, 0] = 6.0 * e4m3_max
    for b in range(1, blocks):
        value = e4m3[(7 * b) % len(e4m3)]
        x[b, 0] = 6 * value
        x[b, 1:8] = value * e2m1_mids
        x[b, 8:15] = -value * e2m1_mids
    rows.append(x.reshape(-1))
    return torch.stack(rows)


def _midpoint_rows(hidden: int, e4m3_max: float = 448.0) -> torch.Tensor:
    """Rows on which the order of the block scale's multiplies shows: a row of amax
    ``e4m3_max * c`` (``MIDPOINT_ROW_C``) has blocks of max ``m * c`` for each midpoint
    ``m`` between neighbouring e4m3 values for which that is a bf16 value, so that the
    block scale ``encode * (block_max * rcp(6))`` is ``m`` in exact arithmetic at an
    encode scale near ``6 / c``, and its f32 roundings (of the per-token scale, the
    encode scale, rcp(6) and the two multiplies) decide each byte. At the probe's
    global scales (``probe_scales``), which move those blocks across their midpoints
    by about an ulp a step, another association of the multiplies or a one-ulp change
    in rcp(6) changes some byte, whichever way rcp.approx rounds."""
    _, mids = _e4m3_values_and_midpoints(e4m3_max)
    mids = mids.double()
    blocks = hidden // 16
    rows = []
    for c in MIDPOINT_ROW_C:
        maxima = mids * c
        maxima = maxima[maxima.to(torch.bfloat16).double() == maxima][: blocks - 1]
        x = torch.zeros(blocks, 16, dtype=torch.float64)
        x[0, 0] = e4m3_max * c
        x[1 : 1 + len(maxima), 0] = maxima
        x[1 : 1 + len(maxima), 1] = -maxima / 2
        rows.append(x.reshape(-1))
    return torch.stack(rows).float()


def probe_rows(
    hidden: int = HIDDEN, device="cuda", e4m3_max: float = 448.0
) -> torch.Tensor:
    """bf16 rows for ``probe_nvfp4`` at the global scale of ``e4m3_max``:
    ``moe_nofill.check_rows``' (Gaussian rows of magnitudes 1e-3 to 1e3, a zero row, a
    constant row, an outlier, tiny values); rows whose 16-value blocks span
    2^-40..2^40; rows on rounding ties at a power-of-two encode scale (``_tie_rows``)
    and at others (``_midpoint_rows``); and last, rows where the two subnormal modes
    differ: an amax near 1e-36 (a subnormal per-token scale), an amax of 1e-33 with a
    block of max 5e-38 (a subnormal block max / 6 that still makes a nonzero e4m3
    scale), and bf16 subnormals (an amax that compares equal to zero when
    flushed)."""
    g = torch.Generator(device="cpu").manual_seed(47)
    rows = [moe_nofill.check_rows(8, hidden, dtype=torch.float32, device="cpu")]
    blocks = hidden // 16
    spread = torch.exp2((torch.arange(blocks) % 81 - 40).float())
    rows.append(torch.randn(4, hidden, generator=g) * spread.repeat_interleave(16))
    rows.append(_tie_rows(hidden, e4m3_max))
    rows.append(_midpoint_rows(hidden, e4m3_max))
    subnormal_scale = torch.randn(hidden, generator=g) * 1e-36
    block_window = torch.randn(hidden, generator=g) * 1e-34
    block_window[7] = 1e-33
    block_window[160:176] = torch.linspace(-5e-38, 5e-38, 16)
    bf16_subnormals = torch.randn(hidden, generator=g) * 1e-39
    rows.append(torch.stack((subnormal_scale, block_window, bf16_subnormals)))
    return torch.cat(rows).to(torch.bfloat16).to(device).contiguous()


def probe_scales(e4m3_max: float, device="cuda") -> list[tuple[int, torch.Tensor]]:
    """``(ulps, [1] fp32)`` for each of ``PROBE_SCALE_ULPS``: the inverse global scale
    SGLang's routed MoE passes (``1 / (e4m3_max * 6)``, rounded as its torch.full
    rounds it) and its f32 neighbours. Both kernels take it at run time and do the
    same arithmetic at any value; off the served one only the roundings move, so
    that the probe rows' tie blocks cross their midpoints whichever way rcp.approx
    rounds (``_midpoint_rows``)."""
    served = torch.full((1,), 1.0 / (e4m3_max * 6.0), dtype=torch.float32)
    scales = []
    for ulps in PROBE_SCALE_ULPS:
        scale = served
        toward = torch.full_like(served, math.copysign(math.inf, ulps))
        for _ in range(abs(ulps)):
            scale = torch.nextafter(scale, toward)
        scales.append((ulps, scale.to(device)))
    return scales


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1).view(torch.uint8)


def probe_nvfp4(
    geometries, hidden: int = HIDDEN, device="cuda"
) -> tuple[Optional[bool], str]:
    """``(ftz, "")``: the subnormal mode under which the kernel's NVFP4 epilogue
    (``nvfp4_row_quant``, at each ``(cluster_size, vecs_per_thread)`` of
    ``geometries``) writes FlashInfer's ``nvfp4_quantize`` (as SGLang's routed MoE
    calls it) bit for bit on ``probe_rows`` at every global scale of
    ``probe_scales``; ``(None, why not)`` when neither does or FlashInfer runs another
    configuration. Launches on this GPU (at build, outside any graph; it compiles
    both kernels)."""
    from flashinfer import SfLayout, nvfp4_quantize
    from sglang.kernels.ops.communication.all_reduce_fusion import nvfp4_row_quant

    why = config_reason()
    if why:
        return None, why
    e4m3_max = moe_nofill.e4m3_max()
    x = probe_rows(hidden, device, e4m3_max)
    differ: dict[bool, list[str]] = {False: [], True: []}
    for ulps, scale in probe_scales(e4m3_max, device):
        want = nvfp4_quantize(
            x,
            scale,
            sfLayout=SfLayout.layout_linear,
            per_token_activation=True,
            backend="cute-dsl",
        )
        at = f", global scale {ulps:+d} ulps" if ulps else ""
        for ftz in (False, True):
            for cluster_size, vecs in geometries:
                got = nvfp4_row_quant(
                    x, scale, ftz=ftz, cluster_size=cluster_size, vecs_per_thread=vecs
                )
                differ[ftz] += [
                    f"{name} at cluster {cluster_size} x {vecs} vectors{at}"
                    for name, a, b in zip(NVFP4_OUTPUTS, got, want)
                    if a.numel() * a.element_size() != b.numel() * b.element_size()
                    or not torch.equal(_bytes(a), _bytes(b))
                ]
    for ftz in (False, True):
        if not differ[ftz]:
            return ftz, ""
    return None, (
        "the kernel's NVFP4 differs from FlashInfer's per-token quantize on the probe "
        f"rows (subnormals kept: {', '.join(differ[False][:3])}; flushed: "
        f"{', '.join(differ[True][:3])})"
    )


class HopFfn:
    """Slot hop_ffn's build: the layers it takes, its buffers, and whether a MoE
    layer's boundary also writes the NVFP4 (``ftz``: the epilogue's subnormal mode;
    None: it does not, ``nvfp4_off`` says why)."""

    def __init__(
        self,
        *,
        widths: tuple[int, ...],
        hops: dict,
        refused: dict,
        hidden: int,
        device,
        ftz: Optional[bool] = None,
        nvfp4_off: str = "",
    ):
        from sglang.kernels.ops.communication.all_reduce_fusion import (
            NVFP4_SF_VEC,
            mxfp8_scale_bytes,
        )

        self.widths = tuple(sorted(widths))
        self.hops = hops  # layer index -> _FfnHop
        self.refused = refused  # layer index -> why the slot does not take it
        self.hidden = hidden
        self.max_rows = self.widths[-1] if self.widths else 0
        self.ftz = ftz
        self.nvfp4_off = nvfp4_off
        self.sf_cols = hidden // NVFP4_SF_VEC
        rows = self.max_rows
        self.data = torch.empty(rows, hidden, dtype=torch.float8_e4m3fn, device=device)
        self.scales = torch.empty(
            mxfp8_scale_bytes(hidden), dtype=torch.uint8, device=device
        )
        nvfp4_rows = rows if self.nvfp4 else 0
        self.fp4 = torch.empty(
            nvfp4_rows, hidden // 2, dtype=torch.uint8, device=device
        )
        self.fp4_scales = torch.empty(
            nvfp4_rows * self.sf_cols, dtype=torch.uint8, device=device
        )
        self.token_scales = torch.empty(nvfp4_rows, dtype=torch.float32, device=device)
        # the stock chain's torch.full of 1 / (e4m3_max * 6), once for each e4m3_max
        self.global_scale_inv = {
            e: torch.full((1,), 1.0 / (e * 6.0), dtype=torch.float32, device=device)
            for e in moe_nofill.E4M3_MAXES
        }
        # width -> the layers its first step took (report)
        self.reported: dict[int, int] = {}

    @property
    def nvfp4(self) -> bool:
        """Whether a MoE layer's boundary also writes the NVFP4."""
        return self.ftz is not None and any(h.moe for h in self.hops.values())

    def nbytes(self) -> int:
        tensors = (self.data, self.scales, self.fp4, self.fp4_scales, self.token_scales)
        return sum(t.numel() * t.element_size() for t in tensors)

    def serves(self, rows: int) -> bool:
        return bool(self.hops) and rows in self.widths

    def ffn_input(self, layer_id: int, layer, out, forward_batch, tally=None):
        """``layer.ffn_boundary.prepare(out, forward_batch)`` with the push-plane
        kernel's quantizing form: an ``FfnInput``, or None, having done nothing, where
        the slot was not built for the layer or the stock prepare would not run the
        push-plane kernel. ``tally`` (a Counter) counts the outcome at a layer the
        slot was built for: TAKEN, or why not."""
        hop = self.hops.get(layer_id)
        if hop is None or hop.layer is not layer:
            return None
        prepared, why = self._take(hop, out, forward_batch)
        if tally is not None:
            tally[why or TAKEN] += 1
        return prepared

    def _take(self, hop, out, forward_batch):
        """``(the FfnInput, outcome)``: None or how the NVFP4 was left out; or
        ``(None, why not)`` having done nothing."""
        from sglang.srt.layers.layer_boundary.layout import SumGroup
        from sglang.srt.layers.layer_boundary.residual.stream import DeclaredSum
        from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

        stream = forward_batch.residual_stream
        if stream is None:
            return None, "no residual stream"
        pending, residual = stream.pending, stream.residual
        if pending is None or residual is None or not pending.update.is_plain_add:
            return None, "no plain add pending on a written residual"
        # _run_entry's declared sum: the attention's partial, still owing it
        if not (
            isinstance(pending.owed, DeclaredSum)
            and pending.owed.group is SumGroup.ATTN_TP
        ):
            return None, "the attention output does not owe its attention-TP sum"
        if hop.layer.ffn_boundary.entry(forward_batch) is not hop.entry:
            return None, "another batch path"
        launch, why = self._launch(hop, pending.value, residual, forward_batch)
        if launch is None:
            return None, why
        # StageBoundary._prepare: the input must be this stream's pending output
        stream.check(out)
        normed, residual_out, data, scales, nvfp4, left_out = launch()
        # the written residual consumes (releases) the pending contribution; the
        # batch keeps this stream object (StageBoundary.prepare re-assigns the same one)
        stream.write(residual_out)
        outcome = f"{TAKEN} without the NVFP4 ({left_out})" if left_out else None
        return FfnInput(normed, Mxfp8SwizzledInput(data, scales), nvfp4), outcome

    def _launch(self, hop, partial_sum, residual, forward_batch):
        """The kernel launch ``CustomAllReduceFusion._ffn_input_add_norm`` would make,
        in its quantizing form, as ``(call, None)``; ``(None, why not)`` where that
        fusion declines or the quantizing form cannot launch."""
        from sglang.kernels.ops.communication.all_reduce_fusion import (
            all_reduce_residual_norm_quant,
            all_reduce_residual_norm_quant_nvfp4,
        )
        from sglang.srt.layers.layer_boundary.fusions.custom_all_reduce_fusion import (
            _bf16_norm_gamma,
        )
        from sglang.srt.runtime_context import get_parallel

        from sgl_nvidia_mega.minimax_m31.boundary import _pads_scale_tile

        plan, fusion = hop.plan, hop.fusion
        gamma = _bf16_norm_gamma(plan.norm)
        if (
            gamma is None
            or residual.dtype != torch.bfloat16
            or partial_sum.dtype != torch.bfloat16
        ):
            return None, "no bf16 norm gamma, residual or partial sum"
        rows = int(partial_sum.shape[0])
        if not fusion._eligible(plan, forward_batch, rows):
            return None, "a batch the push-plane kernel does not take"
        if (
            not 0 < rows <= self.max_rows
            or tuple(partial_sum.shape) != (rows, self.hidden)
            or tuple(residual.shape) != (rows, self.hidden)
        ):
            return None, "rows past the slot's buffers"
        if not _pads_scale_tile(rows):
            return None, "no push-plane counter left to zero the scale tile's padding"
        left_out = ""
        if hop.moe and self.nvfp4:
            left_out = config_reason()
        nvfp4 = hop.moe and self.nvfp4 and not left_out
        eps, world_size = plan.norm.variance_epsilon, get_parallel().tp_size
        sf_cols = self.sf_cols

        def launch():
            # the stock fusion's arguments; a fresh residual_out (and normed row), as
            # there
            outputs = dict(
                residual_out=torch.empty_like(residual),
                quantized=self.data[:rows],
                scales=self.scales,
            )
            if not nvfp4:
                normed, residual_out, data, scales = all_reduce_residual_norm_quant(
                    partial_sum, residual, gamma, eps, world_size=world_size, **outputs
                )
                return normed, residual_out, data, scales, None, left_out
            normed, residual_out, data, scales, fp4, fp4_scales, token_scales = (
                all_reduce_residual_norm_quant_nvfp4(
                    partial_sum,
                    residual,
                    gamma,
                    eps,
                    self.global_scale_inv[moe_nofill.e4m3_max()],
                    world_size=world_size,
                    ftz=self.ftz,
                    fp4=self.fp4[:rows],
                    fp4_scales=self.fp4_scales[: rows * sf_cols],
                    token_scales=self.token_scales[:rows],
                    **outputs,
                )
            )
            # as PerTokenQuant hands them to the routed MoE
            hs_scale = fp4_scales.view(torch.float8_e4m3fn).reshape(rows, sf_cols)
            return normed, residual_out, data, scales, (fp4, hs_scale, token_scales), ""

        return launch, None

    def report(self, rows: int, tally: Counter) -> None:
        """Log what the first step of ``rows`` rows did at the layers the slot was
        built for (``tally``): a warning naming why it left any to the stock prepare,
        or took any without the NVFP4; once every served width has run, the widths at
        which it took them all. Host-side only, so a captured step costs nothing."""
        if rows in self.reported or rows not in self.widths:
            return
        self.reported[rows] = taken = tally.get(TAKEN, 0)
        other = sorted((why, n) for why, n in tally.items() if why != TAKEN)
        if other:
            logger.warning(
                "NVIDIA megakernel slot %s: a step of %d rows took %d of the %d FFN "
                "inputs it was built for in full; the others: %s",
                SLOT,
                rows,
                taken,
                len(self.hops),
                "; ".join(f"{why}: {n} layers" for why, n in other),
            )
        if len(self.reported) == len(self.widths):
            full = [w for w in self.widths if self.reported[w] == len(self.hops)]
            short = [w for w in self.widths if w not in full]
            logger.info(
                "NVIDIA megakernel slot %s: took all %d FFN inputs it was built for at "
                "rows %s%s",
                SLOT,
                len(self.hops),
                full,
                f"; fewer at rows {short}" if short else "",
            )


def _geometries(world_size: int, widths, hidden: int) -> list[tuple[int, int]]:
    """The (cluster size, vectors a thread) the boundary kernel launches at these
    widths (all_reduce_fusion._norm_launch of the plain all-reduce)."""
    from sglang.kernels.ops.communication.all_reduce_fusion import _norm_launch

    out = []
    for rows in widths:
        _, vecs, cluster = _norm_launch(
            world_size, rows, hidden, False, None, None, None
        )
        if (cluster, vecs) not in out:
            out.append((cluster, vecs))
    return out


def _nvfp4_on_every_rank(
    probing: bool, geometries, device
) -> tuple[Optional[bool], str]:
    """The probe (``probe_nvfp4``) on every rank, then ``(ftz, "")`` when every rank
    found the same mode, else ``(None, why not)``; a collective on the TP group."""
    local: tuple[Optional[bool], str] = (None, "the slot takes no MoE layer's input")
    if probing:
        try:
            local = probe_nvfp4(geometries, device=device)
        except Exception as e:  # gathered, then the NVFP4 is off on every rank
            local = (None, f"{type(e).__name__}: {e}")
    results = moe_nofill._gather(local)
    modes = {ftz for ftz, _ in results}
    if len(modes) == 1 and None not in modes:
        return results[0][0], ""
    if modes == {None} and len({why for _, why in results}) == 1:
        return None, results[0][1]
    return None, "the ranks differ: " + "; ".join(
        f"rank {rank}: {why or _mode_name(ftz)}"
        for rank, (ftz, why) in enumerate(results)
    )


def _mode_name(ftz: Optional[bool]) -> str:
    return "subnormals flushed" if ftz else "subnormals kept"


def build(model, widths: tuple[int, ...], slots: frozenset):
    """The slot's build (layers bound, NVFP4 probed, buffers allocated, kernels
    compiled), or None when its slot is not requested; a collective on the TP group."""
    if SLOT not in slots:
        return None
    from sglang.kernels.ops.communication.all_reduce_fusion import (
        MXFP8_QUANT_MAX_TOKENS,
        load_residual_norm_quant,
        load_residual_norm_quant_nvfp4,
    )
    from sglang.srt.runtime_context import get_parallel

    m = model.model
    hops, refused = {}, {}
    for i in range(m.start_layer, m.end_layer):
        hop, why = layer_hop(m.layers[i], HIDDEN)
        if hop is None:
            refused[i] = why
        else:
            hops[i] = hop
    served = tuple(w for w in widths if 0 < w <= MXFP8_QUANT_MAX_TOKENS)
    device = next(iter(hops.values())).plan.norm.gemma_weight.device if hops else None
    world_size = get_parallel().tp_size
    geometries = _geometries(world_size, served, HIDDEN)
    probing = bool(served) and any(h.moe for h in hops.values())
    ftz, nvfp4_off = _nvfp4_on_every_rank(probing, geometries, device)
    built = HopFfn(
        widths=served if hops else (),
        hops=hops,
        refused=refused,
        hidden=HIDDEN,
        device=device,
        ftz=ftz,
        nvfp4_off=nvfp4_off,
    )
    if hops and torch.cuda.is_available():
        # compile the kernels the captured steps launch before graph capture
        for rows in built.widths:
            if any(not h.moe for h in hops.values()) or not built.nvfp4:
                load_residual_norm_quant(world_size, rows, HIDDEN)
            if built.nvfp4:
                load_residual_norm_quant_nvfp4(world_size, rows, HIDDEN, ftz=ftz)
    reasons = sorted(set(refused.values()))
    logger.info(
        "NVIDIA megakernel slot %s: built for %d of %d FFN inputs at %s rows (%s); "
        "%.1f KiB of buffers%s",
        SLOT,
        len(hops),
        m.end_layer - m.start_layer,
        list(built.widths),
        f"MoE layers with the per-token NVFP4, {_mode_name(ftz)}"
        if built.nvfp4
        else f"the MXFP8 only: {nvfp4_off}",
        built.nbytes() / 1024,
        f" (refused: {'; '.join(reasons)})" if reasons else "",
    )
    return built


def serves(bound, rows: int) -> bool:
    """Whether the slot's kernels serve a step of this many rows."""
    return bound is not None and bound.serves(rows)
