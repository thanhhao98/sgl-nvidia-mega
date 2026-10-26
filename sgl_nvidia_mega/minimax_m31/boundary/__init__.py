# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/minimax_m3.py (MiniMaxM3Model.forward, MiniMaxM3DecoderLayer.forward),
# python/sglang/srt/managers/mm_utils.py (prepare_mm_inputs) and
# python/sglang/srt/layers/layer_boundary (StageBoundary.prepare of an attention
# input, CustomAllReduceFusion._finalize_add_norm / _all_reduce_add_norm).
"""The residual stream between sublayers: the embedding and the stream's start
(``start``), each layer's attention input (``attn_input``) and FFN input
(``ffn_input``), and the final norm (``final``).

Stock: SGLang's own stage boundaries (``attn_boundary`` / ``ffn_boundary``, the
push-plane all-reduce + residual + norm fusions) and residual batch, in the order
``MiniMaxM3DecoderLayer.forward`` calls them.

Slot ``hop_mxfp8`` (bitwise): the attention input of layers 1-59 is SGLang's
push-plane kernel (the plain all-reduce + residual + norm after a dense layer, the
MoE finalize + shared add + all-reduce + residual + norm after a MoE layer) in its
MXFP8 form, which also writes the MXFP8 of the normed row as FlashInfer's
``mxfp8_quantize`` does (e4m3 codes, UE8M0 scales of every 32 columns in the 128x4
swizzle). The attention gets that pair (``Mxfp8SwizzledInput``), which the stock
module's qkv|index projection takes as it is (``Fp8LinearMethod.apply``), so its own
quantize launch goes. The bf16 normed row, the updated residual and the DSpark
captures are the stock kernel's, bit for bit; everything else
``attn_boundary.prepare`` does is done the same way. Layer 0 (the embeddings' norm)
and any layer or step whose stock input would not run the push-plane kernel run the
stock ``prepare``; the first step of each width logs the layers it left to the stock
``prepare`` and why (``HopMxfp8.report``).

Early trigger: where the attention's next kernel on the pair is slot attn_ll's
low-latency q|k|v|index GEMM (``attention.low_latency_qkv``: attn_ll serves this
width, binds this layer and runs its q|k|v|index GEMM, and this step's host state is
the one it reproduces), the boundary kernel signals that GEMM's launch (PDL) at its
entry, in every CTA, instead of after its epilogue (``trigger_at_entry``), so the
GEMM streams its weights during the all-reduce. The GEMM reads only its weights before
its own PDL wait, and that wait returns once the boundary kernel has completed, so
its outputs are the same bytes (``mxfp8_ll``). Not at a DSpark capture layer whose
capture copies the updated residual (``_capture_copies``; L4/18/32/46 of the served
verify graph, a memcpy node each): the copy runs between the kernel and the GEMM, so
the GEMM cannot start early there, and the kernel signals at its end. The step's
report counts the layers whose kernel signalled at entry.

The pair is ``mxfp8_quantize(..., is_sf_swizzled_layout=True)``'s layout at up to
128 rows: the codes ``[rows, hidden]`` and a single 128-row tile of 128x4-swizzled
scales at every width, row r's scales in row r of the tile. FlashInfer's cute-dsl
and cutlass MXFP8 GEMMs (the stock projection's) read that tile at any width; a
consumer of another scale layout must not take the pair as it is. FlashInfer's
low-latency MXFP8 GEMM (an attn_ll candidate) reads it only at up to 8 rows: past
that it wants a separate 128-row scale atom for each 8-row tile.

Slot hop_ffn (``hop_ffn.py``, bitwise): the FFN input of every layer is the
push-plane kernel's quantizing form, which also writes the FFN's quantized inputs;
``ffn_input`` then returns a ``hop_ffn.FfnInput`` (the bf16 normed rows with them) for
``moe.step``.
"""

from __future__ import annotations

import logging
from collections import Counter
from functools import partial
from types import SimpleNamespace
from typing import Any, NamedTuple, Optional

import torch

from sgl_nvidia_mega.minimax_m31 import TEXT_CONFIG

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # residual_batch: the module alias SGLang's MiniMax-M3 model imports too
    "srt/models/minimax_m3.py": (
        "residual_batch",
        "norm",
        "finalize_norm",
        "end_layer",
        "layers_to_capture",
        "attn_boundary",
        "ffn_boundary",
        "self_attn",
        "_fused_qkv_index",
        "_qm",
        "start_layer",
        "layers",
    ),
    "srt/models/minimax_m3_vl.py": ("model",),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "final_norm",
        "final_norm(finalize_norm)",
        "final_norm(capture)",
        "final_norm(skip_empty)",
    ),
    "srt/managers/mm_utils.py": ("prepare_mm_inputs",),
    # prepare(**call) hands capture_gathered to _prepare_attention
    "srt/layers/layer_boundary/stage.py": (
        "prepare",
        "_prepare_attention(capture_gathered)",
        "finish",
    ),
    # _capture_copies: the executor's capture list adopts a residual it owns
    "srt/layers/aux_hidden_states.py": (
        "AuxHiddenStateList",
        "capture",
        "capture(owned)",
    ),
    "srt/layers/layer_boundary/contracts.py": ("BatchVariant", "ORDINARY"),
    "srt/layers/layer_boundary/fusions/custom_all_reduce_fusion.py": (
        "CustomAllReduceFusion",
        "_bf16_norm_gamma",
        "hidden_size",
        "_finalize_add_norm",
        "_all_reduce_add_norm",
        "_eligible",
    ),
    "srt/layers/layer_boundary/prepare.py": (
        "_attn_input_default",
        "_run_entry",
        "_run_entry(carried_fusions)",
        "_run_entry(is_plain_add)",
        # what layer_hop declines: fusions carried into the read, a read bound per batch
        "_run_entry(carried_read_fusions)",
        "_run_entry(read)",
    ),
    "srt/layers/layer_boundary/residual/__init__.py": ("bind",),
    "srt/layers/layer_boundary/*.py": (
        "plan",
        "enters_stack",
        "fusions",
        "_publish_lora_layout",
        "qkv_latent_func",
        "paths",
        "entry",
        "prepare",
        "input_move",
        "attn_input_adapter",
        "declared_sum",
        "capture_preserves_residual",
    ),
    # pending.update: the stream's ResidualUpdate (PLAIN_ADD in this model)
    "srt/layers/layer_boundary/residual/__init__.py": ("is_plain_add",),
    "srt/layers/layer_boundary/residual/stream.py": (
        "pending",
        "residual",
        "check",
        "write",
        "owed",
        "update",
    ),
    "srt/model_executor/forward_batch_info.py": ("residual_stream",),
    "srt/layers/layer_boundary/fusions/*.py": (
        "MoeDeferredFinalize",
        "m",
        "routed_output",
        "permuted_indices",
        "expert_weights",
        "gated_shared_output",
    ),
    "srt/layers/layer_boundary/output.py": (
        "UnreducedOutput",
        "group",
        "reduce_to_dp_local",
        "partial",
    ),
    "srt/layers/quantization/mxfp8_input.py": (
        "accepts_mxfp8_swizzled_input",
        "Mxfp8SwizzledInput",
        # the linear's fields accepts_mxfp8_swizzled_input reads, which layer_hop
        # hands it from the fused qkv|index projection
        '"quant_method"',
        '"block_fp8_mxfp8_ready"',
    ),
    # the registered push plane's check, which _pads_scale_tile asks at call time
    "srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py": ("can_fuse_all_reduce",),
    "kernels/ops/communication/all_reduce_fusion.py": (
        "all_reduce_residual_norm_quant",
        "moe_finalize_all_reduce_residual_norm_quant",
        "mxfp8_scale_bytes",
        "MXFP8_QUANT_MAX_TOKENS",
        "load_residual_norm_quant",
        "all_reduce_residual_norm_quant(world_size)",
        "all_reduce_residual_norm_quant(residual_out)",
        "all_reduce_residual_norm_quant(out)",
        "all_reduce_residual_norm_quant(quantized)",
        "all_reduce_residual_norm_quant(scales)",
        "moe_finalize_all_reduce_residual_norm_quant(hidden_dim)",
        # the entry trigger slot attn_ll's low-latency GEMM after the kernel takes
        "all_reduce_residual_norm_quant(trigger_at_entry)",
        "moe_finalize_all_reduce_residual_norm_quant(trigger_at_entry)",
        "load_residual_norm_quant(top_k)",
        "load_residual_norm_quant(weight_dtype)",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_group"', '"tp_size"'),
    "srt/layers/layernorm.py": ("variance_epsilon", '"gemma_weight"'),
    "srt/distributed/parallel_state.py": ("ranks",),
}

logger = logging.getLogger(__name__)

SLOT = "hop_mxfp8"
HIDDEN = TEXT_CONFIG["hidden_size"]
TOP_K = TEXT_CONFIG["num_experts_per_tok"]
# st.scratch[TALLY]: a step's outcome at each layer slot hop_mxfp8 was built for,
# TAKEN or why it left the layer to the stock prepare; and EARLY, the layers taken
# whose kernel signalled attn_ll's low-latency q|k|v|index GEMM at its entry
TALLY = "boundary.hop_mxfp8"
TAKEN = "taken"
EARLY = "taken, early trigger"


def start(st) -> Any:
    """The step's embeddings [rows, hidden]: ``prepare_mm_inputs``' decode branch,
    then ``MiniMaxM3Model.forward``'s ``residual_batch.start``."""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.managers.mm_utils import prepare_mm_inputs

    x = prepare_mm_inputs(st.input_ids, st.fb, st.model, {}, st.lm)
    residual_batch.start(st.fb)
    return x


def attn_input(layer, x, st, captured) -> Any:
    """The attention input of ``layer``: the previous sublayer's output reduced into
    the stream, then the input norm; ``captured`` (the DSpark capture list) takes the
    updated stream at a capture layer. With slot hop_mxfp8, the MXFP8 pair of the
    normed input where the slot takes the layer and step, the outcome counted in
    ``st.scratch[TALLY]``."""
    hop = st.bound.get("boundary") if st.slots.hop_mxfp8 else None
    if hop is not None:
        tally = st.scratch.setdefault(TALLY, Counter())
        prepared = hop.attn_input(
            st.layer_id,
            layer,
            x,
            st.fb,
            captured,
            tally,
            early_trigger=partial(_low_latency_qkv_follows, layer, st=st),
        )
        if prepared is not None:
            return prepared
    return layer.attn_boundary.prepare(x, st.fb, capture_gathered=captured)


def _low_latency_qkv_follows(layer, pair, *, st) -> bool:
    """Whether the attention's next kernel on ``pair`` is slot attn_ll's low-latency
    q|k|v|index GEMM, the attention step's own decision (host state only)."""
    if not st.slots.attn_ll:
        return False
    from sgl_nvidia_mega.minimax_m31 import attention

    return attention.low_latency_qkv(layer, pair, st)


def _capture_copies(captured, owned) -> bool:
    """Whether the DSpark capture of the updated residual launches a copy between the
    boundary kernel and the attention's first kernel (a memcpy, or a copy kernel
    launched without PDL), after which that kernel cannot start before the boundary
    kernel has completed: AuxHiddenStateList, the executor's, copies what it does not
    own; AuxHiddenStatePacker copies everything into its buffer."""
    from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList

    return captured is not None and not (
        owned and isinstance(captured, AuxHiddenStateList)
    )


def ffn_input(layer, out, st) -> Any:
    """The FFN input of ``layer`` from its attention output ``out``. With slot
    hop_ffn, an ``FfnInput`` (the normed rows, their MXFP8 pair and, for a MoE layer,
    their per-token NVFP4) where the slot takes the layer and step, the outcome
    counted in ``st.scratch[hop_ffn.TALLY]``."""
    from sgl_nvidia_mega.minimax_m31.boundary import hop_ffn

    out = layer.attn_boundary.finish(out, st.fb)
    hop = st.bound.get("hop_ffn") if st.slots.hop_ffn else None
    if hop is not None:
        tally = st.scratch.setdefault(hop_ffn.TALLY, Counter())
        prepared = hop.ffn_input(st.layer_id, layer, out, st.fb, tally)
        if prepared is not None:
            return prepared
    return layer.ffn_boundary.prepare(out, st.fb)


def final(st, x, aux) -> Any:
    """The last layer's output reduced into the stream and normed; the DSpark capture
    of the last layer's stream when the draft reads it. With slots hop_mxfp8 and
    hop_ffn, the step's outcomes go to ``HopMxfp8.report`` and ``HopFfn.report``."""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

    from sgl_nvidia_mega.minimax_m31.boundary import hop_ffn

    tally = st.scratch.get(TALLY)
    if tally is not None:
        st.bound["boundary"].report(st.rows, tally)
    tally = st.scratch.get(hop_ffn.TALLY)
    if tally is not None:
        st.bound["hop_ffn"].report(st.rows, tally)
    m = st.model
    return residual_batch.final_norm(
        x,
        st.fb,
        m.norm,
        finalize_norm=m.finalize_norm,
        capture=aux.capture if m.end_layer in m.layers_to_capture else None,
        skip_empty=True,
    )


class _Hop(NamedTuple):
    """A layer whose attention input the slot takes: the layer, its attention
    stage's plan, the plan's ordinary batch path and its push-plane fusion."""

    layer: Any
    plan: Any
    entry: Any
    fusion: Any


def layer_hop(layer, hidden: int = HIDDEN) -> tuple[Optional[_Hop], Optional[str]]:
    """``(hop, None)`` when ``layer``'s stock attention input completes the previous
    layer's output with the push-plane kernel (CustomAllReduceFusion, tried first,
    for an ordinary batch) and its attention takes an MXFP8 pair; else
    ``(None, why not)``."""
    from sglang.srt.layers.layer_boundary.contracts import BatchVariant
    from sglang.srt.layers.layer_boundary.fusions.custom_all_reduce_fusion import (
        CustomAllReduceFusion,
        _bf16_norm_gamma,
    )
    from sglang.srt.layers.layer_boundary.prepare import (
        _attn_input_default,
        _run_entry,
    )
    from sglang.srt.layers.quantization.mxfp8_input import (
        accepts_mxfp8_swizzled_input,
    )

    plan = getattr(getattr(layer, "attn_boundary", None), "plan", None)
    if plan is None:
        return None, "no attention stage boundary"
    if plan.enters_stack:
        return None, "the stack's first attention input (the embeddings' norm)"
    fusion = plan.fusions
    if not isinstance(fusion, CustomAllReduceFusion) or fusion.hidden_size != hidden:
        return None, f"no push-plane fusion of hidden {hidden}"
    if _bf16_norm_gamma(plan.norm) is None:
        return None, "the input norm has no bf16 gamma"
    if plan._publish_lora_layout or plan.qkv_latent_func is not None:
        return None, "the attention input publishes LoRA rows or a QKV latent hook"
    entry = getattr(plan.paths.get(BatchVariant.ORDINARY), "entry", None)
    if entry is None:
        return None, "no ordinary batch path"
    if (
        entry.input_move is not None
        or entry.attn_input_adapter not in (None, _attn_input_default)
        or entry.declared_sum is not None
    ):
        return None, "the attention input moves rows or owes a declared sum"
    prepare = entry.prepare
    carried = (
        prepare.keywords.get("carried_fusions", ())
        if isinstance(prepare, partial)
        and prepare.func is _run_entry
        and prepare.keywords.get("is_plain_add") is True
        else ()
    )
    if carried and (
        prepare.keywords.get("carried_read_fusions")
        or getattr(prepare.keywords.get("read"), "bind", None) is not None
    ):
        return None, "the attention input's read carries fusions or binds per batch"
    first = (fusion._finalize_add_norm, fusion._all_reduce_add_norm)
    if len(carried) < len(first) or not all(
        isinstance(c, partial) and c.func == f and c.args == (plan,) and not c.keywords
        for c, f in zip(carried, first)
    ):
        return None, "the push-plane kernel is not the input's first fused completion"
    proj = getattr(getattr(layer, "self_attn", None), "_fused_qkv_index", None)
    if proj is None:
        return None, "the attention has no fused qkv|index projection"
    linear = SimpleNamespace(
        quant_method=getattr(proj, "_qm", None),
        block_fp8_mxfp8_ready=getattr(proj, "block_fp8_mxfp8_ready", False),
    )
    if not accepts_mxfp8_swizzled_input(linear) or proj.weight.shape[-1] != hidden:
        return None, "the qkv|index projection takes no MXFP8 128x4 input"
    return _Hop(layer, plan, entry, fusion), None


class HopMxfp8:
    """Slot hop_mxfp8's build: the layers it takes and the MXFP8 pair's buffers.

    The pair's buffers are sized for the largest width and sliced per step (the scale
    tile is the same 128 rows at every width); the bf16 normed row, which the slot's
    consumers do not read, goes to a scratch buffer. They are written by each served
    layer's boundary kernel and read only by that layer's qkv|index GEMM, both on the
    main stream: the next writer, the following layer's boundary kernel, starts after
    the GEMM has finished, because MSA's indexer and top-k between them launch without
    PDL, so each waits for every earlier kernel on the stream; every other kernel
    between them that uses PDL (SGLang's MSA-v1 dequant kernels where SGLang launches
    them with PDL) waits for its predecessor before it ends. The GEMM reads them after
    its own PDL wait, so after the boundary kernel has completed, also where that
    kernel let it start at its entry (early trigger). The alternate (MoE) stream never
    touches them, and the decode graphs of different widths, which share them, never
    run at the same time.

    MSA's indexer (its Q8K8 kernel, or its Q8KV4 one) and top-k are not ours, but the
    served verify graph's nsys shows that they launch without PDL (GB300, the c1 and c8
    DSpark verify graphs): in every replayed launch each starts at least 0.42 us
    after its predecessor ends (median 0.58), as kernels launched without PDL do. So
    every layer keeps a kernel that starts only after everything before it on the
    stream has completed. This buffer reuse rests on it, and so do slot hop_ffn's
    (``hop_ffn``) and slot qk_store's PDL order (``attention.qk_store``): a change
    that launches MSA's indexer or top-k with PDL must first prove all three again.
    """

    def __init__(
        self,
        *,
        widths: tuple[int, ...],
        hops: dict,
        refused: dict,
        hidden: int,
        device,
    ):
        self.widths = tuple(sorted(widths))
        self.hops = hops  # layer index -> _Hop
        self.refused = refused  # layer index -> why the slot does not take it
        self.hidden = hidden
        self.max_rows = self.widths[-1] if self.widths else 0
        self.data = torch.empty(
            self.max_rows, hidden, dtype=torch.float8_e4m3fn, device=device
        )
        self.scales = torch.empty(
            _scale_bytes(hidden), dtype=torch.uint8, device=device
        )
        self.normed = torch.empty(
            self.max_rows, hidden, dtype=torch.bfloat16, device=device
        )
        # width -> the layers its first step took, and of those the layers whose
        # kernel signalled the low-latency q|k|v|index GEMM at its entry (report)
        self.reported: dict[int, int] = {}
        self.early: dict[int, int] = {}

    def serves(self, rows: int) -> bool:
        return bool(self.hops) and rows in self.widths

    def attn_input(
        self,
        layer_id: int,
        layer,
        x,
        forward_batch,
        captured,
        tally=None,
        early_trigger=None,
    ):
        """``layer.attn_boundary.prepare(x, forward_batch, capture_gathered=captured)``
        with the push-plane kernel's MXFP8 form: the MXFP8 pair of the normed input,
        or None, having done nothing, where the slot was not built for the layer or
        the stock prepare would not run the push-plane kernel. ``early_trigger``, a
        predicate on the pair asked before the launch where no capture copy follows
        the kernel: whether the next kernel on the pair reads nothing before its PDL
        wait, so that the kernel may signal its launch at entry. ``tally`` (a
        Counter) counts the outcome at a layer the slot was built for: TAKEN, or why
        not; and EARLY where the kernel signalled at entry."""
        hop = self.hops.get(layer_id)
        if hop is None or hop.layer is not layer:
            return None
        prepared, why, early = self._take(
            hop, x, forward_batch, captured, early_trigger
        )
        if tally is not None:
            tally[why or TAKEN] += 1
            if early:
                tally[EARLY] += 1
        return prepared

    def _take(self, hop, x, forward_batch, captured, early_trigger=None):
        """``(the MXFP8 pair, None, whether its kernel signalled at entry)``, or
        ``(None, why not, False)`` having done nothing."""
        from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

        stream = forward_batch.residual_stream
        if stream is None:
            return None, "no residual stream", False
        pending, residual = stream.pending, stream.residual
        if pending is None or residual is None or not pending.update.is_plain_add:
            return None, "no plain add pending on a written residual", False
        if hop.layer.attn_boundary.entry(forward_batch) is not hop.entry:
            return None, "another batch path", False
        launch, why = self._launch(hop, pending.owed, residual, forward_batch)
        if launch is None:
            return None, why, False
        # StageBoundary._prepare: the input must be this stream's pending output
        stream.check(x)
        # the pair the kernel writes and the attention takes, in the slot's buffers;
        # a fresh residual_out, as the stock fusion's
        pair = Mxfp8SwizzledInput(self.data[: residual.shape[0]], self.scales)
        residual_out = torch.empty_like(residual)
        # StageBoundary._prepare_attention's capture_gathered, with no input move: the
        # capture owns the updated residual where the following stage keeps it
        keeps = hop.entry.capture_preserves_residual
        owned = (
            captured is not None
            and keeps is not None
            and keeps(residual_out, forward_batch)
        )
        # the next kernel on the pair follows the kernel directly unless the capture
        # copies the residual between them
        early = (
            early_trigger is not None
            and not _capture_copies(captured, owned)
            and bool(early_trigger(pair))
        )
        _, residual_out, _, _ = launch(pair, residual_out, early)
        # the written residual consumes (releases) the pending contribution; the
        # batch keeps this stream object (StageBoundary.prepare re-assigns the same one)
        stream.write(residual_out)
        if captured is not None:
            captured.capture(residual_out, owned=owned)
        return pair, None, early

    def _launch(self, hop, owed, residual, forward_batch):
        """The kernel launch the stock carried fusion would make for ``owed``, in its
        MXFP8 form, as ``(call, None)``: ``call(pair, residual_out, trigger_at_entry)``
        writes the pair's tensors and ``residual_out``. ``(None, why not)`` where that
        fusion declines or the MXFP8 form cannot launch."""
        from sglang.kernels.ops.communication.all_reduce_fusion import (
            all_reduce_residual_norm_quant,
            moe_finalize_all_reduce_residual_norm_quant,
        )
        from sglang.srt.layers.layer_boundary.fusions.custom_all_reduce_fusion import (
            _bf16_norm_gamma,
        )
        from sglang.srt.layers.layer_boundary.fusions.moe_finalize import (
            MoeDeferredFinalize,
        )
        from sglang.srt.layers.layer_boundary.output import UnreducedOutput
        from sglang.srt.runtime_context import get_parallel

        plan, fusion = hop.plan, hop.fusion
        gamma = _bf16_norm_gamma(plan.norm)
        if gamma is None or residual.dtype != torch.bfloat16:
            return None, "no bf16 norm gamma or residual"
        if isinstance(owed, MoeDeferredFinalize):
            # CustomAllReduceFusion._finalize_add_norm
            rows = int(owed.m)
            if (
                owed.routed_output.dtype != torch.bfloat16
                or owed.gated_shared_output.dtype != torch.bfloat16
                or owed.expert_weights.shape[0] != rows
            ):
                return None, "a MoE handoff the push-plane kernel does not take"
            op = partial(
                moe_finalize_all_reduce_residual_norm_quant,
                owed.routed_output,
                owed.permuted_indices.reshape(-1),
                owed.expert_weights,
                owed.expert_weights.shape[1],
                owed.gated_shared_output,
                hidden_dim=fusion.hidden_size,
            )
        elif isinstance(owed, UnreducedOutput):
            # CustomAllReduceFusion._all_reduce_add_norm, which _run_entry reaches
            # for a sum that moves no rows
            if (
                owed.group is None
                or owed.reduce_to_dp_local is not None
                or owed.partial.dtype != torch.bfloat16
                or owed.partial.shape != residual.shape
                or owed.group.ranks != get_parallel().tp_group.ranks
            ):
                return None, "a partial sum the push-plane kernel does not take"
            rows = int(owed.partial.shape[0])
            op = partial(all_reduce_residual_norm_quant, owed.partial)
        else:
            return None, "an output the push-plane kernel does not complete"
        if not 0 < rows <= self.max_rows or residual.shape != (rows, self.hidden):
            return None, "rows past the slot's buffers"
        if not fusion._eligible(plan, forward_batch, rows):
            return None, "a batch the push-plane kernel does not take"
        if not _pads_scale_tile(rows):
            return None, "no push-plane counter left to zero the scale tile's padding"
        eps, world_size = plan.norm.variance_epsilon, get_parallel().tp_size

        def launch(pair, residual_out, trigger_at_entry: bool):
            # the stock fusion's arguments (residual_out a fresh tensor, as there)
            return op(
                residual,
                gamma,
                eps,
                world_size=world_size,
                residual_out=residual_out,
                out=self.normed[:rows],
                quantized=pair.data,
                scales=pair.scales,
                trigger_at_entry=trigger_at_entry,
            )

        return launch, None

    def report(self, rows: int, tally: Counter) -> None:
        """Log what the first step of ``rows`` rows did at the layers the slot was
        built for (``tally``): a warning naming why it left any to the stock prepare,
        which runs them at no gain; once every served width has run, the widths at
        which it took them all, and how many layers signalled the low-latency
        q|k|v|index GEMM at entry. Host-side only, so a captured step costs nothing."""
        if rows in self.reported or rows not in self.widths:
            return
        self.reported[rows] = taken = tally.get(TAKEN, 0)
        self.early[rows] = tally.get(EARLY, 0)
        declined = sorted(
            (why, n) for why, n in tally.items() if why not in (TAKEN, EARLY)
        )
        if declined:
            logger.warning(
                "NVIDIA megakernel slot %s: a step of %d rows took %d of the %d "
                "attention inputs it was built for; the stock prepare ran the others "
                "(%s)",
                SLOT,
                rows,
                taken,
                len(self.hops),
                "; ".join(f"{why}: {n} layers" for why, n in declined),
            )
        if len(self.reported) == len(self.widths):
            full = [w for w in self.widths if self.reported[w] == len(self.hops)]
            short = [w for w in self.widths if w not in full]
            early = {w: n for w, n in self.early.items() if n}
            logger.info(
                "NVIDIA megakernel slot %s: took all %d attention inputs it was built "
                "for at rows %s%s%s",
                SLOT,
                len(self.hops),
                full,
                f"; fewer at rows {short}" if short else "",
                (
                    "; signalled attn_ll's low-latency q|k|v|index GEMM at entry at "
                    f"{dict(sorted(early.items()))} (rows: layers)"
                    if early
                    else ""
                ),
            )


def _scale_bytes(hidden: int) -> int:
    from sglang.kernels.ops.communication.all_reduce_fusion import mxfp8_scale_bytes

    return mxfp8_scale_bytes(hidden)


def _pads_scale_tile(rows: int) -> bool:
    """Whether the MXFP8 form launches at ``rows`` on the registered push plane
    (``can_fuse_all_reduce``): below 128 rows its trailing (bumper) cluster zeros the
    scale tile's padding rows, which takes a phase counter past the rows
    (all_reduce_fusion.cuh, ``run_impl``), so the plane must also take ``rows + 1``
    rows, while the stock kernel also runs when the rows take every counter."""
    from sglang.kernels.ops.communication.all_reduce_fusion import (
        MXFP8_QUANT_MAX_TOKENS,
    )
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        can_fuse_all_reduce,
    )

    return can_fuse_all_reduce(rows, HIDDEN) and (
        rows == MXFP8_QUANT_MAX_TOKENS or can_fuse_all_reduce(rows + 1, HIDDEN)
    )


def build(model, widths: tuple[int, ...], slots: frozenset):
    """The slot's build (weights bound, buffers allocated, kernels compiled), or None
    when its slot is not requested."""
    if SLOT not in slots:
        return None
    from sglang.kernels.ops.communication.all_reduce_fusion import (
        MXFP8_QUANT_MAX_TOKENS,
        load_residual_norm_quant,
    )

    m = model.model
    hops, refused = {}, {}
    for i in range(m.start_layer, m.end_layer):
        hop, why = layer_hop(m.layers[i])
        if hop is None:
            refused[i] = why
        else:
            hops[i] = hop
    served = tuple(w for w in widths if 0 < w <= MXFP8_QUANT_MAX_TOKENS)
    device = next(iter(hops.values())).plan.norm.gemma_weight.device if hops else None
    built = HopMxfp8(
        widths=served if hops else (),
        hops=hops,
        refused=refused,
        hidden=HIDDEN,
        device=device,
    )
    if hops and torch.cuda.is_available():
        # compile the kernels the captured steps launch before graph capture
        from sglang.srt.runtime_context import get_parallel

        world_size = get_parallel().tp_size
        for rows in built.widths:
            load_residual_norm_quant(world_size, rows, HIDDEN)
            load_residual_norm_quant(
                world_size, rows, HIDDEN, top_k=TOP_K, weight_dtype=torch.float32
            )
    reasons = sorted({why for i, why in refused.items() if i != m.start_layer})
    logger.info(
        "NVIDIA megakernel slot %s: built for %d of %d attention inputs at %s rows%s",
        SLOT,
        len(hops),
        m.end_layer - m.start_layer,
        list(built.widths),
        f" (refused: {'; '.join(reasons)})" if reasons else "",
    )
    return built


def serves(bound, rows: int) -> bool:
    """Whether the slot's kernels serve a step of this many rows."""
    return bound is not None and bound.serves(rows)
