# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/minimax_m3.py
# (MiniMaxM3Attention.forward_prepare / forward_core, the fused q|k|v|index branch),
# python/sglang/srt/layers/linear.py (RowParallelLinear.forward) and
# python/sglang/srt/layers/quantization/fp8_utils.py
# (flashinfer_mxfp8_blockscaled_linear).
"""The attention sublayer of a covered step: the fused q/k/v/index projection, the
Gemma q/k norms + RoPE (+ FP8 q), the NVFP4 KV + index-K store, the MSA-v1 indexer
and sparse attention, and the output projection (an unreduced partial).

Stock: SGLang's ``MiniMaxM3Attention`` module.

The slots below run the sublayer decomposed (``decomposed_attention``):
``MiniMaxM3Attention.forward`` (forward_prepare + forward_core) of this
configuration's branch, call for call, with its two GEMMs given as callables and the
q/k norm + RoPE optionally fused with the KV store. Both slots compose: with both,
qk_store's fused kernel follows attn_ll's low-latency q|k|v|index GEMM.

Slot ``attn_ll`` (tolerance; steps of 1-8 rows): the fused q|k|v|index projection
(N 2560, K 6144) on FlashInfer's low-latency block-scaled GEMM launched directly with
pinned tactics (``mxfp8_ll``), which loads the weights before the PDL wait and splits
K inside a cluster. o_proj (N 6144, K 2048) stays SGLang's by default: on the
low-latency kernel it gained ~0.1 us a layer in the served graph and alone moved the
tokens of every step in the parity bisect, so it runs there only when the test-only
``GEMMS_ENV`` names it. The activation is quantized by the same FlashInfer MXFP8
quantize with the arguments SGLang's MXFP8 linear passes (so the GEMM inputs are the
same bytes), or not at all when the input is already an ``Mxfp8SwizzledInput`` (slot
hop_mxfp8); the GEMM outputs are allocated per call as FlashInfer's ``mm_mxfp8``
allocates them, o_proj's under RowParallelLinear's allocation context. Only the FP32
split-K summation order of the GEMMs differs (``runtime.TOLERANCES``).

Slot ``qk_store`` (bitwise, ``qk_store.py``): one kernel for the q/k/index norm +
RoPE with the FP8 q epilogue and the NVFP4 K/V/index-K store, around SGLang's own
linears (or attn_ll's GEMMs); the attention backend then runs without its store.

A layer whose modules are not the configuration the branch reproduces (checked at
build), or a step whose input or forward state is not (checked on the host before the
layer runs), runs the stock module call (or, for qk_store alone, SGLang's two kernels).

PDL, the kernel before the q|k|v|index GEMM: with slot hop_mxfp8 it is the
attention-input boundary kernel, and nothing is launched between them (the pair needs
no quantize, and slot qk_store decides on host state alone before the GEMM). That
kernel then signals the GEMM's launch at its entry (``boundary.attn_input`` asks
``low_latency_qkv`` before it launches), so the GEMM streams its weights during the
boundary's all-reduce. Safe because the GEMM reads only its weights and their scales
(static) before its PDL wait, and its activation loads, and every store, come after
it (``mxfp8_ll``). At a DSpark capture layer whose capture copies the residual, that
copy would sit between them (a graph node that is not a kernel, which the GEMM waits
for), so the kernel signals at its end there (``boundary._capture_copies``).

PDL, the kernels after the GEMMs: the low-latency GEMM lets the next kernel start
before it has waited on its own predecessor (``mxfp8_ll``), and its predecessor may
itself have signalled at entry (above), so the kernel after each GEMM must read
nothing written before the GEMM ahead of its own PDL wait. Both do: SGLang's q/k norm
+ RoPE kernel (``fused_gemma_qknorm_rope.cuh``) reads only its norm weights and the
positions (a static graph input) before its wait, and so does qk_store's fused
kernel, which runs that kernel's code first and reads the cache slots, its K/V rows
and the global scale after it (``fused_qknorm_rope_store.cuh``); and the attention ->
FFN boundary's all-reduce + residual + norm (``all_reduce_residual_norm``, or its
quantizing forms under slot hop_ffn) waits first (it has no ``prefetch_metadata``;
only ``moe_finalize_all_reduce`` does, and it defaults to False). A change that lets
either kernel read the step's activations before its wait breaks this slot.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any, Callable, Optional

import torch
from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/minimax_m3.py": (
        "self_attn",
        "_QKNORM_GROUP_INDEX_Q",
        "_QKNORM_GROUP_MAIN_Q",
        "MiniMaxM3Attention",
        "_FusedQKVIndexProj",
        "_is_npu",
        '"fused_qkv_index_proj"',
        "_fused_main_size",
        "_qknorm_groups",
        "num_heads",
        "head_dim",
        "num_idx_heads",
        "idx_head_dim",
        "num_kv_heads",
        "q_size",
        "kv_size",
        "rotary_emb",
        "q_norm",
        "_split_index_qkv",
        "attn",
        "o_proj",
        "is_sparse_attention_layer",
        "disable_index_value",
        "index_o_proj",
        "_can_use_aiter_fused_qknorm_static",
        "_combined_qknorm_ok",
        "_fp8_q_epilogue_ok",
        "_fused_qkv_index",
        "_qm",
        "forward_prepare",
        "forward_core",
        "_aiter_sparse_qknorm_cache",
        # the stock step's MiniMaxM3Attention.forward(positions=, hidden_states=,
        # forward_batch=), and the layers build walks
        "forward(positions)",
        "forward(hidden_states)",
        "forward(forward_batch)",
        "start_layer",
        "end_layer",
        "layers",
    ),
    "srt/models/minimax_m3_vl.py": ("model",),
    "srt/layers/quantization/mxfp8_input.py": ("Mxfp8SwizzledInput",),
    "kernels/ops/attention/minimax_qknorm_rope.py": (
        "minimax_qknorm_rope_grouped_fp8",
        "minimax_qknorm_rope_grouped_fp8(fp8_a)",
        "minimax_qknorm_rope_grouped_fp8(group_a)",
        "minimax_qknorm_rope_grouped_fp8(fp8_b)",
        "minimax_qknorm_rope_grouped_fp8(group_b)",
    ),
    # attn.attn(..., idx_q=, idx_k=, idx_v=) reaches the MSA NVFP4 backend's _forward
    # through RadixAttention.forward(**kwargs); save_kv_cache=False (slot qk_store)
    # skips its store, the only work it gates there
    "srt/layers/radix_attention.py": ("forward(save_kv_cache)",),
    "srt/layers/attention/minimax_sparse_nvfp4_backend.py": (
        "_forward(save_kv_cache)",
        "_forward(idx_q)",
        "_forward(idx_k)",
        "_forward(idx_v)",
    ),
    "srt/layers/rotary_embedding/*.py": ("cos_sin_cache",),
    "srt/layers/layernorm.py": ("variance_epsilon",),
    "srt/distributed/device_communicators/pynccl_allocator.py": (
        "use_symmetric_memory",
        "use_symmetric_memory(disabled)",
    ),
    "srt/layers/dp_attention.py": ("is_allocation_symmetric",),
    "srt/runtime_context.py": (
        "get_parallel",
        "get_forward",
        '"tp_group"',
        '"attn_tp_group"',
        '"sp_active"',
    ),
    "srt/layers/linear.py": (
        "RowParallelLinear",
        "reduce_results",
        "input_is_parallel",
        "use_dp_attention_reduce",
        "use_decode_attn_tp",
        "quant_method",
    ),
    "srt/layers/quantization/fp8_utils.py": (
        "flashinfer_mxfp8_quantize",
        "flashinfer_mxfp8_blockscaled_linear",
        "flashinfer_mxfp8_quantize(is_sf_swizzled_layout)",
        "flashinfer_mxfp8_quantize(alignment)",
        "is_flashinfer_cutedsl",
        "is_flashinfer_cutlass",
    ),
    "srt/layers/quantization/fp8.py": (
        "Fp8LinearMethod",
        "use_mxfp8",
        "use_marlin",
        "convert_mxfp8_to_block",
        "block_fp8_as_mxfp8",
        "mxfp8_dense_backend",
        "w8a8_mxfp8_linear",
        "apply",
        "weight_scale_inv_swizzled",
    ),
}

logger = logging.getLogger(__name__)

SLOT = "attn_ll"
# the GEMMs the slot can run on the low-latency kernel, of one TP4 rank: name -> (N, K)
SHAPES = {"qkv_index": mxfp8_ll.QKV_INDEX, "o_proj": mxfp8_ll.O_PROJ}
# the GEMMs it runs there unless GEMMS_ENV says otherwise
DEFAULT_GEMMS: tuple[str, ...] = ("qkv_index",)

# Test only (the composition check, bisecting, timing o_proj): the layers the slot runs
# (comma-separated ids; unset: every bound layer) and its GEMMs (comma-separated names
# of SHAPES; unset or empty: DEFAULT_GEMMS; "qkv_index,o_proj" adds o_proj). A layer or
# GEMM left out runs SGLang's module.
LAYERS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_ATTN_LL_LAYERS"
GEMMS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_ATTN_LL_GEMMS"


def _env_set(name: str, environ=None):
    raw = (os.environ if environ is None else environ).get(name)
    return None if raw is None else {x.strip() for x in raw.split(",")} - {""}


def requested_gemms(environ=None) -> tuple[str, ...]:
    """The GEMMs the slot runs on the low-latency kernel, in SHAPES order:
    ``GEMMS_ENV``'s, else DEFAULT_GEMMS. Raises ValueError on a name SHAPES lacks."""
    names = _env_set(GEMMS_ENV, environ)
    if not names:
        return DEFAULT_GEMMS
    unknown = sorted(names - set(SHAPES))
    if unknown:
        raise ValueError(
            f"{GEMMS_ENV}: unknown GEMM {', '.join(map(repr, unknown))} (known: "
            f"{', '.join(SHAPES)})"
        )
    return tuple(name for name in SHAPES if name in names)


def step(layer, x, st) -> Any:
    """``layer``'s attention partial [rows, hidden] from its normed input ``x`` (BF16,
    or an ``Mxfp8SwizzledInput`` from slot hop_mxfp8)."""
    attn = layer.self_attn
    gemms = _step_gemms(layer, x, st)
    store = _fused_store(attn, st)
    if gemms is None and store is None:
        return attn(positions=st.positions, hidden_states=x, forward_batch=st.fb)
    if gemms is not None and gemms.qkv_index is not None:
        qkv_index = gemms.qkv_index
    else:
        qkv_index = attn.fused_qkv_index_proj
    if gemms is not None and gemms.o_proj is not None:
        o_proj = partial(gemms.o_proj_output, attn.o_proj)
    else:
        o_proj = partial(_stock_o_proj, attn.o_proj)
    return decomposed_attention(
        attn, x, st, qkv_index=qkv_index, o_proj=o_proj, store=store
    )


def _step_gemms(layer, x, st) -> Optional[LayerGemms]:
    """The layer's low-latency GEMMs when this step runs slot attn_ll on ``x``, else
    None: the slot serves this width and binds the layer, and the step's input and
    forward state are the ones it reproduces. Host state only, alike on every rank."""
    if not st.slots.attn_ll:
        return None
    built = st.bound.get("attention")
    gemms = None if built is None else built.layers.get(st.layer_id)
    if gemms is None or not gemms.runs(layer.self_attn, x):
        return None
    return gemms


def low_latency_qkv(layer, x, st) -> bool:
    """Whether this step runs ``layer``'s q|k|v|index GEMM on the low-latency kernel
    with input ``x`` (``step``'s own decision, host state only, alike on every rank).
    On an MXFP8 pair (slot hop_mxfp8) that GEMM is then the attention's first kernel,
    with no quantize before it: the pair's producer, the attention-input boundary
    kernel, asks before it launches whether it may signal the GEMM's launch at its
    entry (``boundary.attn_input``)."""
    gemms = _step_gemms(layer, x, st)
    return gemms is not None and gemms.qkv_index is not None


def _fused_store(attn, st) -> Optional[Callable[..., bool]]:
    """Slot qk_store's fused q/k norm + RoPE + KV store where it runs this step's
    layer, else None."""
    if not st.slots.qk_store:
        return None
    bound = st.bound.get("qk_store")
    return None if bound is None else bound.layer_store(st.layer_id, attn, st)


def _stock_o_proj(o_proj, attn_output) -> torch.Tensor:
    output, _ = o_proj(attn_output)
    return output


def decomposed_attention(
    attn,
    x,
    st,
    *,
    qkv_index: Callable[[Any], torch.Tensor],
    o_proj: Callable[[torch.Tensor], torch.Tensor],
    store: Optional[Callable[..., bool]] = None,
) -> torch.Tensor:
    """``MiniMaxM3Attention.forward`` (forward_prepare + forward_core) of this
    configuration's branch: the fused q|k|v|index projection (not AITER), the combined
    q/k/index norm + RoPE with the FP8 q epilogue, a sparse layer without the index
    value head. ``qkv_index(x)`` gives the BF16 q|k|v|index rows and
    ``o_proj(attn_output)`` the output partial. ``store`` (slot qk_store), when it
    takes the step, launches the norm + RoPE fused with the NVFP4 K / V / index-K
    store, and the attention backend then runs without its store; else SGLang's norm
    + RoPE kernel runs and the backend stores."""
    from sglang.kernels.ops.attention.minimax_qknorm_rope import (
        minimax_qknorm_rope_grouped_fp8,
    )
    from sglang.srt.models.minimax_m3 import (
        _QKNORM_GROUP_INDEX_Q,
        _QKNORM_GROUP_MAIN_Q,
    )

    positions, forward_batch = st.positions, st.fb
    # forward_prepare
    fused_out = qkv_index(x)
    qkv = fused_out[:, : attn._fused_main_size]
    groups = attn._qknorm_groups()
    num_tokens = fused_out.shape[0]
    fp8_q = fused_out.new_empty(
        (num_tokens, attn.num_heads, attn.head_dim), dtype=torch.float8_e4m3fn
    )
    fp8_idx_q = fused_out.new_empty(
        (num_tokens, attn.num_idx_heads, attn.idx_head_dim), dtype=torch.float8_e4m3fn
    )
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    idx_qkv = fused_out[:, attn._fused_main_size :]
    idx_q, idx_k, idx_v = attn._split_index_qkv(idx_qkv)
    # forward_core's views of the rows the norm + RoPE writes in place
    k = k.view(k.shape[0], attn.num_kv_heads, attn.head_dim)
    v = v.view(v.shape[0], attn.num_kv_heads, attn.head_dim)
    idx_k = idx_k.reshape(idx_k.shape[0], 1, attn.idx_head_dim)
    if idx_v is not None:
        idx_v = idx_v.reshape(idx_v.shape[0], 1, attn.idx_head_dim)
    norm_rope = (
        fused_out,
        groups,
        attn.rotary_emb.cos_sin_cache,
        positions,
        attn.q_norm.variance_epsilon,
    )
    stored = store is not None and store(
        *norm_rope, fp8_q=fp8_q, fp8_idx_q=fp8_idx_q, k=k, v=v, idx_k=idx_k
    )
    if not stored:
        minimax_qknorm_rope_grouped_fp8(
            *norm_rope,
            fp8_a=fp8_q,
            group_a=_QKNORM_GROUP_MAIN_Q,
            fp8_b=fp8_idx_q,
            group_b=_QKNORM_GROUP_INDEX_Q,
        )
    q, idx_q = fp8_q, fp8_idx_q
    # forward_core
    q = q.view(q.shape[0], attn.num_heads, attn.head_dim)
    idx_q = idx_q.reshape(idx_q.shape[0], attn.num_idx_heads, attn.idx_head_dim)
    # the backend's store is the only work save_kv_cache gates (qk_store.py)
    skip_store = {"save_kv_cache": False} if stored else {}
    _idx_o, attn_output = attn.attn(
        q,
        k,
        v,
        forward_batch,
        **skip_store,
        idx_q=idx_q,
        idx_k=idx_k,
        idx_v=idx_v,
    )
    return o_proj(attn_output)


class LowLatencyLinear:
    """An MXFP8 linear on the low-latency GEMM: its weight and swizzled scales as
    SGLang loaded them (read at each call, as the stock linear reads them), one pinned
    tactic per served row count."""

    def __init__(self, linear, tactics: dict[int, mxfp8_ll.Tactic]):
        self.linear = linear
        self.n, self.k = (int(d) for d in linear.weight.shape)
        self.tactics = dict(tactics)

    def reasons(self, x) -> list[str]:
        """Why ``x`` cannot be this GEMM's input ([] if it can); host state only."""
        if isinstance(x, Mxfp8SwizzledInput):
            rows = int(x.data.shape[0]) if x.data.dim() == 2 else -1
            reasons = mxfp8_ll.activation_reasons(x.data, x.scales, self.k)
        elif isinstance(x, torch.Tensor):
            rows = int(x.shape[0]) if x.dim() == 2 else -1
            reasons = (
                []
                if x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == self.k
                else [f"input {x.dtype} {tuple(x.shape)} is not bfloat16 [M, {self.k}]"]
            )
        else:
            return [f"input is a {type(x).__name__}"]
        if rows not in self.tactics:
            reasons.append(
                f"{rows} rows have no tactic (served: {sorted(self.tactics)})"
            )
        return reasons

    def __call__(self, x) -> torch.Tensor:
        """``x @ weight.T`` in BF16 [M, N]."""
        if isinstance(x, Mxfp8SwizzledInput):
            a, a_scale = x.data, x.scales
        else:
            from sglang.srt.layers.quantization.fp8_utils import (
                flashinfer_mxfp8_quantize,
            )

            # flashinfer_mxfp8_blockscaled_linear's quantize, with its arguments
            input_2d = x.view(-1, x.shape[-1])
            a, a_scale = flashinfer_mxfp8_quantize(
                input_2d, is_sf_swizzled_layout=True, alignment=32
            )
        return mxfp8_ll.linear(
            a,
            a_scale,
            self.linear.weight,
            self.linear.weight_scale_inv_swizzled,
            tactic=self.tactics[int(a.shape[0])],
        )


_WARNED: set[str] = set()


def _stock_because(reason: str) -> bool:
    """Log a step's fallback to the stock module once per reason; False."""
    if reason not in _WARNED:
        _WARNED.add(reason)
        logger.warning(
            "NVIDIA megakernel slot %s: %s; the layer runs SGLang's attention",
            SLOT,
            reason,
        )
    return False


def has_forward_hooks(*modules) -> bool:
    """Whether calling these modules runs hooks that the slot, calling none of them,
    would skip."""
    from torch.nn.modules import module as nn_module

    if getattr(nn_module, "_global_forward_hooks", None) or getattr(
        nn_module, "_global_forward_pre_hooks", None
    ):
        return True
    return any(m._forward_hooks or m._forward_pre_hooks for m in modules)


def _row_parallel_allocation(o_proj):
    """The allocation context ``RowParallelLinear.forward`` runs its quant method in."""
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.runtime_context import get_parallel

    if o_proj.use_dp_attention_reduce:
        return use_symmetric_memory(get_parallel().attn_tp_group)
    return use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    )


class LayerGemms:
    """One layer's GEMMs on the low-latency kernel: its q|k|v|index projection and its
    o_proj, each None where the slot leaves that GEMM to SGLang's linear."""

    def __init__(
        self,
        qkv_index: Optional[LowLatencyLinear] = None,
        o_proj: Optional[LowLatencyLinear] = None,
    ):
        self.qkv_index = qkv_index
        self.o_proj = o_proj

    def runs(self, attn, x) -> bool:
        """Whether this step's layer runs the slot (host state only, alike on every
        rank); else the stock module call runs."""
        from sglang.srt.runtime_context import get_forward

        reasons = [] if self.qkv_index is None else self.qkv_index.reasons(x)
        if reasons:
            return _stock_because("; ".join(reasons))
        if has_forward_hooks(attn, attn.fused_qkv_index_proj, attn.o_proj):
            return _stock_because("forward hooks on the attention or its linears")
        # RowParallelLinear.forward would reduce-scatter (sequence parallel) or
        # all-reduce (decode attention TP) the o_proj output
        if get_forward().sp_active:
            return _stock_because("sequence parallelism is active")
        if attn.o_proj.use_decode_attn_tp:
            return _stock_because("o_proj all-reduces for decode attention TP")
        return True

    def o_proj_output(self, o_proj, attn_output) -> torch.Tensor:
        """``o_proj(attn_output)``'s output: RowParallelLinear.forward of an input
        already split, no bias and no reduction."""
        if self.o_proj.reasons(attn_output):
            output, _ = o_proj(attn_output)
            return output
        with _row_parallel_allocation(o_proj):
            return self.o_proj(attn_output)


class AttnLowLatency:
    """The build of slot attn_ll: the GEMMs it runs on the low-latency kernel, the
    layers it binds and the widths it serves."""

    def __init__(
        self,
        *,
        layers: dict[int, LayerGemms],
        widths: tuple[int, ...],
        tactics: dict[str, dict[int, mxfp8_ll.Tactic]],
        reasons: list[str],
        skipped: dict[int, list[str]],
        gemms: tuple[str, ...] = DEFAULT_GEMMS,
    ):
        self.gemms = gemms
        self.layers = layers
        self.widths = widths
        self.tactics = tactics
        # why the slot serves no width ([] when it serves one), and the layers it
        # leaves to the stock module with their reasons
        self.reasons = reasons
        self.skipped = skipped


def build(model, widths: tuple[int, ...], slots: frozenset):
    """The slot's build (weights bound, kernels compiled and launched once), or None
    when its slot is not requested."""
    if SLOT not in slots:
        return None
    gemms = requested_gemms()
    if gemms != DEFAULT_GEMMS:
        logger.warning(
            "NVIDIA megakernel slot %s (test only, %s): GEMMs %s on the low-latency "
            "kernel",
            SLOT,
            GEMMS_ENV,
            list(gemms),
        )
    built = build_low_latency(model, widths, _GpuKernels(), gemms=gemms)
    only = _env_set(LAYERS_ENV)
    if only is not None:
        built.layers = {i: g for i, g in built.layers.items() if str(i) in only}
        logger.warning(
            "NVIDIA megakernel slot %s (test only, %s): layers %s",
            SLOT,
            LAYERS_ENV,
            sorted(built.layers),
        )
    _agree(built)
    return built


def _agree(built: AttnLowLatency) -> None:
    """Raise on every rank unless every TP rank built the same plan: the build decides
    from each rank's own GPU (FlashInfer's tactics, a launch of each), and every rank
    must run the same kernels."""
    from sgl_nvidia_mega.minimax_m31.runtime.agree import agree_tp

    # the tactics are keyed by the GEMMs the slot runs
    agree_tp(
        (built.widths, tuple(sorted(built.layers)), built.tactics),
        what=f"slot {SLOT} plans (widths, layers, tactics)",
    )


def serves(bound, rows: int) -> bool:
    """Whether the slot's kernels serve a step of this many rows."""
    return bound is not None and rows in bound.widths


class _GpuKernels:
    """The build's GPU side (the CPU tests pass a fake)."""

    def __init__(self):
        self.device = torch.device("cuda", torch.cuda.current_device())

    def sm(self) -> int:
        return mxfp8_ll.sm_version(self.device)

    def reasons(self, sm: int) -> list[str]:
        reasons = []
        if not mxfp8_ll.supported(sm):
            reasons.append(
                f"FlashInfer's {mxfp8_ll.BACKEND} MXFP8 GEMM is not available on "
                f"sm_{sm} (mm_mxfp8.is_backend_supported, CuTe DSL)"
            )
        try:
            from sglang.srt.layers.quantization.fp8_utils import (  # noqa: F401
                flashinfer_mxfp8_quantize,
            )
        except ImportError:
            reasons.append("SGLang's FlashInfer MXFP8 quantize is not available")
        return reasons

    def offered(self, n: int, k: int, rows: int) -> list[mxfp8_ll.Tactic]:
        return mxfp8_ll.offered_tactics(n, k, rows)

    def prepare(self, tactic: mxfp8_ll.Tactic, linear, rows: int) -> None:
        """Compile the tactic, then launch it once on ``linear``'s weights (raises on
        a compile, launch or output failure)."""
        mxfp8_ll.compiled(tactic, self.device)
        mxfp8_ll.probe(
            linear.weight, linear.weight_scale_inv_swizzled, tactic=tactic, rows=rows
        )


def build_low_latency(
    model, widths: tuple[int, ...], kernels, gemms: tuple[str, ...] = DEFAULT_GEMMS
) -> AttnLowLatency:
    """Bind every layer whose attention is the branch the slot reproduces, pick the
    pinned tactic of each of ``gemms`` per width (1-8 rows), and compile and launch
    each tactic once; a width without a tactic FlashInfer offers, or whose tactic
    fails, for one of ``gemms`` runs the stock module."""
    m = model.model
    sm = kernels.sm()
    reasons = list(kernels.reasons(sm))
    rows = [w for w in widths if w <= mxfp8_ll.MAX_ROWS]
    if not rows:
        reasons.append(
            f"no width of {list(widths)} has at most {mxfp8_ll.MAX_ROWS} rows"
        )
    attns: dict[int, Any] = {}
    skipped: dict[int, list[str]] = {}
    for i in range(m.start_layer, m.end_layer):
        why = layer_reasons(m.layers[i])
        if why:
            skipped[i] = why
        else:
            attns[i] = m.layers[i].self_attn
    if not attns:
        reasons.append("no layer's attention is the branch the slot reproduces")
    tactics: dict[str, dict[int, mxfp8_ll.Tactic]] = {name: {} for name in gemms}
    unserved: dict[int, list[str]] = {}
    if not reasons:
        tactics, unserved = _pick_tactics(kernels, sm, rows, gemms)
        _prepare_tactics(kernels, tactics, rows, unserved, next(iter(attns.values())))
    served = () if reasons else tuple(w for w in rows if w not in unserved)
    if not reasons and not served:
        reasons += [f"rows {ws}: {why}" for why, ws in _widths_by_reason(unserved)]
    layers: dict[int, LayerGemms] = {}
    if not reasons:
        for i, attn in attns.items():
            linears = {"qkv_index": attn.fused_qkv_index_proj, "o_proj": attn.o_proj}
            layers[i] = LayerGemms(
                **{
                    name: LowLatencyLinear(
                        linears[name], {w: tactics[name][w] for w in served}
                    )
                    for name in gemms
                }
            )
    _log_build(sm, served, layers, tactics, reasons, unserved, skipped)
    return AttnLowLatency(
        layers=layers,
        widths=served,
        tactics={name: {w: tactics[name][w] for w in served} for name in gemms},
        reasons=reasons,
        skipped=skipped,
        gemms=gemms,
    )


def _pick_tactics(kernels, sm: int, rows: list[int], gemms: tuple[str, ...]):
    """Each of ``gemms``' pinned tactic per width when FlashInfer offers it; the
    widths without one, with the reasons."""
    tactics: dict[str, dict[int, mxfp8_ll.Tactic]] = {name: {} for name in gemms}
    unserved: dict[int, list[str]] = {}
    offered: dict[tuple[int, int, int], list[mxfp8_ll.Tactic]] = {}
    for w in rows:
        bucket = mxfp8_ll.row_bucket(w)
        for name in gemms:
            n, k = SHAPES[name]
            tactic = mxfp8_ll.pinned_tactic(sm, n, k, w)
            if tactic is None:
                unserved.setdefault(w, []).append(
                    f"no pinned {name} tactic for sm_{sm} at bucket {bucket}"
                )
                continue
            if (n, k, bucket) not in offered:
                offered[n, k, bucket] = list(kernels.offered(n, k, bucket))
            if tactic not in offered[n, k, bucket]:
                unserved.setdefault(w, []).append(
                    f"FlashInfer does not offer the pinned {name} tactic {tactic} "
                    f"(N {n}, K {k}, bucket {bucket})"
                )
                continue
            tactics[name][w] = tactic
    return tactics, unserved


def _prepare_tactics(kernels, tactics, rows, unserved, attn) -> None:
    """Compile each GEMM's tactic and launch it once, at the most rows it serves, on
    ``attn``'s weights (at build, outside any graph). A tactic that fails leaves its
    widths to the stock module (``unserved``, with the error)."""
    linears = {"qkv_index": attn.fused_qkv_index_proj, "o_proj": attn.o_proj}
    for name in tactics:
        widths_of: dict[mxfp8_ll.Tactic, list[int]] = {}
        for w in rows:
            if w not in unserved:
                widths_of.setdefault(tactics[name][w], []).append(w)
        for tactic, ws in widths_of.items():
            try:
                kernels.prepare(tactic, linears[name], max(ws))
            except Exception as e:  # compile, launch or output: the stock module
                logger.warning(
                    "NVIDIA megakernel slot %s: the %s tactic %s failed at build",
                    SLOT,
                    name,
                    tactic,
                    exc_info=True,
                )
                why = (
                    f"the pinned {name} tactic {tactic} failed at build: "
                    f"{type(e).__name__}: {str(e)[:300]}"
                )
                for w in ws:
                    unserved.setdefault(w, []).append(why)


def _widths_by_reason(unserved: dict[int, list[str]]) -> list[tuple[str, list[int]]]:
    """The unserved widths grouped by their reasons."""
    grouped: dict[str, list[int]] = {}
    for w, why in unserved.items():
        grouped.setdefault("; ".join(why), []).append(w)
    return list(grouped.items())


def _log_build(sm, served, layers, tactics, reasons, unserved, skipped) -> None:
    by_reason: dict[str, list[int]] = {}
    for i, why in skipped.items():
        by_reason.setdefault("; ".join(why), []).append(i)
    for why, ids in by_reason.items():
        logger.warning(
            "NVIDIA megakernel slot %s: layers %s run SGLang's attention: %s",
            SLOT,
            ids,
            why,
        )
    if not served:
        logger.warning(
            "NVIDIA megakernel slot %s serves no width: %s", SLOT, "; ".join(reasons)
        )
        return
    for why, ws in _widths_by_reason(unserved):
        logger.warning(
            "NVIDIA megakernel slot %s: rows %s run SGLang's attention: %s",
            SLOT,
            ws,
            why,
        )
    logger.info(
        "NVIDIA megakernel slot %s: %d layers on FlashInfer's low-latency MXFP8 GEMM "
        "at %s rows (sm_%d); tactics (cta_k, stages, sfb stages, split-K): %s",
        SLOT,
        len(layers),
        list(served),
        sm,
        "; ".join(
            f"{name} {per_width}"
            for name, per_width in (
                (name, {w: tactics[name][w] for w in served}) for name in tactics
            )
        ),
    )


def layer_reasons(layer) -> list[str]:
    """Why a layer's attention is not the configuration the slot reproduces ([] if
    it is): the branch ``decomposed_attention`` rebuilds (``branch_reasons``), with
    the fused q|k|v|index projection and o_proj two MXFP8 linears on FlashInfer's
    MXFP8 path with the expected shapes and no bias, o_proj neither reducing its
    output nor splitting its input (the slot's GEMMs replace their forwards)."""
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.models.minimax_m3 import _FusedQKVIndexProj

    reasons = branch_reasons(layer)
    if reasons:
        return reasons
    attn = layer.self_attn
    fused = attn._fused_qkv_index
    if not isinstance(fused, _FusedQKVIndexProj):
        reasons.append(f"fused_qkv_index_proj is a {type(fused).__name__}")
    else:
        reasons += _overridden(fused, _FusedQKVIndexProj, ("forward",))
        reasons += _mxfp8_linear_reasons(
            "fused_qkv_index_proj", fused, fused._qm, SHAPES["qkv_index"]
        )
    o_proj = attn.o_proj
    if not isinstance(o_proj, RowParallelLinear):
        reasons.append(f"o_proj is a {type(o_proj).__name__}")
    else:
        reasons += _overridden(o_proj, RowParallelLinear, ("forward",))
        if o_proj.reduce_results or not o_proj.input_is_parallel:
            reasons.append("o_proj reduces its output or splits its input")
        reasons += _mxfp8_linear_reasons(
            "o_proj", o_proj, o_proj.quant_method, SHAPES["o_proj"]
        )
    return reasons


def branch_reasons(layer) -> list[str]:
    """Why a layer's attention is not the branch ``decomposed_attention`` rebuilds
    ([] if it is): the classes whose forward it rebuilds, the fused q|k|v|index
    projection with the combined q/k/index norm + RoPE and its FP8 q epilogue, a
    sparse layer without the index value head."""
    from sglang.srt.models.minimax_m3 import MiniMaxM3Attention, _is_npu

    attn = getattr(layer, "self_attn", None)
    if not isinstance(attn, MiniMaxM3Attention):
        return [f"self_attn is a {type(attn).__name__}, not MiniMaxM3Attention"]
    reasons = []
    if _is_npu:
        reasons.append("the attention runs its NPU forward")
    reasons += _overridden(
        attn,
        MiniMaxM3Attention,
        ("forward", "forward_prepare", "forward_core", "_aiter_sparse_qknorm_cache"),
    )
    if not (attn.is_sparse_attention_layer and attn.disable_index_value):
        reasons.append("not a sparse attention layer without the index value head")
    if getattr(attn, "index_o_proj", None) is not None:
        reasons.append("the layer has an index o_proj")
    if attn._can_use_aiter_fused_qknorm_static:
        reasons.append("the AITER fused q/k norm path is on")
    if not (attn._combined_qknorm_ok and attn._fp8_q_epilogue_ok):
        reasons.append(
            "not the combined q/k/index norm + RoPE with the FP8 q epilogue "
            "(_combined_qknorm_ok, _fp8_q_epilogue_ok)"
        )
    fused = attn._fused_qkv_index
    if fused is None or getattr(attn, "fused_qkv_index_proj", None) is not fused:
        reasons.append("no fused q|k|v|index projection")
    return reasons


def _overridden(obj, cls, names) -> list[str]:
    """The methods of ``cls`` that ``obj`` does not run as ``cls`` defines them:
    overridden in its class, or replaced on the instance (e.g. the wrapper SGLang's
    kernel API logging sets on a quant method's ``apply``)."""
    reasons = []
    for name in names:
        if name in getattr(obj, "__dict__", {}):
            reasons.append(f"{type(obj).__name__}.{name} is replaced on the instance")
        elif getattr(type(obj), name, None) is not getattr(cls, name):
            reasons.append(f"{type(obj).__name__}.{name} is not {cls.__name__}.{name}")
    return reasons


def _mxfp8_linear_reasons(name: str, linear, qm, shape) -> list[str]:
    """Why a linear is not an MXFP8 linear on FlashInfer's MXFP8 path whose GEMM the
    low-latency kernel can replace ([] if it is)."""
    from functools import partial

    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )

    if not isinstance(qm, Fp8LinearMethod):
        return [f"{name}'s quant method is a {type(qm).__name__}, not Fp8LinearMethod"]
    reasons = [
        f"{name}'s quant method: {why}"
        for why in _overridden(qm, Fp8LinearMethod, ("apply",))
    ]
    if not (
        qm.use_mxfp8
        and not qm.use_marlin
        and not qm.convert_mxfp8_to_block
        and not qm.block_fp8_as_mxfp8
    ):
        reasons.append(f"{name} is not an MXFP8 linear")
    backend = qm.mxfp8_dense_backend
    if backend is None or not (
        backend.is_flashinfer_cutedsl() or backend.is_flashinfer_cutlass()
    ):
        reasons.append(f"{name}'s MXFP8 backend {backend} is not FlashInfer's")
    fn = qm.w8a8_mxfp8_linear
    if not (isinstance(fn, partial) and fn.func is flashinfer_mxfp8_blockscaled_linear):
        reasons.append(
            f"{name}'s MXFP8 linear is not flashinfer_mxfp8_blockscaled_linear"
        )
    if getattr(linear, "bias", None) is not None:
        reasons.append(f"{name} has a bias")
    reasons += [
        f"{name}: {why}"
        for why in mxfp8_ll.weight_reasons(
            getattr(linear, "weight", None),
            getattr(linear, "weight_scale_inv_swizzled", None),
            shape,
        )
    ]
    return reasons
