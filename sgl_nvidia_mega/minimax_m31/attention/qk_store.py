# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/minimax_m3.py
# (MiniMaxM3Attention.forward_prepare, the FP8 q epilogue branch),
# python/sglang/srt/layers/attention/minimax_sparse_nvfp4_backend.py
# (MiniMaxSparseNvfp4AttnBackend._forward, the KV write) and
# python/sglang/srt/mem_cache/minimax_m3_nvfp4_pool.py (set_fused_kv_index_buffer).
"""Slot ``qk_store`` (bitwise): the attention's q / k / index norm + RoPE with the FP8
q epilogue and its NVFP4 K / V / index-K page store in one launch per layer.

SGLang launches two kernels there: ``fused_gemma_qknorm_rope_kernel``
(``minimax_qknorm_rope_grouped_fp8`` in ``MiniMaxM3Attention.forward_prepare``) and,
in the NVFP4 attention backend's forward, ``nvfp4_store_kv_index_kernel``
(``MiniMaxM3Nvfp4KVPool.set_fused_kv_index_buffer``). With the slot, SGLang's KV pool
launches ``minimax_qknorm_rope_store_fp8`` instead, on its own buffers
(``MiniMaxM3Nvfp4KVPool.set_kv_index_buffer_fused_qknorm_rope``): each warp runs the
first kernel's code for its head unchanged, then the warps of the main-K and index-K
heads store those heads (and V beside main K) with the second kernel's quantizer and
addressing, on the arguments the pool hands the second kernel. The attention backend
then runs with ``save_kv_cache=False``, which skips its store and nothing else
(``MiniMaxSparseNvfp4AttnBackend._forward``). So the FP8 q and index q, the
q|k|v|index rows normed in place and every written page byte and block scale are
SGLang's, padded rows included: they write where their ``out_cache_loc`` points, as
SGLang's store does.

The step reaches the pool only through SGLang's binding: ``Step.view.kv_pool`` is the
running forward's ``KvPoolView`` (``sglang.srt.kernel_plugins.nvidia.binding``), whose
``set_kv_index_buffer_fused_qknorm_rope`` is the fused K / V / index-K write SGLang
tolerates; the slot reads neither the attention backend nor the pool object. That
pool is the NVFP4 backend's (``MiniMaxSparseNvfp4AttnBackend``, which takes no other
pool): its ``token_to_kv_pool``, which the view wraps, is the ``kv_pool`` its store
writes.

The attention runs decomposed (``attention.decomposed_attention``) around SGLang's own
linears, or around slot attn_ll's GEMMs when both slots run.

``build`` binds each layer whose attention is that branch and checks on the GPU, once,
that the fused kernel writes what SGLang's two kernels write (``verify_against_stock``,
on caches laid out as the pool's page rows); it serves every width when every rank
finds both, else none, and none when SGLang's KV pool will not store fused
(``store_reasons``), so a requested slot that could never run refuses the server at
build. A step runs SGLang's two kernels for a layer whose forward state is not the one
the slot reproduces (host state only, checked before the layer's first kernel), and
where the pool would store per buffer: its fused store then launches nothing and
returns False (``LayerStore``).

PDL: the fused kernel reads only the norm weights and the positions before its PDL
wait (in the q/k norm kernel's code it runs first) and triggers its dependents where
that kernel does, before its writes; the cache slots, the K / V rows and the global
scale are read after the wait. Its successor must then read nothing the fused kernel
writes before it has completed, and does not: in the FP8 indexer fallback it is the
full-page index-K dequant, launched without PDL (or with PDL where SGLang launches its
MSA-v1 dequant kernels so, reading after its own wait); with MSA's Q8KV4 indexer (its
QMUL4 path, CUDA 13.4 and later) it is that indexer, which reads the FP8 index q and the index-K pages and launches without
PDL, as the served verify graph's nsys shows (``boundary.HopMxfp8``). The GPU test
(test/kernels/test_minimax_m31_qk_store.py) checks the Q8KV4 indexer directly: behind
a predecessor that lets it launch at once and writes its index q, or every index-K
page, only after a 2 ms spin, its top-k is the one of a run on the inputs written
beforehand.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Optional

import msgspec
import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # the running forward's KV pool, the slot's only way to it (Step.view.kv_pool),
    # and the pool's fused q/k norm + RoPE + NVFP4 K / V / index-K write
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "KvPoolView",
        "kv_pool",
        "set_kv_index_buffer_fused_qknorm_rope",
    ),
    "srt/mem_cache/minimax_m3_nvfp4_pool.py": (
        # the keyword arguments LayerStore passes it
        "set_kv_index_buffer_fused_qknorm_rope(fp8_a)",
        "set_kv_index_buffer_fused_qknorm_rope(group_a)",
        "set_kv_index_buffer_fused_qknorm_rope(fp8_b)",
        "set_kv_index_buffer_fused_qknorm_rope(group_b)",
        "set_kv_index_buffer_fused_qknorm_rope(k_group)",
        "set_kv_index_buffer_fused_qknorm_rope(index_k_group)",
        "set_kv_index_buffer_fused_qknorm_rope(cache_k)",
        "set_kv_index_buffer_fused_qknorm_rope(cache_v)",
        "set_kv_index_buffer_fused_qknorm_rope(cache_idx_k)",
        "set_kv_index_buffer_fused_qknorm_rope(k_scale)",
        "set_kv_index_buffer_fused_qknorm_rope(v_scale)",
        "set_kv_index_buffer_fused_qknorm_rope(idx_k_scale)",
        # the build's check: one layer's page rows, as the pool lays them out
        "_index_page_components",
        "_main_page_components",
        "_page_row_layout",
        # the page-row components cache_views views (_CACHE_BUFFERS)
        '"k_buffer"',
        '"k_scale_buffer"',
        '"v_buffer"',
        '"v_scale_buffer"',
    ),
    "srt/model_executor/forward_batch_info.py": (
        "out_cache_loc",
        "forward_mode",
        "is_decode",
        "is_target_verify",
    ),
    # the attention's RadixAttention, and the fields of it the backend's store reads
    "srt/layers/radix_attention.py": (
        "RadixAttention",
        "forward",
        "tp_k_head_num",
        "qk_head_dim",
        "tp_v_head_num",
        "v_head_dim",
        "k_scale_float",
        "v_scale_float",
        "idx_k_scale_float",
    ),
    "srt/models/minimax_m3.py": (
        "_QKNORM_GROUP_INDEX_Q",
        "_QKNORM_GROUP_MAIN_Q",
        "self_attn",
        "attn",
        "head_dim",
        "idx_head_dim",
        "num_heads",
        "num_kv_heads",
        "num_idx_heads",
        "q_size",
        "kv_size",
        "_qknorm_group_meta",
        "_qknorm_groups",
        "_fused_main_size",
        "_split_index_qkv",
        "rotary_emb",
        "q_norm",
        "start_layer",
        "end_layer",
        "layers",
    ),
    "srt/models/minimax_m3_vl.py": ("model",),
    "srt/layers/rotary_embedding/*.py": ("cos_sin_cache",),
    "srt/layers/layernorm.py": ("variance_epsilon",),
    # the switch of the pool's fused store (store_reasons)
    "srt/environ.py": ("envs", "SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE"),
    "srt/runtime_context.py": ("get_platform", '"is_sm100"'),
    # the build's check: SGLang's two kernels and the fused one, called by keyword
    "kernels/ops/attention/minimax_qknorm_rope.py": (
        "minimax_qknorm_rope_grouped_fp8",
        "minimax_qknorm_rope_grouped_fp8(fp8_a)",
        "minimax_qknorm_rope_grouped_fp8(group_a)",
        "minimax_qknorm_rope_grouped_fp8(fp8_b)",
        "minimax_qknorm_rope_grouped_fp8(group_b)",
    ),
    "kernels/ops/attention/minimax_qknorm_rope_store.py": (
        "minimax_qknorm_rope_store_fp8",
        "minimax_qknorm_rope_store_fp8(fp8_a)",
        "minimax_qknorm_rope_store_fp8(group_a)",
        "minimax_qknorm_rope_store_fp8(fp8_b)",
        "minimax_qknorm_rope_store_fp8(group_b)",
        "minimax_qknorm_rope_store_fp8(k_group)",
        "minimax_qknorm_rope_store_fp8(index_k_group)",
        "minimax_qknorm_rope_store_fp8(k)",
        "minimax_qknorm_rope_store_fp8(v)",
        "minimax_qknorm_rope_store_fp8(idx_k)",
        "minimax_qknorm_rope_store_fp8(k_payload)",
        "minimax_qknorm_rope_store_fp8(k_scale)",
        "minimax_qknorm_rope_store_fp8(v_payload)",
        "minimax_qknorm_rope_store_fp8(v_scale)",
        "minimax_qknorm_rope_store_fp8(idx_k_payload)",
        "minimax_qknorm_rope_store_fp8(idx_k_scale)",
        "minimax_qknorm_rope_store_fp8(indices)",
        "minimax_qknorm_rope_store_fp8(global_scale)",
        "minimax_qknorm_rope_store_fp8(page_size)",
    ),
    "kernels/ops/kvcache/nvfp4_store_kv_index.py": (
        "nvfp4_store_kv_index",
        "nvfp4_store_kv_index(k)",
        "nvfp4_store_kv_index(v)",
        "nvfp4_store_kv_index(idx_k)",
        "nvfp4_store_kv_index(k_payload)",
        "nvfp4_store_kv_index(k_scale)",
        "nvfp4_store_kv_index(v_payload)",
        "nvfp4_store_kv_index(v_scale)",
        "nvfp4_store_kv_index(idx_k_payload)",
        "nvfp4_store_kv_index(idx_k_scale)",
        "nvfp4_store_kv_index(indices)",
        "nvfp4_store_kv_index(global_scale)",
        "nvfp4_store_kv_index(page_size)",
    ),
}

logger = logging.getLogger(__name__)

SLOT = "qk_store"
# The k and index-k groups in MiniMaxM3Attention._qknorm_groups' order (q, k, index q,
# index k: layer_reasons checks it); the model names the two q groups.
GROUP_MAIN_K, GROUP_INDEX_K = 1, 3
# the fused kernel's specialization (minimax_qknorm_rope_store.HEAD_DIM)
HEAD_DIM = 128
# M3.1's page size (MSA_V1_PAGE_SIZE); the build's check uses caches of these pages
PAGE = 128


class LayerStore:
    """``decomposed_attention``'s ``store`` for one layer and step: the fused kernel,
    launched by the running forward's KV pool on its own buffers (``kv_pool``, the
    ``KvPoolView`` of ``Step.view``)."""

    def __init__(self, attn, kv_pool, forward_batch):
        self.attn = attn
        self.kv_pool = kv_pool
        self.forward_batch = forward_batch

    def __call__(
        self,
        fused_out: torch.Tensor,
        groups,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
        *,
        fp8_q: torch.Tensor,
        fp8_idx_q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_k: torch.Tensor,
    ) -> bool:
        """Have the pool launch the fused kernel and return True; or, where the pool
        would store these rows per buffer, return False, the pool having launched
        nothing (the caller then runs SGLang's norm + RoPE kernel and the backend
        stores)."""
        from sglang.srt.models.minimax_m3 import (
            _QKNORM_GROUP_INDEX_Q,
            _QKNORM_GROUP_MAIN_Q,
        )

        layer = self.attn.attn
        # MiniMaxSparseNvfp4AttnBackend._forward's store call, on the K / V views
        # RadixAttention.forward hands the backend, with the norm + RoPE before it
        stored = self.kv_pool.set_kv_index_buffer_fused_qknorm_rope(
            layer,
            self.forward_batch.out_cache_loc,
            fused_out,
            groups,
            cos_sin_cache,
            positions,
            eps,
            fp8_a=fp8_q,
            group_a=_QKNORM_GROUP_MAIN_Q,
            fp8_b=fp8_idx_q,
            group_b=_QKNORM_GROUP_INDEX_Q,
            k_group=GROUP_MAIN_K,
            index_k_group=GROUP_INDEX_K,
            cache_k=k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
            cache_v=v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
            cache_idx_k=idx_k,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
            idx_k_scale=layer.idx_k_scale_float,
        )
        if not stored:
            return _stock_because("the pool stores these K / V / index K per buffer")
        return True


class Bound(msgspec.Struct):
    """``build``'s result: the widths the slot serves (every built width, or none), the
    attention module of each bound layer, the layers left to SGLang with their
    reasons, and why the slot serves no width ("" when it serves)."""

    widths: tuple = ()
    attns: dict = msgspec.field(default_factory=dict)
    skipped: dict = msgspec.field(default_factory=dict)
    reason: str = ""

    def serves(self, rows: int) -> bool:
        return rows in self.widths

    def layer_store(self, layer_id: int, attn, st) -> Optional[LayerStore]:
        """The store of ``attn`` for this step, or None where SGLang's two kernels run
        it (the reason logged once)."""
        if not self.serves(st.rows):
            return None
        why = self._step_reason(layer_id, attn, st)
        if why:
            _stock_because(why)
            return None
        return LayerStore(attn, st.view.kv_pool, st.fb)

    def _step_reason(self, layer_id: int, attn, st) -> str:
        """Why the slot does not run this layer at this step ("" when it does): host
        state only, alike on every rank."""
        from sgl_nvidia_mega.minimax_m31.attention import has_forward_hooks

        if self.attns.get(layer_id) is not attn:
            return "the layer's attention is not the module bound at build"
        if has_forward_hooks(attn):
            return "forward hooks on the attention, which the slot would skip"
        mode = st.fb.forward_mode
        if not (mode.is_decode() or mode.is_target_verify()):
            return f"a {mode} step (the slot reproduces decode and verify steps)"
        positions = st.positions
        if not (
            isinstance(positions, torch.Tensor)
            and positions.dtype == torch.int64
            and tuple(positions.shape) == (st.rows,)
        ):
            return "the positions are not one int64 position a row"
        # the running forward's KV pool (SGLang's binding), the slot's only way to it;
        # where the pool stores per buffer, its fused store declines (LayerStore)
        view = getattr(st, "view", None)
        if view is None:
            return "no forward view (Step.view): the slot reaches no KV pool"
        if getattr(view.kv_pool, "set_kv_index_buffer_fused_qknorm_rope", None) is None:
            return "the running forward's KV pool has no fused q/k norm + RoPE store"
        return ""


_WARNED: set[str] = set()


def _stock_because(reason: str) -> bool:
    """Log a step's fallback to SGLang's two kernels once per reason; False."""
    if reason not in _WARNED:
        _WARNED.add(reason)
        logger.warning(
            "NVIDIA megakernel slot %s: %s; the layer runs SGLang's q/k norm and KV "
            "store kernels",
            SLOT,
            reason,
        )
    return False


# ------------------------------------------------------------------------- build


def layer_reasons(layer) -> list[str]:
    """Why a layer's attention is not the configuration the slot reproduces ([] if it
    is): the branch ``decomposed_attention`` rebuilds, SGLang's RadixAttention as its
    attention, 128-wide heads, and the k and index-k norm groups on exactly the heads
    the backend's store reads (``k`` after the q heads, ``idx_k`` after the index q
    head)."""
    from sglang.srt.layers.radix_attention import RadixAttention

    from sgl_nvidia_mega.minimax_m31.attention import branch_reasons

    reasons = branch_reasons(layer)
    if reasons:
        return reasons
    attn = layer.self_attn
    if type(attn.attn) is not RadixAttention or "forward" in vars(attn.attn):
        reasons.append(f"the attention is a {type(attn.attn).__name__}")
    if attn.head_dim != HEAD_DIM or attn.idx_head_dim != HEAD_DIM:
        reasons.append(
            f"heads of {attn.head_dim} / index heads of {attn.idx_head_dim}, not "
            f"{HEAD_DIM}"
        )
    nh, nkv, nih = attn.num_heads, attn.num_kv_heads, attn.num_idx_heads
    want = ((0, nh), (nh, nkv), (nh + 2 * nkv, nih), (nh + 2 * nkv + nih, 1))
    if tuple(tuple(g) for g in attn._qknorm_group_meta) != want:
        reasons.append(
            f"the q/k norm groups {attn._qknorm_group_meta} are not the "
            f"[q | k | v | index q | index k] heads {want}"
        )
    return reasons


def store_reasons() -> list[str]:
    """Why SGLang's KV pool will not run the fused NVFP4 store the slot replaces ([] if
    it will), as far as the build can tell: the pool is built after it (at KV cache
    sizing), so this reads the switch of the pool's own condition
    (``MiniMaxM3Nvfp4KVPool.use_nvfp4_fused_quantize_store``). The rest of that
    condition holds for M3.1 or is checked elsewhere: SM100 / SM103
    (``_GpuKernels.reasons``), 128-wide heads (``layer_reasons``), one page size and
    unit global scales for main and index K (the pool's construction). At each step
    the pool decides itself: its fused store launches nothing where it stores per
    buffer (``LayerStore``)."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE.get():
        return [
            "SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE is off: SGLang's KV pool "
            "stores per buffer, so the slot would replace no store"
        ]
    return []


class _GpuKernels:
    """The build's GPU side (the CPU tests pass a fake)."""

    def reasons(self) -> list[str]:
        from sglang.srt.runtime_context import get_platform

        if not get_platform().is_sm100:
            # the pool's own condition for its fused store
            return ["SGLang's fused NVFP4 KV store runs on SM100 / SM103 only"]
        return []

    def verify(self, attn, rows: Iterable[int]) -> list[str]:
        return verify_against_stock(attn, rows)


def build(model, widths: tuple[int, ...], slots: frozenset):
    """The slot's build (layers bound, the kernel compiled and checked), or None when
    its slot is not requested."""
    if SLOT not in slots:
        return None
    return build_qk_store(model, widths, _GpuKernels())


def build_qk_store(model, widths: tuple[int, ...], kernels) -> Bound:
    """Bind every layer whose attention is the branch the slot reproduces and check
    the fused kernel against SGLang's two kernels on this GPU; every width when every
    rank finds both and SGLang's KV pool will store fused, else none (a collective on
    the TP group)."""
    m = model.model
    attns: dict[int, Any] = {}
    skipped: dict[int, list[str]] = {}
    reasons: list[str] = []
    try:
        for i in range(m.start_layer, m.end_layer):
            why = layer_reasons(m.layers[i])
            if why:
                skipped[i] = why
            else:
                attns[i] = m.layers[i].self_attn
        if not attns:
            reasons.append("no layer's attention is the branch the slot reproduces")
        reasons += store_reasons()
        reasons += kernels.reasons()
        if not reasons:
            checked = sorted({1, min(8, max(widths)), max(widths)})
            reasons += kernels.verify(next(iter(attns.values())), checked)
    except Exception as e:  # gathered, then the slot is off on every rank
        reasons.append(f"{type(e).__name__}: {e}")
    gathered = _gather((reasons, sorted(attns)))
    failed = [
        f"rank {rank}: {why}"
        for rank, (rank_reasons, _) in enumerate(gathered)
        for why in rank_reasons
    ]
    if not failed and any(layers != sorted(attns) for _, layers in gathered):
        failed.append(
            "the ranks bound different layers: "
            + "; ".join(f"rank {r}: {layers}" for r, (_, layers) in enumerate(gathered))
        )
    _log_skipped(skipped)
    if failed:
        logger.warning(
            "NVIDIA megakernel: slot %s serves no width:\n  - %s",
            SLOT,
            "\n  - ".join(failed),
        )
        return Bound(skipped=skipped, reason="; ".join(failed))
    logger.info(
        "NVIDIA megakernel: slot %s on %d layers at %d-%d rows: the q/k norm + RoPE "
        "and the NVFP4 KV store in one kernel (checked against SGLang's two)",
        SLOT,
        len(attns),
        min(widths),
        max(widths),
    )
    return Bound(widths=tuple(widths), attns=attns, skipped=skipped)


def _log_skipped(skipped: dict[int, list[str]]) -> None:
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


def _gather(obj) -> list:
    """``obj`` from every TP rank, through the megakernel's startup gather."""
    from sgl_nvidia_mega.minimax_m31.runtime.agree import gather_tp

    return gather_tp(obj)


def serves(bound, rows: int) -> bool:
    """Whether the slot's kernel serves a step of this many rows."""
    return bound is not None and bound.serves(rows)


# ------------------------------------------------------- the check against SGLang


def store_views(attn, qkv: torch.Tensor):
    """The K ``[T, kv_heads, 128]``, V and index K ``[T, 128]`` views of q|k|v|index
    rows that SGLang's attention hands its KV store (forward_prepare's splits)."""
    rows = qkv.shape[0]
    _, k, v = qkv[:, : attn._fused_main_size].split(
        [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
    )
    _, idx_k, _ = attn._split_index_qkv(qkv[:, attn._fused_main_size :])
    return (
        k.view(rows, attn.num_kv_heads, attn.head_dim),
        v.view(rows, attn.num_kv_heads, attn.head_dim),
        idx_k.reshape(rows, attn.idx_head_dim),
    )


def check_rows(rows: int, width: int, gen: torch.Generator) -> torch.Tensor:
    """BF16 q|k|v|index rows for the check: Gaussian rows of magnitudes 1e-3 to 1e3,
    and from row 1 on rows that reach the store's other branches: all zeros (zero
    block maxima), one value throughout, one large outlier, values below the E4M3
    scale's underflow."""
    device = gen.device
    x = torch.randn(rows, width, generator=gen, device=device)
    x *= torch.logspace(-3, 3, rows, device=device)[:, None]
    special = (
        torch.zeros(width, device=device),
        torch.full((width,), -1.5, device=device),
        torch.zeros(width, device=device).index_fill_(
            0, torch.arange(7, width, 97, device=device), 3.0e4
        ),
        torch.full((width,), 1e-4, device=device),
    )
    for i, row in enumerate(special[: rows - 1]):
        x[i + 1] = row
    return x.to(torch.bfloat16)


# nvfp4_store_kv_index's cache buffers: the pool's slab and the page-row component
# whose per-layer view each one is (minimax_m3_nvfp4_pool)
_CACHE_BUFFERS = {
    "k_payload": ("main", "k_buffer"),
    "k_scale": ("main", "k_scale_buffer"),
    "v_payload": ("main", "v_buffer"),
    "v_scale": ("main", "v_scale_buffer"),
    "idx_k_payload": ("index", "k_buffer"),
    "idx_k_scale": ("index", "k_scale_buffer"),
}


def page_row_layout(kv_heads: int) -> dict:
    """The M3.1 pool's page row of one layer, per slab (main K / V, index K): its width
    in bytes and each component's byte offset and per-page shape, the layout
    ``minimax_m3_nvfp4_pool._allocate_page_rows`` allocates."""
    from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import (
        _index_page_components,
        _main_page_components,
        _page_row_layout,
    )

    layout = {}
    for slab, components in (
        ("main", _main_page_components(kv_heads, HEAD_DIM, PAGE)),
        ("index", _index_page_components(HEAD_DIM, PAGE)),
    ):
        offsets, row_bytes = _page_row_layout(components)
        layout[slab] = (
            row_bytes,
            {name: (off, shape) for (name, shape), off in zip(components, offsets)},
        )
    return layout


def random_page_rows(kv_heads: int, pages: int, gen: torch.Generator) -> dict:
    """One layer's page rows of the M3.1 KV pool, ``{"main": [pages, row bytes],
    "index": [pages, row bytes]}`` uint8, filled with random bytes."""
    return {
        slab: torch.randint(
            0,
            256,
            (pages, row_bytes),
            dtype=torch.uint8,
            generator=gen,
            device=gen.device,
        )
        for slab, (row_bytes, _) in page_row_layout(kv_heads).items()
    }


def cache_views(page_rows: dict, kv_heads: int) -> dict:
    """The six buffers ``nvfp4_store_kv_index`` writes, viewed in these page rows as
    the pool views its slab (``_allocate_page_rows``): a component's bytes of every
    row, so the page stride is the row width (payloads ``[pages, kv_heads, 128, 64]``
    and ``[pages, 128, 64]``, scales ``[.., 128, 8]``)."""
    layout = page_row_layout(kv_heads)
    views = {}
    for name, (slab, component) in _CACHE_BUFFERS.items():
        offset, shape = layout[slab][1][component]
        slab_rows = page_rows[slab]
        views[name] = slab_rows[:, offset : offset + math.prod(shape)].view(
            slab_rows.shape[0], *shape
        )
    return views


def verify_against_stock(attn, rows: Iterable[int], *, seed: int = 31) -> list[str]:
    """The fused kernel against SGLang's two (``minimax_qknorm_rope_grouped_fp8``,
    then ``nvfp4_store_kv_index``) on the same synthetic q|k|v|index rows, with
    ``attn``'s norm weights, RoPE cache and epsilon, positions from 0 to the cache's
    last, distinct cache slots, and caches of random bytes laid out as the pool's page
    rows (``cache_views``: the served strides, where the page stride is not the head
    stride): why it does not write what they write bit for bit (the rows normed in
    place, the FP8 q and index q, every byte of the page rows), [] when it does.
    Compiles the kernels; at build, outside graphs."""
    from sglang.kernels.ops.attention.minimax_qknorm_rope import (
        minimax_qknorm_rope_grouped_fp8,
    )
    from sglang.kernels.ops.attention.minimax_qknorm_rope_store import (
        minimax_qknorm_rope_store_fp8,
    )
    from sglang.kernels.ops.kvcache.nvfp4_store_kv_index import nvfp4_store_kv_index
    from sglang.srt.models.minimax_m3 import (
        _QKNORM_GROUP_INDEX_Q,
        _QKNORM_GROUP_MAIN_Q,
    )

    groups = attn._qknorm_groups()
    cos_sin_cache = attn.rotary_emb.cos_sin_cache
    eps = attn.q_norm.variance_epsilon
    device = cos_sin_cache.device
    width = attn._fused_main_size + (attn.num_idx_heads + 1) * attn.idx_head_dim
    reasons = []
    for n in rows:
        gen = torch.Generator(device=device).manual_seed(seed * 1000 + n)
        x = check_rows(n, width, gen)
        positions = torch.randint(
            0, cos_sin_cache.shape[0], (n,), generator=gen, device=device
        )
        positions[0] = cos_sin_cache.shape[0] - 1
        pages = n // PAGE + 2
        slots = torch.randperm(pages * PAGE, generator=gen, device=device)[:n]
        page_rows = random_page_rows(attn.num_kv_heads, pages, gen)
        unit = torch.ones(1, dtype=torch.float32, device=device)
        written = {}
        for arm in ("SGLang", "fused"):
            qkv = x.clone()
            arm_rows = {slab: t.clone() for slab, t in page_rows.items()}
            fp8_q = qkv.new_empty(
                (n, attn.num_heads, attn.head_dim), dtype=torch.float8_e4m3fn
            )
            fp8_idx_q = qkv.new_empty(
                (n, attn.num_idx_heads, attn.idx_head_dim), dtype=torch.float8_e4m3fn
            )
            k, v, idx_k = store_views(attn, qkv)
            norm = (qkv, groups, cos_sin_cache, positions, eps)
            fp8 = dict(
                fp8_a=fp8_q,
                group_a=_QKNORM_GROUP_MAIN_Q,
                fp8_b=fp8_idx_q,
                group_b=_QKNORM_GROUP_INDEX_Q,
            )
            store = dict(
                k=k,
                v=v,
                idx_k=idx_k,
                **cache_views(arm_rows, attn.num_kv_heads),
                indices=slots,
                global_scale=unit,
                page_size=PAGE,
            )
            if arm == "SGLang":
                minimax_qknorm_rope_grouped_fp8(*norm, **fp8)
                nvfp4_store_kv_index(**store)
            else:
                minimax_qknorm_rope_store_fp8(
                    *norm,
                    **fp8,
                    k_group=GROUP_MAIN_K,
                    index_k_group=GROUP_INDEX_K,
                    **store,
                )
            written[arm] = {
                "q|k|v|index rows": qkv,
                "FP8 q": fp8_q,
                "FP8 index q": fp8_idx_q,
                "main K / V page rows": arm_rows["main"],
                "index-K page rows": arm_rows["index"],
            }
        for name, want in written["SGLang"].items():
            got = written["fused"][name]
            if not torch.equal(got.view(torch.uint8), want.view(torch.uint8)):
                reasons.append(
                    f"at {n} rows the fused kernel and SGLang's two kernels write "
                    f"different {name}"
                )
    return reasons
