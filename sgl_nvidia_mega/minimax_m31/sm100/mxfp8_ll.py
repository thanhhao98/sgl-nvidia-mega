# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from sgl-nvidia-mega (Apache License 2.0, Copyright (c) 2026 NVIDIA
# CORPORATION & AFFILIATES and SGLang contributors):
# sgl_nvidia_mega/deepseek_v41/sm103/mxfp8_ll.py (the direct launch of FlashInfer's
# low-latency block-scaled GEMM). Kept: the support check, FlashInfer's tactic list,
# the compile per tactic and the launch on the current stream at 1-8 rows. Dropped:
# the DeepSeek-V4.1 tactics, the row-tile scale layout past 8 rows and the autotuner
# patch. Added: the MiniMax-M3.1 tactics and the operand checks. The kernel is
# FlashInfer's LowLatencyBlockscaledGemmKernel
# (flashinfer/gemm/kernels/cute_dsl/low_latency_blockscaled_gemm.py, BSD-3-Clause,
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES), compiled as FlashInfer's own
# ``cutedsl_low_latency`` mm_mxfp8 runner compiles it (FlashInfer 0.7.0.post1).
"""FlashInfer's low-latency MXFP8 GEMM, launched directly, for the M3.1 attention
projections at 1-8 rows (slot ``attn_ll``).

``LowLatencyBlockscaledGemmKernel`` swaps A and B: the weight [N, K] is operand A (128
output columns a CTA) and the <= 8 activation rows are operand B; a cluster of
``split_k`` CTAs splits K and reduces the partial sums through distributed shared
memory, with no workspace. Its weight (and weight scale) loads are issued before the
PDL wait, so they overlap the kernel before it; only the activation loads wait.

PDL ordering. SGLang's stock cute-dsl MXFP8 GEMMs wait before their first load and
trigger their dependents at their end. This kernel (FlashInfer 0.7.0's source):
warp 0 issues the weight and weight-scale TMA loads of its K tiles with no PDL wait
(``dma_a_warp``), one tile into each free stage, and triggers its dependents after
its last tile (``pdl_count`` -1); warp 1 executes ``griddepcontrol.wait`` before its
first activation load (``dma_b_warp``); the MMA warp needs both operands of a stage,
and the epilogue's stores (TMEM -> global, split-K through distributed shared memory)
follow the last MMA. So before its wait it reads only its weights, their scales and
its own parameters, and writes no global memory; a CTA has ``tiles_before_wait`` K
tiles of weights in flight by then.

Its predecessor may therefore signal it early: slot hop_mxfp8's boundary kernel does
so at its entry when this GEMM follows it (``boundary.attn_input``), and the weight
loads overlap the boundary's all-reduce. Its own trigger comes after its weight loads,
before its wait when every K tile of a CTA fits its stages (``tiles_before_wait`` =
``k_tiles_per_cta``). The next kernel can then start, and run up to its own PDL wait,
while the GEMM's predecessor (the activation quantize, or the boundary kernel) still
runs: PDL completion is not transitive through an early trigger. So a kernel launched
after this GEMM must read nothing, before its own PDL wait, that a kernel before the
GEMM writes; after its wait it may read anything (the GEMM completes only after its
own wait). ``attention`` lists the kernels that follow it and why they keep this.

The operands are SGLang's: the weight [N, K] e4m3 and ``weight_scale_inv_swizzled``
(UE8M0 in FlashInfer's 128x4 layout, ``block_scale_interleave``) as the MXFP8 linear
loaded them, and the activation rows [M, K] e4m3 with their UE8M0 scales in the same
layout (one 128-row atom at 1-8 rows), as ``mxfp8_quantize`` writes them.

A tactic is ``(cta_k, ab_stages, sfb_tmem_stages, split_k)``. FlashInfer's untuned
default, (128, 8, 1, 1), runs one CTA per 128 output columns (20 CTAs for the
q|k|v|index projection), so the tactics are pinned per (SM, N, K, row bucket)
(``PINNED_TACTICS``) and compiled at build; a covered step only launches.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any, Optional

import torch

BACKEND = "cutedsl_low_latency"
_GEMM_MODULE = "flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm"
# FlashInfer's runner takes at most 8 rows: one 8-row tile, whose activation scales
# are the first 128-row atom of the plain 128x4 layout.
MAX_ROWS = 8
SF_VEC = 32  # MXFP8: one UE8M0 scale per 32 elements of K

Tactic = tuple[int, int, int, int]

# One TP4 rank's attention projections, (N, K): the fused q|k|v|index projection
# (rows [q 2048 | k 128 | v 128 | index q 128 | index k 128]) and o_proj (a partial).
QKV_INDEX = (2560, 6144)
O_PROJ = (6144, 2048)
# FlashInfer's tuning buckets at 1-8 rows (the row count rounded up to a power of two)
ROW_BUCKETS = (1, 2, 4, 8)

# (cta_k, ab_stages, sfb_tmem_stages, split_k) per (SM, N, K, row bucket): sm 103 is
# GB300 (152 SMs) and B300, sm 100 is B200 (148 SMs).
# - sm 103: measured on GB300 with the GPU test's sweep mode
#   (SGLANG_M31_ATTN_LL_SWEEP=1): the best of every tactic FlashInfer offers, each timed
#   as quantize + GEMM in a CUDA-graph chain over 4 weight copies; stock = SGLang's
#   autotuned cute-dsl MXFP8 linear on the same chain. q|k|v|index 6.52 / 6.75 / 6.79 /
#   6.78 us at 1 / 2 / 4 / 8 rows (stock 6.84 / 6.89 / 7.16 / 8.54); o_proj 5.87 /
#   5.88 / 6.45 / 6.47 us (stock 5.78 / 6.16 / 6.82 / 6.77). At 1 row o_proj is
#   0.09 us slower than stock; with o_proj on the kernel (attention.GEMMS_ENV) a
#   width is served by both GEMMs or neither, and the pair still saves 0.23 us
#   there. That chain ran the GEMMs back to back with their weights resident in L2
#   (4 copies, 65 MB), not as served: the q|k|v|index pins keep 8 (1 row) and 10
#   (2-8 rows) of a CTA's 12 K tiles in flight before the PDL wait
#   (``tiles_before_wait``), where (128, 12, 1, 4) and (256, 6, 1, 4) keep all of
#   them (224,896 B of shared memory by FlashInfer's estimate). The GPU test's
#   serving-chain sweep (SGLANG_M31_ATTN_LL_SWEEP=serving: a predecessor that runs
#   ~6 us and signals at entry, the GEMM over 16 weight copies, the q/k norm after
#   it) re-measures the q|k|v|index pins where they now run; replace them from its
#   printed table, then confirm them with the 4-GPU hop_mxfp8 test's entry-trigger
#   timings (the real boundary kernel before the pinned tactic and the two above):
#   the sweep's stand-in spins on the clock, with none of the real kernel's HBM and
#   NVLink traffic for the early weight loads to contend with.
# - sm 100: a priori, not measured: q|k|v|index split-K 4 (80 CTAs, the whole K slice
#   in flight before the PDL wait), o_proj split-K 2 (96 CTAs).
PINNED_TACTICS: dict[tuple[int, int, int, int], Tactic] = {
    (103, 2560, 6144, 1): (128, 8, 1, 4),
    (103, 2560, 6144, 2): (128, 10, 1, 4),
    (103, 2560, 6144, 4): (128, 10, 1, 4),
    (103, 2560, 6144, 8): (128, 10, 1, 4),
    (103, 6144, 2048, 1): (128, 11, 1, 1),
    (103, 6144, 2048, 2): (128, 11, 1, 1),
    (103, 6144, 2048, 4): (128, 11, 1, 1),
    (103, 6144, 2048, 8): (128, 8, 1, 2),
    (100, 2560, 6144, 1): (256, 6, 1, 4),
    (100, 2560, 6144, 2): (256, 6, 1, 4),
    (100, 2560, 6144, 4): (256, 6, 1, 4),
    (100, 2560, 6144, 8): (256, 6, 1, 4),
    (100, 6144, 2048, 1): (256, 4, 1, 2),
    (100, 6144, 2048, 2): (256, 4, 1, 2),
    (100, 6144, 2048, 4): (256, 4, 1, 2),
    (100, 6144, 2048, 8): (256, 4, 1, 2),
}

# (device index, tactic) -> the compiled kernel
_COMPILED: dict[tuple[int, Tactic], Any] = {}


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def row_bucket(rows: int) -> int:
    """FlashInfer's tuning bucket of a row count: rounded up to a power of two."""
    return 1 << (rows - 1).bit_length()


def k_tiles_per_cta(tactic: Tactic, k: int) -> int:
    """The K tiles one CTA of this tactic loads: its split's share of K / cta_k."""
    cta_k, _, _, split_k = tactic
    return -(-(k // cta_k) // split_k)


def tiles_before_wait(tactic: Tactic, k: int) -> int:
    """The K tiles of weights a CTA of this tactic has in flight before its PDL wait:
    its weight warp fills each free stage without waiting; past ``ab_stages`` it needs
    the MMA to free a stage, which needs the activations, loaded after the wait."""
    return min(tactic[1], k_tiles_per_cta(tactic, k))


def scale_bytes(rows: int, k: int) -> int:
    """UE8M0 bytes of ``rows`` rows in FlashInfer's 128x4 layout (rows padded to 128,
    K / 32 to 4): what ``mxfp8_quantize`` allocates, and what the kernel reads."""
    return _round_up(rows, 128) * _round_up(k // SF_VEC, 4)


def sm_version(device: Optional[torch.device] = None) -> int:
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


@lru_cache(maxsize=None)
def supported(sm: int) -> bool:
    """Whether FlashInfer has the low-latency MXFP8 GEMM for this SM, and the CuTe DSL
    it is written in imports."""
    try:
        from flashinfer import mm_mxfp8

        importlib.import_module(_GEMM_MODULE)
        return bool(mm_mxfp8.is_backend_supported(BACKEND, sm))
    except Exception:
        return False


def pinned_tactic(sm: int, n: int, k: int, rows: int) -> Optional[Tactic]:
    """The pinned tactic of this shape at this row count, or None (not served)."""
    if not 0 < rows <= MAX_ROWS:
        return None
    return PINNED_TACTICS.get((sm, n, k, row_bucket(rows)))


def offered_tactics(n: int, k: int, rows: int) -> list[Tactic]:
    """FlashInfer's tactic list for an MXFP8 (N, K) problem of ``rows`` rows, in its
    default order (its runner's problem is (N, M, K, 1): it computes out.T)."""
    import cutlass

    ll_gemm = importlib.import_module(_GEMM_MODULE)
    e4m3, e8m0 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
    return [
        tuple(t)
        for t in ll_gemm.autotune_tactics(
            (n, rows, k, 1), e4m3, e4m3, e8m0, SF_VEC, cutlass.BFloat16
        )
    ]


def weight_reasons(
    weight: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    shape: tuple[int, int],
) -> list[str]:
    """Why an MXFP8 weight and its swizzled scales cannot be operand A ([] if they
    can): an e4m3 [N, K] row-major weight and its 128x4 UE8M0 scales."""
    n, k = shape
    if weight is None:
        return ["no weight"]
    reasons = []
    if weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != (n, k):
        reasons.append(
            f"weight {weight.dtype} {tuple(weight.shape)} is not float8_e4m3fn {shape}"
        )
    if not weight.is_contiguous() or weight.data_ptr() % 16:
        reasons.append("weight is not contiguous and 16-byte aligned")
    if k % 128:
        reasons.append(f"K {k} is not a multiple of 128")
    if weight_scale is None:
        return reasons + ["no weight_scale_inv_swizzled (FlashInfer's 128x4 layout)"]
    if weight_scale.element_size() != 1 or weight_scale.numel() != scale_bytes(n, k):
        reasons.append(
            f"weight_scale_inv_swizzled {weight_scale.dtype} of {weight_scale.numel()} "
            f"elements is not {scale_bytes(n, k)} UE8M0 bytes (the 128x4 layout)"
        )
    if not weight_scale.is_contiguous() or weight_scale.data_ptr() % 32:
        reasons.append(
            "weight_scale_inv_swizzled is not contiguous and 32-byte aligned"
        )
    return reasons


def activation_reasons(a: torch.Tensor, a_scale: torch.Tensor, k: int) -> list[str]:
    """Why MXFP8 rows and their scales cannot be operand B ([] if they can):
    e4m3 [M, K] row-major, 1 <= M <= 8, UE8M0 scales in the 128x4 layout."""
    reasons = []
    if a.dtype != torch.float8_e4m3fn or a.dim() != 2 or a.shape[1] != k:
        reasons.append(
            f"rows {a.dtype} {tuple(a.shape)} are not float8_e4m3fn [M, {k}]"
        )
    elif not 0 < a.shape[0] <= MAX_ROWS:
        reasons.append(f"{a.shape[0]} rows (the kernel takes 1-{MAX_ROWS})")
    if not a.is_contiguous() or a.data_ptr() % 16:
        reasons.append("rows are not contiguous and 16-byte aligned")
    if a_scale.element_size() != 1 or a_scale.numel() < scale_bytes(MAX_ROWS, k):
        reasons.append(
            f"row scales {a_scale.dtype} of {a_scale.numel()} elements are not one "
            f"128-row atom of UE8M0 bytes ({scale_bytes(MAX_ROWS, k)})"
        )
    if not a_scale.is_contiguous() or a_scale.data_ptr() % 16:
        reasons.append("row scales are not contiguous and 16-byte aligned")
    return reasons


def _ptr(dtype, address: int, align: int):
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


def _device(device: Optional[torch.device]) -> torch.device:
    device = torch.device("cuda") if device is None else torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def compiled(tactic: Tactic, device: Optional[torch.device] = None):
    """The kernel of this tactic, compiled once per device (at build: a covered step
    must find it here)."""
    device = _device(device)
    key = (device.index, tuple(tactic))
    fn = _COMPILED.get(key)
    if fn is None:
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream

        ll_gemm = importlib.import_module(_GEMM_MODULE)
        cta_k, ab_stages, sfb_tmem_stages, split_k = tactic
        # as FlashInfer's cutedsl_low_latency runner builds it for mm_mxfp8 (PDL on)
        gemm = ll_gemm.LowLatencyBlockscaledGemmKernel(
            acc_dtype=cutlass.Float32,
            mma_tiler_mnk=(128, 8, cta_k),
            num_ab_stage=ab_stages,
            num_sfb_tmem_stage=sfb_tmem_stages,
            sf_vec_size=SF_VEC,
            use_pdl=True,
            split_k=split_k,
            use_scale=False,
            use_bias=False,
        )
        e4m3, e8m0 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
        with torch.cuda.device(device):
            fn = cute.compile(
                gemm,
                _ptr(e4m3, 16, 16),
                _ptr(e8m0, 32, 32),
                _ptr(e4m3, 16, 16),
                _ptr(e8m0, 16, 16),
                _ptr(cutlass.BFloat16, 16, 16),
                None,
                None,
                tuple(cutlass.Int32(v) for v in (128, 8, cta_k, 1)),
                make_fake_stream(),
            )
        _COMPILED[key] = fn
    return fn


def probe(
    weight: torch.Tensor, weight_scale: torch.Tensor, *, tactic: Tactic, rows: int
) -> None:
    """Launch this tactic once on ``weight`` with ``rows`` zero activation rows into a
    NaN-filled output and wait for it (at build, outside any graph). Raises when the
    kernel does not compile or launch, or leaves an output element other than zero."""
    n, k = (int(d) for d in weight.shape)
    device = weight.device
    a = torch.zeros((rows, k), dtype=torch.uint8, device=device)
    a_scale = torch.full(  # UE8M0 127: a scale of 1
        (scale_bytes(MAX_ROWS, k),), 127, dtype=torch.uint8, device=device
    )
    out = torch.full((rows, n), float("nan"), dtype=torch.bfloat16, device=device)
    linear(
        a.view(torch.float8_e4m3fn),
        a_scale,
        weight,
        weight_scale,
        tactic=tactic,
        out=out,
    )
    torch.cuda.synchronize(device)
    wrong = int((out != 0).sum())  # NaN (unwritten) included
    if wrong:
        raise RuntimeError(
            f"tactic {tactic} left {wrong} of {out.numel()} outputs other than zero "
            f"for {rows} zero rows (N {n}, K {k})"
        )


def linear(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    tactic: Tactic,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``a @ weight.T`` in BF16 [M, N] for MXFP8 rows ``a`` [M, K] (1 <= M <= 8) with
    their 128x4 UE8M0 scales, and an MXFP8 weight [N, K] with its 128x4 scales as
    SGLang loads them (``weight_scale_inv_swizzled``), on the current CUDA stream."""
    from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Int32

    from sgl_nvidia_mega.common.sync import current_cu_stream

    rows, k = a.shape
    n = weight.shape[0]
    assert not activation_reasons(a, a_scale, k), activation_reasons(a, a_scale, k)
    assert weight.shape[1] == k and weight.is_contiguous()
    if out is None:
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=a.device)
    assert out.shape == (rows, n) and out.dtype == torch.bfloat16
    assert out.is_contiguous() and out.data_ptr() % 16 == 0
    compiled(tactic, a.device)(
        _ptr(Float8E4M3FN, weight.data_ptr(), 16),
        _ptr(Float8E8M0FNU, weight_scale.data_ptr(), 32),
        _ptr(Float8E4M3FN, a.data_ptr(), 16),
        _ptr(Float8E8M0FNU, a_scale.data_ptr(), 16),
        _ptr(BFloat16, out.data_ptr(), 16),
        None,
        None,
        (Int32(n), Int32(rows), Int32(k), Int32(1)),
        current_cu_stream(),
    )
    return out
