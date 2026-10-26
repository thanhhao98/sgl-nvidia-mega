# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax M3.1 megakernel slot attn_ll on one SM100-family GPU (GB300, B300, B200).

The attention's two MXFP8 GEMMs of one TP4 rank, the fused q|k|v|index projection
(N 2560, K 6144) and o_proj (N 6144, K 2048), on FlashInfer's low-latency block-scaled
GEMM launched directly with the slot's pinned tactics (``attention.LowLatencyLinear``,
``mxfp8_ll``), against SGLang's stock MXFP8 linear, ``Fp8LinearMethod.apply`` as M3.1
loads it (``flashinfer_mxfp8_blockscaled_linear``, cute-dsl), on synthetic weights
whose ``weight_scale_inv_swizzled`` SGLang's own load step derives
(``_process_mxfp8_linear_weight_scale``), at every served width (1-8 rows):

- against the stock output (``ATTN_LL_REL``): max |diff| / max |ref| and the norm
  ratio within ``runtime.TOLERANCES["attn_ll"]``, and every element within one BF16 ulp
  of the larger value plus an FP32-accumulation allowance (``ACC_REL`` of
  sum_k |a_k w_k|); both outputs within the same per-element bound of an FP64 product
  of the same MXFP8 operands;
- what SGLang's kernels write from the q|k|v|index output (``ATTN_LL_DOWNSTREAM``):
  the FP8 q / index q of its q/k norm + RoPE kernel, the NVFP4 K / V / index-K codes
  and E4M3 block scales of its KV store, and the index top-16 pages of each query row
  (the MSA-v1 indexer's page scores over a synthetic context): the fraction that moves
  and the largest move, within the declared tolerances;
- the slot reads the same MXFP8 input bytes as the stock linear: a pre-quantized
  ``Mxfp8SwizzledInput`` (slot hop_mxfp8) of the stock quantize's bytes gives the slot
  the same output bit for bit as the BF16 input, and the stock linear too;
- the kernel is deterministic, writes every output element (NaN-filled output), and
  ignores the activation scale atom's padding rows (0xFF-poisoned);
- a CUDA graph replay equals the eager call, bit for bit;
- every pinned tactic of this GPU is one FlashInfer offers, and the build's probe
  (``mxfp8_ll.probe``: compile, one launch, every output written) passes for it;
- timings (``ATTN_LL_TIME``): stock vs low-latency per layer, in a captured chain of
  60 calls cycling over 16 weight copies (weights stream from DRAM, as over the 60
  layers), with BF16 input (quantize + GEMM) and with MXFP8 input (GEMM only);
- FlashInfer's kernel reads only its weights before its PDL wait (a check of its
  source), which is what lets slot hop_mxfp8's boundary kernel signal it at entry;
  the tactics that keep every K tile in flight before the wait, (128, 12, 1, 4) and
  (256, 6, 1, 4), are offered and fit FlashInfer's own shared-memory estimate;
- the serving chain (``ATTN_LL_SERVING``, the q|k|v|index GEMM as it now runs): 60
  layers in a CUDA graph of a stand-in for the attention-input boundary kernel
  (``pdl_spin.cuh``: launched with PDL, runs ~6 us, signals its dependent at its entry
  or at its end, then writes the GEMM's MXFP8 input), the GEMM over 16 weight copies,
  and SGLang's q/k norm + RoPE kernel after it: every layer's output bit for bit as
  the same calls make it eagerly (the layers alternate between two inputs, so a GEMM
  that read its input before its wait would show), and the time per layer with the
  pinned tactic and the two above, both triggers, against the stock GEMM.

Sweep mode, ``SGLANG_M31_ATTN_LL_SWEEP=1`` (or ``=qkv_index`` / ``=o_proj``): times
every tactic FlashInfer offers per (N, K, rows in 1, 2, 4, 8) in the back-to-back
chain, checks each against the stock output, and prints the best per row bucket and
the PINNED_TACTICS lines to paste into sm100/mxfp8_ll.py (compiles every tactic:
minutes). ``=serving`` (also run by ``=1``): every q|k|v|index tactic in the serving
chain with the entry trigger, the table per row bucket and the PINNED_TACTICS lines for
it.

    python3 -m pytest -q -s test/kernels/test_minimax_m31_attn_ll.py
"""

import functools
import inspect
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

SHAPES = {"qkv_index": (2560, 6144), "o_proj": (6144, 2048)}
TOLERANCE_KEYS = {"qkv_index": "attn.qkv_index", "o_proj": "attn.o_proj"}
ROWS = tuple(range(1, 9))
TIMED_ROWS = (1, 2, 4, 8)
CHAIN = 60  # calls in a timed chain: the 60 layers of a step
COPIES = 16  # weight copies the chain cycles through (more bytes than L2 holds)
SWEEP_ENV = "SGLANG_M31_ATTN_LL_SWEEP"
# the serving chain: the attention-input boundary kernel the stand-in replaces runs
# ~6.2 us at 8 rows on GB300 (nsys, the served verify graph), in 54 CTAs (9 clusters
# of 6: 8 rows and the bumper); the layers alternate between two inputs
SERVING_SPIN_NS = 6000
SERVING_BLOCKS = 54
SERVING_INPUTS = 2
SERVING_ROWS = (1, 8)
# every K tile of a CTA in flight before the PDL wait, 224,896 B of shared memory
ALL_TILES_IN_FLIGHT = ((128, 12, 1, 4), (256, 6, 1, 4))
# Per element, two MXFP8 GEMMs of the same operands differ by the BF16 rounding of the
# larger output plus the difference of their accumulations, a small multiple of
# sum_k |a_k w_k| (FP32 summation order, the MMA's own block sums); 2^-13 of that sum
# is far below what one wrong K block of 32 moves (2^-10 of it or more at these K),
# let alone a wrong row or scale.
ACC_REL = 2.0**-13


def _skip_reason():
    if not torch.cuda.is_available():
        return "needs CUDA"
    if torch.cuda.get_device_capability()[0] != 10:
        return "needs an SM100-family GPU (B200, B300, GB300)"
    try:
        import cutlass  # noqa: F401
        import flashinfer  # noqa: F401
    except ImportError:
        return "needs FlashInfer and the CuTe DSL"
    from sglang.srt.layers.quantization import fp8_utils

    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    if not mxfp8_ll.supported(mxfp8_ll.sm_version()):
        return "FlashInfer has no low-latency MXFP8 GEMM on this GPU"
    if not hasattr(fp8_utils, "flashinfer_mxfp8_quantize"):
        return "SGLang's FlashInfer MXFP8 quantize is not available"
    return None


_SKIP = _skip_reason()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")


def _sm():
    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    return mxfp8_ll.sm_version()


def _round_up(x, m):
    return -(-x // m) * m


# ------------------------------------------------------------------ SGLang's linear


@pytest.fixture(scope="module")
def stock_method():
    """SGLang's Fp8LinearMethod as the M3.1 checkpoint's MXFP8 projections get it."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.minimax_m3_nvfp4 import MiniMaxM3Nvfp4Config

    qm = Fp8LinearMethod(MiniMaxM3Nvfp4Config())
    backend = qm.mxfp8_dense_backend
    if not (qm.use_mxfp8 and backend is not None and backend.is_flashinfer_cutedsl()):
        pytest.skip(f"SGLang resolves the MXFP8 linear to {backend}, not cute-dsl")
    return qm


def _linear(qm, n, k, gen):
    """An MXFP8 linear as SGLang loads one: the e4m3 weight [N, K], the checkpoint's
    UE8M0 scales [N, K / 32] (weight_scale_inv) and weight_scale_inv_swizzled derived
    by SGLang's load step (block_scale_interleave)."""
    from torch.nn import Parameter

    layer = torch.nn.Module()
    codes = (torch.randn(n, k, device="cuda", generator=gen) * 2).to(
        torch.float8_e4m3fn
    )
    # 2^-7 .. 2^-4: outputs of order one for these K
    scales = torch.randint(
        120, 124, (n, k // 32), device="cuda", dtype=torch.uint8, generator=gen
    )
    layer.weight = Parameter(codes, requires_grad=False)
    layer.weight_scale_inv = Parameter(scales, requires_grad=False)
    layer.register_parameter("bias", None)
    qm._process_mxfp8_linear_weight_scale(layer)
    return layer


@pytest.fixture(scope="module")
def linears(stock_method):
    gen = torch.Generator(device="cuda").manual_seed(7)
    return {name: _linear(stock_method, n, k, gen) for name, (n, k) in SHAPES.items()}


def _ll_linear(layer, rows, tactic=None):
    """The slot's GEMM (attention.LowLatencyLinear) with this row count's tactic."""
    from sgl_nvidia_mega.minimax_m31 import attention
    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    n, k = layer.weight.shape
    if tactic is None:
        tactic = mxfp8_ll.pinned_tactic(_sm(), n, k, rows)
    return attention.LowLatencyLinear(layer, {rows: tactic})


def _activations(name, rows, gen):
    """The q|k|v|index input as the attention boundary writes it (RMS-normed rows
    with heavy-tailed channels, times a Gemma norm weight), or o_proj's (an attention
    output); BF16."""
    n, k = SHAPES[name]
    if name == "o_proj":
        return (torch.randn(rows, k, device="cuda", generator=gen) * 0.5).bfloat16()
    gain = torch.exp(0.75 * torch.randn(k, device="cuda", generator=gen))
    gain[torch.randperm(k, device="cuda", generator=gen)[:8]] *= 30.0
    h = torch.randn(rows, k, device="cuda", generator=gen) * gain
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    norm = 1.0 + 0.3 * torch.randn(k, device="cuda", generator=gen)
    return (y * norm).bfloat16()


def _quantize(x):
    """SGLang's MXFP8 quantize as flashinfer_mxfp8_blockscaled_linear calls it."""
    from sglang.srt.layers.quantization import fp8_utils

    return fp8_utils.flashinfer_mxfp8_quantize(
        x.view(-1, x.shape[-1]), is_sf_swizzled_layout=True, alignment=32
    )


def _swizzled_offsets(rows, k, device):
    """Byte offsets of rows' UE8M0 scales [rows, K / 32] in one 128x4 atom."""
    r = torch.arange(rows, device=device)[:, None]
    kb = torch.arange(k // 32, device=device)[None, :]
    return (kb // 4) * 512 + (r % 32) * 16 + (r // 32) * 4 + kb % 4


def _dequant(codes, scales):
    """MXFP8 codes [R, K] and UE8M0 scales [R, K / 32] in FP64."""
    r, k = codes.shape
    v = codes.float().double().view(r, k // 32, 32)
    return (v * torch.exp2(scales.double() - 127).unsqueeze(-1)).view(r, k)


def _rel(out, ref):
    d = (out.double() - ref.double()).abs().max().item()
    return d / max(ref.double().abs().max().item(), 1e-30), d


def _norm_rel(out, ref):
    """|out - ref| / |ref| in the 2-norm."""
    d = (out.double() - ref.double()).norm().item()
    return d / max(ref.double().norm().item(), 1e-30)


def _bf16_ulp(x):
    """The BF16 spacing at |x| (float64): 2^(e - 7) for |x| in [2^e, 2^(e + 1)),
    2^-133 below the normal range."""
    e = torch.floor(torch.log2(x.double().abs().clamp_min(2.0**-126)))
    return torch.exp2(e - 7)


def _ulp_excess(out, ref, mag):
    """max |out - ref| / (the BF16 ulp of the larger of the two + ACC_REL * mag) over
    the elements, ``mag`` = sum_k |a_k w_k|: at most 1 when the two differ only by
    BF16 rounding and summation order (near zero an element's own ulp alone is too
    fine for any reordered sum)."""
    out, ref = out.double(), ref.double()
    allowed = _bf16_ulp(torch.maximum(out.abs(), ref.abs())) + ACC_REL * mag
    return ((out - ref).abs() / allowed).max().item()


def _tolerance(name, what="rel"):
    from sgl_nvidia_mega.minimax_m31 import runtime

    return runtime.TOLERANCES["attn_ll"][f"{TOLERANCE_KEYS[name]}.{what}"]


# --------------------------------------------------------------------------- tests


def test_pinned_tactics_are_offered():
    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    sm = _sm()
    pinned = {
        key: tactic for key, tactic in mxfp8_ll.PINNED_TACTICS.items() if key[0] == sm
    }
    assert pinned, f"no pinned tactics for sm_{sm}"
    for (_, n, k, bucket), tactic in pinned.items():
        assert tactic in mxfp8_ll.offered_tactics(n, k, bucket), (n, k, bucket)
    for n, k in SHAPES.values():
        for rows in ROWS:
            assert mxfp8_ll.pinned_tactic(sm, n, k, rows) is not None, (n, k, rows)


def test_the_builds_probe_passes_for_every_pinned_tactic(linears):
    """What the build runs per tactic before it serves a width (attention._GpuKernels
    .prepare: compile, one launch on zero rows into a NaN-filled output, every output
    zero), at 1 and 8 rows; and the probe catches an unwritten output."""
    from sgl_nvidia_mega.minimax_m31 import attention
    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    sm = _sm()
    kernels = attention._GpuKernels()
    for name, (n, k) in SHAPES.items():
        layer = linears[name]
        pinned = {
            t
            for (s, pn, pk, _), t in mxfp8_ll.PINNED_TACTICS.items()
            if s == sm and (pn, pk) == (n, k)
        }
        for tactic in sorted(pinned):
            for rows in (1, 8):
                kernels.prepare(tactic, layer, rows)
    # a launch that writes nothing leaves the NaN fill: the probe raises
    layer = linears["o_proj"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mxfp8_ll, "linear", lambda *args, **kwargs: kwargs["out"])
        with pytest.raises(RuntimeError, match="other than zero"):
            mxfp8_ll.probe(
                layer.weight,
                layer.weight_scale_inv_swizzled,
                tactic=mxfp8_ll.pinned_tactic(sm, *SHAPES["o_proj"], 8),
                rows=8,
            )


def test_sglangs_swizzled_weight_scales_are_the_kernels_layout(linears):
    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    for name, (n, k) in SHAPES.items():
        layer = linears[name]
        assert (
            mxfp8_ll.weight_reasons(
                layer.weight, layer.weight_scale_inv_swizzled, (n, k)
            )
            == []
        ), name


@pytest.mark.parametrize("name", list(SHAPES))
def test_low_latency_matches_the_stock_linear(stock_method, linears, name):
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

    n, k = SHAPES[name]
    layer = linears[name]
    tol, tol_norm = _tolerance(name), _tolerance(name, "norm_rel")
    gen = torch.Generator(device="cuda").manual_seed(11)
    worst = dict(rel=0.0, norm_rel=0.0, ulp_excess=0.0)
    w64 = _dequant(layer.weight, layer.weight_scale_inv)
    w64_abs = w64.abs()
    for rows in ROWS:
        x = _activations(name, rows, gen)
        gemm = _ll_linear(layer, rows)
        stock = stock_method.apply(layer, x, None)
        out = gemm(x)
        assert out.shape == (rows, n) and out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
        rel, max_abs = _rel(out, stock)
        norm_rel = _norm_rel(out, stock)
        # the same MXFP8 operands in FP64, and the scale of each output's sum
        q, sf = _quantize(x)
        a_sf = sf.view(torch.uint8).reshape(-1)[_swizzled_offsets(rows, k, sf.device)]
        a64 = _dequant(q, a_sf)
        ref64 = a64 @ w64.t()
        mag = a64.abs() @ w64_abs.t()
        excess = _ulp_excess(out, stock, mag)
        excess_ll64 = _ulp_excess(out, ref64, mag)
        excess_stock64 = _ulp_excess(stock, ref64, mag)
        moved = (out != stock).double().mean().item()
        print(
            f"ATTN_LL_REL {name} rows={rows} max|diff|/max|ref|={rel:.3e} "
            f"max|diff|={max_abs:.3e} norm_rel={norm_rel:.3e} moved={moved:.3e} "
            f"ulp_excess={excess:.3f} (vs fp64: low-latency {_rel(out, ref64)[0]:.3e} "
            f"excess {excess_ll64:.3f}, stock {_rel(stock, ref64)[0]:.3e} excess "
            f"{excess_stock64:.3f}; tolerances {tol:g}, norm {tol_norm:g})"
        )
        assert rel <= tol, (name, rows, rel)
        assert norm_rel <= tol_norm, (name, rows, norm_rel)
        # every element: the BF16 rounding and the summation order only
        assert excess <= 1.0, (name, rows, excess)
        assert excess_ll64 <= 1.0, (name, rows, "low-latency vs fp64", excess_ll64)
        # the reference itself (SGLang's GEMM against the FP64 product)
        assert excess_stock64 <= 1.0, (name, rows, "stock vs fp64", excess_stock64)
        for key, value in (
            ("rel", rel),
            ("norm_rel", norm_rel),
            ("ulp_excess", excess),
        ):
            worst[key] = max(worst[key], value)
        # the same MXFP8 input bytes: the quantize is deterministic, and pre-quantized
        # both linears give their BF16 outputs bit for bit
        q2, sf2 = _quantize(x)
        assert torch.equal(q2.view(torch.uint8), q.view(torch.uint8))
        assert torch.equal(sf2.view(torch.uint8), sf.view(torch.uint8))
        pre = Mxfp8SwizzledInput(q, sf)
        assert torch.equal(gemm(pre), out), (name, rows)
        assert torch.equal(stock_method.apply(layer, pre, None), stock), (name, rows)
        # deterministic
        assert torch.equal(gemm(x), out), (name, rows)
    print(
        f"ATTN_LL_REL_WORST {name} {json.dumps(worst)} (tolerances {tol:g}, norm "
        f"{tol_norm:g})"
    )


@pytest.mark.parametrize("name", list(SHAPES))
def test_every_output_written_and_padding_scales_ignored(linears, name):
    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    n, k = SHAPES[name]
    layer = linears[name]
    gen = torch.Generator(device="cuda").manual_seed(31)
    for rows in ROWS:
        tactic = mxfp8_ll.pinned_tactic(_sm(), n, k, rows)
        q, sf = _quantize(_activations(name, rows, gen))
        args = (layer.weight, layer.weight_scale_inv_swizzled)
        clean = mxfp8_ll.linear(q, sf, *args, tactic=tactic)
        out = torch.full((rows, n), float("nan"), dtype=torch.bfloat16, device="cuda")
        mxfp8_ll.linear(q, sf, *args, tactic=tactic, out=out)
        assert torch.equal(out, clean), (name, rows)
        poisoned = torch.full_like(sf.view(torch.uint8), 0xFF)
        real = _swizzled_offsets(rows, k, sf.device).flatten()
        poisoned[real] = sf.view(torch.uint8)[real]
        assert torch.equal(mxfp8_ll.linear(q, poisoned, *args, tactic=tactic), clean)


@pytest.mark.parametrize("name", list(SHAPES))
def test_cuda_graph_replay_matches_eager(linears, name):
    layer = linears[name]
    gen = torch.Generator(device="cuda").manual_seed(13)
    for rows in (1, 3, 8):
        gemm = _ll_linear(layer, rows)
        static_x = _activations(name, rows, gen)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gemm(static_x)  # warm-up off the capture
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = gemm(static_x)
        for _ in range(2):
            x = _activations(name, rows, gen)
            static_x.copy_(x)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(static_out, gemm(x)), (name, rows)


# ------------------------------------------ what SGLang writes from q|k|v|index

DOWNSTREAM_SEEDS = 8
EPS = 1e-6  # text_config.rms_norm_eps: the q / k / index norms
ROPE_DIM = 64  # NeoX RoPE on the first 64 of each head's 128 dims
MAX_POS = 1 << 16
PAGE = 128
CONTEXT_PAGES = 64  # the synthetic index-K context the top-k pages come from
TOPK_PAGES = 16  # the pages the MSA-v1 indexer keeps for a query row, the local one too
# _qknorm_groups' (head offset, heads) of one TP4 rank's q|k|v|index row
# [q 16 heads | k | v | index q | index k]
NORM_GROUPS = ((0, 16), (16, 1), (18, 1), (19, 1))
LAYER = SimpleNamespace(layer_id=0)


def _rope_cache():
    """M3.1's RoPE cache as SGLang builds it (theta 1e7, rotary dim 64): float32
    [positions, cos 32 | sin 32]."""
    from sgl_nvidia_mega.minimax_m31 import ROPE_THETA

    dims = torch.arange(0, ROPE_DIM, 2, dtype=torch.float, device="cuda") / ROPE_DIM
    inv_freq = 1.0 / ROPE_THETA**dims
    freqs = torch.outer(
        torch.arange(MAX_POS, dtype=torch.float, device="cuda"), inv_freq
    )
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def _norm_groups(gen):
    """The q / k / index-q / index-k Gemma norm weights (raw: the kernel applies
    1 + w), as MiniMaxM3Attention._qknorm_groups lists them."""
    return [
        ((0.1 * torch.randn(128, device="cuda", generator=gen)).bfloat16(), off, cnt)
        for off, cnt in NORM_GROUPS
    ]


def _kv_pool():
    """SGLang's M3.1 KV pool (NVFP4 main K/V and index K) for one layer, 4 pages."""
    from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool

    return MiniMaxM3Nvfp4KVPool(
        size=3 * PAGE,
        page_size=PAGE,
        dtype=torch.float8_e4m3fn,
        head_num=1,
        head_dim=128,
        idx_head_dim=128,
        dense_layer_ids=[],
        sparse_layer_ids=[0],
        disable_value_sparse_layer_ids=[0],
        device="cuda",
        start_layer=0,
        end_layer=1,
    )


def _downstream(fused_out, groups, cache, positions, pool, loc):
    """What SGLang's kernels write from one q|k|v|index output, called as
    MiniMaxM3Attention.forward_prepare and the NVFP4 backend's KV store call them (the
    norm + RoPE runs in place, on a copy): the FP8 q and index q, and the NVFP4 codes
    [rows, 64] and E4M3 block scales [rows, 8] of K, V and index K at ``loc``."""
    from sglang.kernels.ops.attention.minimax_qknorm_rope import (
        minimax_qknorm_rope_grouped_fp8,
    )
    from sglang.srt.models.minimax_m3 import (
        _QKNORM_GROUP_INDEX_Q,
        _QKNORM_GROUP_MAIN_Q,
    )

    fused = fused_out.clone()
    rows = fused.shape[0]
    fp8_q = fused.new_empty((rows, 16, 128), dtype=torch.float8_e4m3fn)
    fp8_idx_q = fused.new_empty((rows, 1, 128), dtype=torch.float8_e4m3fn)
    minimax_qknorm_rope_grouped_fp8(
        fused,
        groups,
        cache,
        positions,
        EPS,
        fp8_a=fp8_q,
        group_a=_QKNORM_GROUP_MAIN_Q,
        fp8_b=fp8_idx_q,
        group_b=_QKNORM_GROUP_INDEX_Q,
    )
    _, k, v = fused[:, :2304].split([2048, 128, 128], dim=-1)
    _, idx_k = fused[:, 2304:].split([128, 128], dim=-1)
    pool.set_fused_kv_index_buffer(
        LAYER,
        loc,
        k.view(rows, 1, 128),
        v.view(rows, 1, 128),
        idx_k.reshape(rows, 1, 128),
        None,
    )
    k_codes, v_codes, k_scales, v_scales = pool.main_pool.get_quantized_rows(0, loc)
    idx_codes, idx_scales = pool.index_k_pool.get_quantized_rows(0, loc)

    def rows_of(codes, scales):
        return (
            codes.reshape(rows, 64).clone(),
            scales.view(torch.uint8).reshape(rows, 8).clone(),
        )

    return {
        "q": fp8_q,
        "idx_q": fp8_idx_q,
        "k": rows_of(k_codes, k_scales),
        "v": rows_of(v_codes, v_scales),
        "idx_k": rows_of(idx_codes, idx_scales),
    }


def _same_bytes(a, b):
    if isinstance(a, tuple):
        return all(_same_bytes(x, y) for x, y in zip(a, b))
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _e4m3_steps(a, b):
    """The distance of two e4m3 codes in representable steps."""

    def ordered(x):
        bits = x.view(torch.uint8).long()
        mag = bits & 0x7F
        return torch.where(bits & 0x80 != 0, -mag, mag)

    return (ordered(a) - ordered(b)).abs()


def _nibbles(codes):
    """NVFP4 codes [R, 64] (two a byte, low nibble first) as [R, 128]."""
    c = codes.view(torch.uint8).long()
    return torch.stack((c & 0xF, c >> 4), dim=-1).flatten(-2)


def _nvfp4_values(codes, scales):
    """NVFP4 rows and their E4M3 block scales (16 values a block, unit global scale) as
    float32 [R, 128], and each value's block scale."""
    from sglang.srt.layers.quantization.kvfp4_tensor import E2M1_VALUES

    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=codes.device)
    unit = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return lut[_nibbles(codes)] * unit, unit


def _index_context(gen):
    """A synthetic index-K context of CONTEXT_PAGES pages, stored as SGLang's NVFP4
    index cache stores it and read as the indexer scores it (dequantized, then FP8):
    float32 [pages * 128, 128]."""
    from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

    keys = torch.randn(CONTEXT_PAGES * PAGE, 1, 128, device="cuda", generator=gen)
    unit = torch.ones(1, dtype=torch.float32, device="cuda")
    packed, scales, _ = NVFP4KVQuantizeUtil.quantize(keys.bfloat16(), unit)
    keys = NVFP4KVQuantizeUtil.dequantize(packed, scales, unit)
    return keys.to(torch.float8_e4m3fn).float().reshape(-1, 128)


def _topk_pages(fp8_idx_q, context):
    """Each query row's index pages, scored as SGLang's MSA-v1 plan test scores the
    decode indexer: a page's score is the largest FP8 index q . key over its tokens;
    the TOPK_PAGES - 1 best pages besides the local one, which is always kept (sorted
    page ids)."""
    q = fp8_idx_q.float().reshape(fp8_idx_q.shape[0], 128)
    scores = (q @ context.t()).unflatten(-1, (-1, PAGE)).amax(-1)
    return scores.topk(TOPK_PAGES - 1, dim=-1).indices.sort(dim=-1).values


def test_what_sglang_writes_from_the_qkv_index_output(stock_method, linears):
    """The low-latency q|k|v|index GEMM against SGLang's on the same MXFP8 input,
    through what SGLang's kernels write from its output: the FP8 q and index q (q/k
    norm + RoPE), the NVFP4 K / V / index-K codes and E4M3 block scales (the KV store),
    and the index top-16 pages of each query row. A BF16 difference moves a quantized
    code only when it crosses a rounding boundary: the fraction that moves and the
    largest move, against runtime.TOLERANCES["attn_ll"]; SGLang's kernels are
    deterministic, so every move is the GEMM's."""
    from sgl_nvidia_mega.minimax_m31 import runtime

    tol = runtime.TOLERANCES["attn_ll"]
    layer = linears["qkv_index"]
    gen = torch.Generator(device="cuda").manual_seed(41)
    cache, groups, context = _rope_cache(), _norm_groups(gen), _index_context(gen)
    pools = {arm: _kv_pool() for arm in ("stock", "ll", "again")}
    st = Counter()
    worst = dict(fp8_steps=0, block_top_steps=0.0)
    for _ in range(DOWNSTREAM_SEEDS):
        for rows in ROWS:
            x = _activations("qkv_index", rows, gen)
            stock = stock_method.apply(layer, x, None)
            ll = _ll_linear(layer, rows)(x)
            positions = torch.randint(0, MAX_POS, (rows,), device="cuda", generator=gen)
            loc = torch.randperm(2 * PAGE, device="cuda", generator=gen)[:rows] + PAGE
            args = (groups, cache, positions)
            ref = _downstream(stock, *args, pools["stock"], loc)
            got = _downstream(ll, *args, pools["ll"], loc)
            again = _downstream(stock, *args, pools["again"], loc)
            assert all(_same_bytes(ref[key], again[key]) for key in ref), rows
            st["gemm"] += stock.numel()
            st["gemm_moved"] += int((stock != ll).sum())
            for name in ("q", "idx_q"):
                steps = _e4m3_steps(ref[name], got[name])
                st[f"{name}_codes"] += steps.numel()
                st[f"{name}_moved"] += int((steps > 0).sum())
                worst["fp8_steps"] = max(worst["fp8_steps"], int(steps.max()))
            for name in ("k", "v", "idx_k"):
                (codes_ref, scales_ref), (codes_ll, scales_ll) = ref[name], got[name]
                moved = _nibbles(codes_ref) != _nibbles(codes_ll)
                st[f"{name}_codes"] += moved.numel()
                st[f"{name}_moved"] += int(moved.sum())
                st["scales"] += scales_ref.numel()
                st["scales_moved"] += int((scales_ref != scales_ll).sum())
                values_ref, unit_ref = _nvfp4_values(codes_ref, scales_ref)
                values_ll, unit_ll = _nvfp4_values(codes_ll, scales_ll)
                # the block's coarsest E2M1 step (4 -> 6) is twice its scale
                top_step = 2.0 * torch.maximum(unit_ref, unit_ll).clamp_min(1e-30)
                moved_steps = ((values_ref - values_ll).abs() / top_step).max().item()
                worst["block_top_steps"] = max(worst["block_top_steps"], moved_steps)
            pages_ref = _topk_pages(ref["idx_q"], context)
            pages_ll = _topk_pages(got["idx_q"], context)
            st["topk_rows"] += rows
            st["topk_moved"] += int((pages_ref != pages_ll).any(dim=-1).sum())
    fraction = {
        name: st[f"{name}_moved"] / st[f"{name}_codes"]
        for name in ("q", "idx_q", "k", "v", "idx_k")
    }
    measured = {
        "attn.fp8_q.moved_fraction": (st["q_moved"] + st["idx_q_moved"])
        / (st["q_codes"] + st["idx_q_codes"]),
        "attn.fp8_q.max_e4m3_steps": worst["fp8_steps"],
        "attn.kv.nvfp4.moved_fraction": max(fraction[n] for n in ("k", "v", "idx_k")),
        "attn.kv.scale.moved_fraction": st["scales_moved"] / st["scales"],
        "attn.kv.nvfp4.block_top_steps": worst["block_top_steps"],
        "attn.index.topk_moved_fraction": st["topk_moved"] / st["topk_rows"],
    }
    print(
        "ATTN_LL_DOWNSTREAM "
        + json.dumps(
            dict(
                seeds=DOWNSTREAM_SEEDS,
                rows=list(ROWS),
                gemm_bf16_moved_fraction=st["gemm_moved"] / st["gemm"],
                moved_fraction=fraction,
                measured=measured,
                declared={key: tol[key] for key in measured},
            )
        )
    )
    for key, value in measured.items():
        assert value <= tol[key], (key, value, tol[key])


# -------------------------------------------------------------------------- timing


def _copies(stock_method, name):
    gen = torch.Generator(device="cuda").manual_seed(17)
    n, k = SHAPES[name]
    return [_linear(stock_method, n, k, gen) for _ in range(COPIES)]


def _chain_us(call, reps=20) -> float:
    """Per-call time of CHAIN calls call(i), captured in one CUDA graph (PDL edges
    kept), replayed reps times."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(CHAIN):
            call(i)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(CHAIN):
            call(i)
    for _ in range(3):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (reps * CHAIN)


def _tune_stock(stock_method, layer, x):
    """FlashInfer's autotune of the stock GEMM at this row count, as SGLang tunes it
    at startup; False when it cannot run here."""
    try:
        from flashinfer.autotuner import autotune

        with autotune(True):
            stock_method.apply(layer, x, None)
        torch.cuda.synchronize()
        return True
    except Exception as e:  # report, and time the untuned stock tactic
        print(f"ATTN_LL_TIME note: stock autotune unavailable ({e!r})")
        return False


@pytest.mark.parametrize("name", list(SHAPES))
def test_timing(stock_method, name):
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

    layers = _copies(stock_method, name)
    gen = torch.Generator(device="cuda").manual_seed(19)
    for rows in TIMED_ROWS:
        x = _activations(name, rows, gen)
        tuned = _tune_stock(stock_method, layers[0], x)
        gemms = [_ll_linear(layer, rows) for layer in layers]
        pre = Mxfp8SwizzledInput(*_quantize(x))
        times = {
            "bf16": (
                _chain_us(lambda i: stock_method.apply(layers[i % COPIES], x, None)),
                _chain_us(lambda i: gemms[i % COPIES](x)),
            ),
            "mxfp8": (
                _chain_us(lambda i: stock_method.apply(layers[i % COPIES], pre, None)),
                _chain_us(lambda i: gemms[i % COPIES](pre)),
            ),
        }
        for kind, (stock_us, ll_us) in times.items():
            what = "quantize + GEMM" if kind == "bf16" else "GEMM"
            print(
                f"ATTN_LL_TIME {name} rows={rows} {kind} input ({what}): stock "
                f"{stock_us:.2f} us{'' if tuned else ' (untuned)'}, low-latency "
                f"{ll_us:.2f} us, gain {stock_us - ll_us:+.2f} us per layer "
                f"(tactic {gemms[0].tactics[rows]})"
            )


# ---------------------------------------------------------- the serving chain


def _ll_gemm_module():
    import importlib

    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    return importlib.import_module(mxfp8_ll._GEMM_MODULE)


def _method_source(src, name):
    """A method's source in a module's text: from its ``def`` to the next member of
    its class or the next top-level definition."""
    start = src.index(f"    def {name}(")
    ends = [
        i
        for i in (
            src.find(marker, start + 1)
            for marker in ("\n    @", "\n    def ", "\n@", "\ndef ", "\nclass ")
        )
        if i != -1
    ]
    return src[start : min(ends) if ends else len(src)]


def test_flashinfers_gemm_reads_only_weights_before_its_pdl_wait():
    """What lets the boundary kernel signal this GEMM at its entry (FlashInfer's
    LowLatencyBlockscaledGemmKernel): the weight warp issues its loads with no PDL
    wait and triggers the next kernel after them; the activation warp waits before
    its first load; no other warp waits or loads from global memory before the wait,
    and the epilogue's stores follow the MMA's completion. The slot compiles it
    without its optional output scale and bias, the only global reads elsewhere."""
    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    src = Path(inspect.getsourcefile(_ll_gemm_module())).read_text()
    dma_a = _method_source(src, "dma_a_warp")
    dma_b = _method_source(src, "dma_b_warp")
    epilog = _method_source(src, "epilog_warp")
    assert "griddepcontrol_wait" not in dma_a
    assert "griddepcontrol_launch_dependents" in dma_a
    assert "griddepcontrol_wait()" in dma_b
    assert dma_b.index("griddepcontrol_wait()") < dma_b.index("cute.copy(")
    for name in ("mma_warp", "epilog_warp", "sfb_warp", "kernel"):
        assert "griddepcontrol" not in _method_source(src, name), name
    stores = [i for i in range(len(epilog)) if epilog.startswith("tDgC)", i)]
    assert stores, "the epilogue's global stores moved"
    assert all(
        epilog.index("mma_epilog_full_bar + mma_epilog_state.index") < i for i in stores
    )
    compiled = inspect.getsource(mxfp8_ll.compiled)
    assert "use_scale=False" in compiled and "use_bias=False" in compiled
    assert "use_pdl=True" in compiled


def test_all_k_tiles_in_flight_tactics_are_offered_and_fit():
    import cutlass

    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    ll_gemm = _ll_gemm_module()
    n, k = SHAPES["qkv_index"]
    e4m3 = cutlass.Float8E4M3FN
    for tactic in ALL_TILES_IN_FLIGHT:
        cta_k, stages, sfb_stages, split_k = tactic
        smem = ll_gemm._smem_bytes(e4m3, e4m3, 32, cta_k, stages, sfb_stages, split_k)
        print(
            f"ATTN_LL_SERVING tactic {tactic}: {smem} B of shared memory "
            f"(FlashInfer's estimate; capacity {ll_gemm._SMEM_CAPACITY_BYTES} B), "
            f"{mxfp8_ll.tiles_before_wait(tactic, k)} of "
            f"{mxfp8_ll.k_tiles_per_cta(tactic, k)} K tiles before the wait"
        )
        assert smem <= ll_gemm._SMEM_CAPACITY_BYTES, tactic
        assert mxfp8_ll.tiles_before_wait(tactic, k) == mxfp8_ll.k_tiles_per_cta(
            tactic, k
        )
        for rows in TIMED_ROWS:
            assert tactic in mxfp8_ll.offered_tactics(n, k, rows), (tactic, rows)


@functools.lru_cache(maxsize=None)
def _spin_module():
    """pdl_spin.cuh, the stand-in for the attention-input boundary kernel."""
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "m31_attn_ll_test_pdl_spin",
        cuda_files=[str(Path(__file__).with_name("pdl_spin.cuh"))],
        cuda_wrappers=[("pdl_spin_copy", "pdl_spin_copy")],
    )


class _Pairs:
    """An MXFP8 pair (e4m3 rows, one 128-row atom of UE8M0 scales) in one uint8
    buffer, as the stand-in copies it: the rows at 0, the scales at a 128-byte
    boundary after them."""

    def __init__(self, rows, k):
        from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

        self.rows, self.k = rows, k
        self.offset = _round_up(rows * k, 128)
        self.scale_bytes = mxfp8_ll.scale_bytes(mxfp8_ll.MAX_ROWS, k)
        nbytes = _round_up(self.offset + self.scale_bytes, 16)
        self.buffer = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")

    def pair(self):
        from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

        data = self.buffer[: self.rows * self.k].view(torch.float8_e4m3fn)
        scales = self.buffer[self.offset : self.offset + self.scale_bytes]
        return Mxfp8SwizzledInput(data.view(self.rows, self.k), scales)

    def fill(self, x):
        """The MXFP8 of the bf16 rows ``x``, by SGLang's quantize."""
        q, sf = _quantize(x)
        pair = self.pair()
        pair.data.view(torch.uint8).copy_(q.view(torch.uint8))
        pair.scales.copy_(sf.view(torch.uint8).reshape(-1)[: self.scale_bytes])
        return self


class _QkNorm:
    """SGLang's q/k norm + RoPE kernel on the q|k|v|index output (in place, FP8 q and
    index q out), as the attention launches it after the GEMM."""

    def __init__(self, rows, gen):
        self.groups = _norm_groups(gen)
        self.cache = _rope_cache()
        self.positions = torch.randint(
            0, MAX_POS, (rows,), device="cuda", generator=gen
        )
        self.fp8_q = torch.empty(
            rows, 16, 128, dtype=torch.float8_e4m3fn, device="cuda"
        )
        self.fp8_idx_q = torch.empty(
            rows, 1, 128, dtype=torch.float8_e4m3fn, device="cuda"
        )

    def __call__(self, fused):
        from sglang.kernels.ops.attention.minimax_qknorm_rope import (
            minimax_qknorm_rope_grouped_fp8,
        )
        from sglang.srt.models.minimax_m3 import (
            _QKNORM_GROUP_INDEX_Q,
            _QKNORM_GROUP_MAIN_Q,
        )

        return minimax_qknorm_rope_grouped_fp8(
            fused,
            self.groups,
            self.cache,
            self.positions,
            EPS,
            fp8_a=self.fp8_q,
            group_a=_QKNORM_GROUP_MAIN_Q,
            fp8_b=self.fp8_idx_q,
            group_b=_QKNORM_GROUP_INDEX_Q,
        )


def _serving_chain_us(gemm_of, rows, *, early, gen, reps=20, check=True) -> float:
    """Per-layer time of CHAIN layers of the served order around the q|k|v|index GEMM,
    captured in one CUDA graph: the stand-in boundary kernel (pdl_spin.cuh, launched
    with PDL, SERVING_SPIN_NS, signalling its dependent at its entry when ``early``,
    else at its end) copies layer i's MXFP8 input into the pair, ``gemm_of(i)`` (layer
    i's weights) reads it, the q/k norm + RoPE kernel reads the GEMM's output. With
    ``check``, after a replay from poisoned inputs, every layer's output must equal the
    same GEMM and norm called eagerly on its input, bit for bit: the layers alternate
    between two inputs, so a GEMM that read the pair before its PDL wait would read
    the other one (or the poison)."""
    k = SHAPES["qkv_index"][1]
    spin = _spin_module()
    inputs = [
        _Pairs(rows, k).fill(_activations("qkv_index", rows, gen))
        for _ in range(SERVING_INPUTS)
    ]
    target = _Pairs(rows, k)
    pair = target.pair()
    norm = _QkNorm(rows, gen)

    def layer(i):
        source = inputs[i % SERVING_INPUTS].buffer
        spin.pdl_spin_copy(
            source, target.buffer, SERVING_SPIN_NS, early, SERVING_BLOCKS
        )
        return norm(gemm_of(i)(pair))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(CHAIN):
            layer(i)  # warm up (the compiles), off the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [layer(i) for i in range(CHAIN)]
    if check:
        target.buffer.fill_(0x7F)  # e4m3 NaN codes, scale 2^0
        for out in outs:
            out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for i, out in enumerate(outs):
            want = norm(gemm_of(i)(inputs[i % SERVING_INPUTS].pair()))
            assert torch.equal(out, want), (rows, early, i)
    for _ in range(3):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (reps * CHAIN)


def _serving_tactics(rows):
    """The pinned q|k|v|index tactic at ``rows``, then those that keep every K tile in
    flight before the wait."""
    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    n, k = SHAPES["qkv_index"]
    pinned = mxfp8_ll.pinned_tactic(_sm(), n, k, rows)
    return [pinned] + [t for t in ALL_TILES_IN_FLIGHT if t != pinned]


def _in_flight(tactic):
    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    k = SHAPES["qkv_index"][1]
    return (
        f"{mxfp8_ll.tiles_before_wait(tactic, k)}/{mxfp8_ll.k_tiles_per_cta(tactic, k)}"
    )


def test_serving_chain(stock_method):
    """The q|k|v|index GEMM where it now runs: after a ~6 us kernel that signals it at
    its entry (or at its end), over 16 weight copies, with the q/k norm after it.
    Checks every layer's output bit for bit (both triggers, each tactic); prints the
    time per layer against the stock GEMM in the same chain."""
    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    layers = _copies(stock_method, "qkv_index")
    gen = torch.Generator(device="cuda").manual_seed(29)
    spin_us = SERVING_SPIN_NS / 1e3
    for rows in SERVING_ROWS:
        tuned = _tune_stock(
            stock_method, layers[0], _activations("qkv_index", rows, gen)
        )
        # as served without the slot: the boundary signals at its end
        stock = _serving_chain_us(
            lambda i: (
                lambda pair, layer=layers[i % COPIES]: stock_method.apply(
                    layer, pair, None
                )
            ),
            rows,
            early=False,
            gen=gen,
        )
        print(
            f"ATTN_LL_SERVING rows={rows} stock{'' if tuned else ' (untuned)'}: "
            f"{stock:.2f} us per layer (stand-in {spin_us:.2f} us, end trigger)"
        )
        for tactic in _serving_tactics(rows):
            gemms = [_ll_linear(layer, rows, tactic) for layer in layers]
            us = {
                early: _serving_chain_us(
                    lambda i: gemms[i % COPIES], rows, early=early, gen=gen
                )
                for early in (False, True)
            }
            pinned = tactic == mxfp8_ll.pinned_tactic(_sm(), *SHAPES["qkv_index"], rows)
            print(
                f"ATTN_LL_SERVING rows={rows} {'pinned ' if pinned else ''}{tactic} "
                f"({_in_flight(tactic)} K tiles before the wait): end trigger "
                f"{us[False]:.2f} us, entry trigger {us[True]:.2f} us per layer "
                f"({us[True] - us[False]:+.2f} us; vs stock {us[True] - stock:+.2f} us)"
            )


# --------------------------------------------------------------------------- sweep


def _sweep_shapes():
    """The shapes the back-to-back chain sweeps."""
    value = os.environ.get(SWEEP_ENV, "")
    if value in ("", "0"):
        return []
    if value == "1":
        return list(SHAPES)
    return [s for s in value.split(",") if s in SHAPES]


def _sweep_serving():
    """Whether the serving chain is swept (q|k|v|index)."""
    value = os.environ.get(SWEEP_ENV, "")
    return value == "1" or "serving" in value.split(",")


@pytest.mark.skipif(not _sweep_shapes(), reason=f"sweep mode: {SWEEP_ENV}=1")
def test_sweep_tactics(stock_method):
    """Every tactic FlashInfer offers, per shape and row bucket, timed in the chain;
    prints the best and the PINNED_TACTICS lines for this GPU."""
    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    sm = _sm()
    gen = torch.Generator(device="cuda").manual_seed(23)
    lines = []
    for name in _sweep_shapes():
        n, k = SHAPES[name]
        layers = _copies(stock_method, name)
        tol = _tolerance(name)
        for rows in TIMED_ROWS:
            x = _activations(name, rows, gen)
            tuned = _tune_stock(stock_method, layers[0], x)
            ref = stock_method.apply(layers[0], x, None)
            stock_us = _chain_us(
                lambda i: stock_method.apply(layers[i % COPIES], x, None)
            )
            results, wrong = [], []
            offered = mxfp8_ll.offered_tactics(n, k, rows)
            for j, tactic in enumerate(offered):
                gemms = [_ll_linear(layer, rows, tactic) for layer in layers]
                try:
                    rel, _ = _rel(gemms[0](x), ref)
                except Exception as e:  # a tactic that fails to compile or run
                    wrong.append((tactic, repr(e)[:80]))
                    continue
                if not rel <= tol:
                    wrong.append((tactic, f"rel {rel:.2e}"))
                    continue
                us = _chain_us(lambda i: gemms[i % COPIES](x), reps=10)
                results.append((us, tactic))
                print(
                    f"SWEEP {name} rows={rows} [{j + 1}/{len(offered)}] {tactic} "
                    f"{us:.2f} us",
                    file=sys.stderr,
                )
            results.sort()
            pinned = mxfp8_ll.pinned_tactic(sm, n, k, rows)
            pinned_us = next((us for us, t in results if t == pinned), float("nan"))
            print(
                f"SWEEP_BEST {name} rows={rows}: stock {stock_us:.2f} us"
                f"{'' if tuned else ' (untuned)'}; pinned {pinned} {pinned_us:.2f} us; "
                f"best {[(t, round(us, 2)) for us, t in results[:5]]}; "
                f"{len(wrong)} tactics wrong or failed {wrong[:4]}"
            )
            if results:
                us, best = results[0]
                lines.append(
                    f"    ({sm}, {n}, {k}, {rows}): {best},  # {name} {us:.2f} us "
                    f"(stock {stock_us:.2f} us)"
                )
    print("SWEEP_PINNED_TACTICS (mxfp8_ll.PINNED_TACTICS entries):")
    for line in lines:
        print(line)


SERVING_SWEEP_TOP = 8  # the best tactics the table also times with the end trigger


@pytest.mark.skipif(not _sweep_serving(), reason=f"sweep mode: {SWEEP_ENV}=serving")
def test_sweep_qkv_tactics_in_the_serving_chain(stock_method):
    """Every q|k|v|index tactic FlashInfer offers, per row bucket, timed in the serving
    chain with the entry trigger (each layer's output checked bit for bit, and
    against the stock output within the tolerance); prints a table per row bucket and
    the PINNED_TACTICS lines for this GPU. A tactic whose chain output differs from
    its eager output (it read its input before its PDL wait) fails the sweep."""
    import cutlass

    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    sm = _sm()
    n, k = SHAPES["qkv_index"]
    ll_gemm = _ll_gemm_module()
    e4m3 = cutlass.Float8E4M3FN
    layers = _copies(stock_method, "qkv_index")
    tol = _tolerance("qkv_index")
    gen = torch.Generator(device="cuda").manual_seed(37)
    lines, mismatched = [], []
    for rows in TIMED_ROWS:
        x = _activations("qkv_index", rows, gen)
        tuned = _tune_stock(stock_method, layers[0], x)
        ref = stock_method.apply(layers[0], x, None)
        stock_us = _serving_chain_us(
            lambda i: (
                lambda pair, layer=layers[i % COPIES]: stock_method.apply(
                    layer, pair, None
                )
            ),
            rows,
            early=False,
            gen=gen,
            reps=10,
        )
        offered = mxfp8_ll.offered_tactics(n, k, rows)
        for tactic in ALL_TILES_IN_FLIGHT:
            assert tactic in offered, (tactic, rows)
        results, wrong = [], []
        for j, tactic in enumerate(offered):
            gemms = [_ll_linear(layer, rows, tactic) for layer in layers]
            try:
                rel, _ = _rel(gemms[0](x), ref)
                if not rel <= tol:
                    wrong.append((tactic, f"rel {rel:.2e}"))
                    continue
                us = _serving_chain_us(
                    lambda i: gemms[i % COPIES], rows, early=True, gen=gen, reps=10
                )
            except AssertionError as e:  # the chain's output is not the eager one
                mismatched.append((rows, tactic, repr(e)[:80]))
                continue
            except Exception as e:  # a tactic that fails to compile or run
                wrong.append((tactic, repr(e)[:80]))
                continue
            results.append((us, tactic, gemms))
            print(
                f"SERVING_SWEEP rows={rows} [{j + 1}/{len(offered)}] {tactic} "
                f"{us:.2f} us",
                file=sys.stderr,
            )
        results.sort(key=lambda r: r[0])
        pinned = mxfp8_ll.pinned_tactic(sm, n, k, rows)
        shown = results[:SERVING_SWEEP_TOP] + [
            r
            for r in results[SERVING_SWEEP_TOP:]
            if r[1] in (pinned, *ALL_TILES_IN_FLIGHT)
        ]
        print(
            f"SERVING_SWEEP_TABLE q|k|v|index rows={rows} (stand-in "
            f"{SERVING_SPIN_NS / 1e3:.2f} us; stock with the end trigger {stock_us:.2f} "
            f"us{'' if tuned else ', untuned'}; {len(results)} tactics timed, "
            f"{len(wrong)} wrong or failed {wrong[:4]})"
        )
        print(
            "  rank  tactic            entry us  end us  K tiles before wait  "
            "CTAs  smem B"
        )
        for rank, (us, tactic, gemms) in enumerate(shown, start=1):
            late = _serving_chain_us(
                lambda i: gemms[i % COPIES], rows, early=False, gen=gen, reps=10
            )
            cta_k, stages, sfb_stages, split_k = tactic
            smem = ll_gemm._smem_bytes(
                e4m3, e4m3, 32, cta_k, stages, sfb_stages, split_k
            )
            mark = " (pinned)" if tactic == pinned else ""
            print(
                f"  {rank:>4}  {str(tactic):<16}  {us:>8.2f}  {late:>6.2f}  "
                f"{_in_flight(tactic):>19}  {n // 128 * split_k:>4}  {smem}{mark}"
            )
        if results:
            us, best, _ = results[0]
            pinned_us = next((u for u, t, _ in results if t == pinned), float("nan"))
            lines.append(
                f"    ({sm}, {n}, {k}, {rows}): {best},  # q|k|v|index {us:.2f} us per "
                f"layer in the serving chain (pinned {pinned} {pinned_us:.2f} us, stock "
                f"{stock_us:.2f} us)"
            )
    print("SERVING_SWEEP_PINNED_TACTICS (mxfp8_ll.PINNED_TACTICS q|k|v|index entries):")
    for line in lines:
        print(line)
    assert not mismatched, f"serving-chain outputs differ from eager: {mismatched}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
