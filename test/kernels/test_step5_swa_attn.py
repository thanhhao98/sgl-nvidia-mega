# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot swa_attn's kernels (sm103/attention/swa_split.py) on one SM10x GPU (GB300 / B300),
at the deployed target-verify shape (4 rows a request, 16 heads, one KV head, head dim
192, window 511, page 64), against an FP64 reference of FA4's semantics and against
SGLang's own FA4 call on the same inputs (its arguments as FlashAttentionBackend passes
them for these layers), with the bounds runtime.TOLERANCES["swa_attn"] declares:

- vs FP64: the largest error in BF16 ulps at its (row, head) scale (the scale the next
  GEMM sees) and the relative Frobenius error;
- vs FA4: the same two, and no worse than FA4 against FP64 (rel_fro within 10 %);
- bitwise: run to run, CUDA-graph replay vs eager on fresh data, and batch invariance.

Every covered width (4-32 rows) eager over enough layer buffers to cycle L2 (cold
windows, as in serving) and replayed; every window edge and page alignment; padded rows.
"""

from __future__ import annotations

import random

import pytest
import torch

from sgl_nvidia_mega.step5 import runtime


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import triton  # noqa: F401
        from sglang.kernels.ops.attention.flash_attention import (  # noqa: F401
            flash_attn_with_kvcache,
        )
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _ready(), reason="needs an SM10x GPU and SGLang's FA4"
)

T, H, D, PAGE, WINDOW_LEFT = 4, 16, 192, 64, 511
TOL = runtime.TOLERANCES["swa_attn"]
ROWS = list(range(4, 33, 4))
# window edges, page alignments and long contexts (L = the sequence before this step)
EDGE_L = [
    0,
    1,
    3,
    60,
    63,
    64,
    65,
    127,
    128,
    300,
    507,
    508,
    509,
    510,
    511,
    512,
    513,
    514,
    515,
    516,
    575,
    576,
    577,
    1023,
    8188,
    131068,
]
L2_BYTES = 136 * 2**20


def _kernels(max_bs):
    from sgl_nvidia_mega.step5.sm103.attention.swa_split import SwaVerifyAttention

    return SwaVerifyAttention(max_bs, torch.device("cuda"))


def fa4(q, k_rows, v_rows, page_table, cache_seqlens, gate):
    """FlashAttentionBackend's FA4 call for a Step-5 SWA verify layer (deployed options:
    one split, the default tile, PDL, the gate epilogue, no window-aligned leftpad)."""
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache

    bs = cache_seqlens.shape[0]
    cu = torch.arange(0, bs * T + 1, T, dtype=torch.int32, device=q.device)
    o = flash_attn_with_kvcache(
        q=q.view(-1, H, D),
        k_cache=k_rows.view(-1, PAGE, 1, D),
        v_cache=v_rows.view(-1, PAGE, 1, D),
        page_table=page_table[:bs],
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu,
        max_seqlen_q=T,
        softmax_scale=D**-0.5,
        causal=True,
        window_size=(WINDOW_LEFT, 0),
        softcap=0.0,
        num_splits=1,
        head_gate=gate,
        use_pdl=True,
        ver=4,
    )
    return o.reshape(bs * T, H * D)


def reference(q, k_rows, v_rows, page_table, cache_seqlens, gate):
    """FP64: query token i of a request sees keys [L + i - 511, L + i], L = n_k - 4; the
    gate's sigmoid scales the normalized output; no visible key gives 0."""
    q, k_rows, v_rows = q.double().cpu(), k_rows.double().cpu(), v_rows.double().cpu()
    page_table, cache_seqlens, gate = page_table.cpu(), cache_seqlens.cpu(), gate.cpu()
    bs = cache_seqlens.numel()
    out = torch.zeros(bs * T, H, D, dtype=torch.float64)
    for r in range(bs):
        n_k = int(cache_seqlens[r])
        for i in range(T):
            q_abs = n_k - T + i
            lo, hi = max(0, q_abs - WINDOW_LEFT), min(q_abs, n_k - 1)
            if hi < lo:
                continue
            pos = torch.arange(lo, hi + 1)
            slots = page_table[r, pos // PAGE].long() * PAGE + pos % PAGE
            p = torch.softmax((q[r * T + i] @ k_rows[slots].T) * D**-0.5, dim=-1)
            out[r * T + i] = (p @ v_rows[slots]) * torch.sigmoid(
                gate[r * T + i].double()
            )[:, None]
    return out.reshape(bs * T, H * D)


def bf16_ulp(x):
    e = torch.floor(torch.log2(x.abs().double().clamp(min=2.0**-126)))
    return torch.exp2(e - 7)


def head_scale_ulp(a, ref):
    a, ref = a.double().cpu().reshape(-1, H, D), ref.double().cpu().reshape(-1, H, D)
    return ((a - ref).abs() / bf16_ulp(ref.abs().amax(dim=-1, keepdim=True))).flatten()


def rel_fro(a, ref):
    a, ref = a.double().cpu(), ref.double().cpu()
    return ((a - ref).norm() / ref.norm().clamp(min=1e-300)).item()


class Pool:
    """A real-shaped SWA pool over enough layer buffers to exceed L2, a shuffled page
    table (non-contiguous pages) and page 0 left as the padding page."""

    def __init__(self, seq_lens, layers=None, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.bs = len(seq_lens)
        per_req = (max(seq_lens) + T + PAGE - 1) // PAGE + 1
        pages = self.bs * per_req + 1
        self.layers = layers or max(2, -(-2 * L2_BYTES // (pages * PAGE * D * 4)))

        def rows():
            return torch.randn(pages * PAGE, D, generator=g).to(torch.bfloat16).cuda()

        self.k = [rows() for _ in range(self.layers)]
        self.v = [rows() for _ in range(self.layers)]
        perm = (torch.randperm(pages - 1, generator=g) + 1).to(torch.int32)
        self.page_table = (
            perm[: self.bs * per_req].view(self.bs, -1).contiguous().cuda()
        )
        self.cache_seqlens = torch.tensor(
            [L + T for L in seq_lens], dtype=torch.int32
        ).cuda()
        self.q = torch.randn(self.bs * T, H, D, generator=g).to(torch.bfloat16).cuda()
        self.gate = (
            (torch.randn(self.bs * T, H, generator=g) * 3).to(torch.bfloat16).cuda()
        )


def _check(out, q, k, v, page_table, cache_seqlens, gate):
    ref = reference(q, k, v, page_table, cache_seqlens, gate)
    o_fa4 = fa4(q, k, v, page_table, cache_seqlens, gate)
    assert torch.isfinite(out.float()).all()
    assert head_scale_ulp(out, ref).max() <= TOL["swa_attn.head_ulp_vs_fp64"]
    assert rel_fro(out, ref) <= TOL["swa_attn.rel_fro_vs_fp64"]
    assert head_scale_ulp(out, o_fa4).max() <= TOL["swa_attn.head_ulp_vs_fa4"]
    assert rel_fro(out, o_fa4) <= TOL["swa_attn.rel_fro_vs_fa4"]
    assert rel_fro(out, ref) <= 1.1 * rel_fro(o_fa4, ref) + 1e-4


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("graph", [False, True])
def test_every_width_eager_and_replay(rows, graph):
    bs = rows // T
    rnd = random.Random(rows)
    pool = Pool([rnd.choice(EDGE_L) for _ in range(bs)], seed=rows)
    att = _kernels(8)
    att.warmup()
    if not graph:
        for layer in range(pool.layers):  # cold windows: cycle the buffers past L2
            out = att(
                pool.q,
                pool.gate,
                pool.k[layer],
                pool.v[layer],
                pool.page_table,
                pool.cache_seqlens,
            )
            if layer in (0, pool.layers - 1):
                _check(
                    out,
                    pool.q,
                    pool.k[layer],
                    pool.v[layer],
                    pool.page_table,
                    pool.cache_seqlens,
                    pool.gate,
                )
        return
    out = torch.empty(rows, H * D, dtype=torch.bfloat16, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        att(
            pool.q,
            pool.gate,
            pool.k[0],
            pool.v[0],
            pool.page_table,
            pool.cache_seqlens,
            out=out,
        )
    for trial in range(8):  # replay on new data written into the captured buffers
        fresh = Pool(
            [rnd.choice(EDGE_L) for _ in range(bs)], layers=1, seed=1000 + trial
        )
        if fresh.k[0].shape[0] > pool.k[0].shape[0]:
            continue
        n = min(pool.page_table.shape[1], fresh.page_table.shape[1])
        pool.page_table.zero_()
        pool.page_table[:, :n].copy_(fresh.page_table[:, :n])
        for dst, src in (
            (pool.cache_seqlens, fresh.cache_seqlens),
            (pool.q, fresh.q),
            (pool.gate, fresh.gate),
        ):
            dst.copy_(src)
        pool.k[0][: fresh.k[0].shape[0]].copy_(fresh.k[0])
        pool.v[0][: fresh.v[0].shape[0]].copy_(fresh.v[0])
        g.replay()
        torch.cuda.synchronize()
        eager = att(
            pool.q, pool.gate, pool.k[0], pool.v[0], pool.page_table, pool.cache_seqlens
        )
        assert torch.equal(out, eager)
        _check(
            out,
            pool.q,
            pool.k[0],
            pool.v[0],
            pool.page_table,
            pool.cache_seqlens,
            pool.gate,
        )


@pytest.mark.parametrize("L", EDGE_L + [1000 + r for r in range(64)])
def test_window_edges_and_page_alignments(L):
    pool = Pool([L], layers=1, seed=L)
    out = _kernels(1)(
        pool.q, pool.gate, pool.k[0], pool.v[0], pool.page_table, pool.cache_seqlens
    )
    _check(
        out,
        pool.q,
        pool.k[0],
        pool.v[0],
        pool.page_table,
        pool.cache_seqlens,
        pool.gate,
    )


@pytest.mark.parametrize("bs", [1, 2, 3, 8])
def test_each_head_grouping_meets_the_bounds(bs):
    """Batch 1-2 and 3-8 launch different head groupings (swa_split.head_groups)."""
    rnd = random.Random(bs)
    for trial in range(4):
        pool = Pool(
            [rnd.choice(EDGE_L) for _ in range(bs)], layers=1, seed=bs * 10 + trial
        )
        out = _kernels(8)(
            pool.q, pool.gate, pool.k[0], pool.v[0], pool.page_table, pool.cache_seqlens
        )
        _check(
            out,
            pool.q,
            pool.k[0],
            pool.v[0],
            pool.page_table,
            pool.cache_seqlens,
            pool.gate,
        )


def test_padded_rows_stay_finite():
    """A padded request: sequence length 1 and a zero page row."""
    pool = Pool([5000, 1, 1, 1], layers=1)
    pool.page_table[1:].zero_()
    out = _kernels(4)(
        pool.q, pool.gate, pool.k[0], pool.v[0], pool.page_table, pool.cache_seqlens
    )
    assert torch.isfinite(out.float()).all()


def test_deterministic_and_batch_invariant():
    pool = Pool([131068, 513, 7, 4000, 600, 64, 65, 2000], layers=1)
    att = _kernels(8)
    a = att(
        pool.q, pool.gate, pool.k[0], pool.v[0], pool.page_table, pool.cache_seqlens
    ).clone()
    b = att(
        pool.q, pool.gate, pool.k[0], pool.v[0], pool.page_table, pool.cache_seqlens
    )
    assert torch.equal(a, b)

    def first(n):
        return att(
            pool.q[: n * T].contiguous(),
            pool.gate[: n * T].contiguous(),
            pool.k[0],
            pool.v[0],
            pool.page_table[:n],
            pool.cache_seqlens[:n],
        )

    # within one head grouping (swa_split.head_groups) a request's output does not
    # depend on its neighbours: batch 3 groups as batch 8 does, batch 1 as batch 2
    assert torch.equal(first(3), a[: 3 * T])
    assert torch.equal(first(1), first(2)[:T])
