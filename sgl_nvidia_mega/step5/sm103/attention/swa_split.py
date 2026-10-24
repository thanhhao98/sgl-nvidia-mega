# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Step-5's SWA attention of a target-verify step as a page-split kernel and a merge
(slot ``swa_attn``), in place of SGLang's FA4 call, which runs one CTA per request.

Semantics are FA4's call in SGLang's FlashAttentionBackend for these layers: window
(511, 0) with causal bottom-right alignment (query token t of a request sees keys
[L + t - 511, L + t], L = cache_seqlens - 4), softmax scale 192 ** -0.5, BF16 P into
the PV product, and the sigmoid head gate applied in FP32 to the normalized output
before its one BF16 rounding. The split-KV structure changes the reduction order, so the
slot is a tolerance slot (runtime.TOLERANCES["swa_attn"]).

Shapes per rank (R = 4 * bs verify rows, 16 query heads, one KV head, head dim 192):
  q              [R, 16, 192] bf16 contiguous (request r's rows are [4 r, 4 r + 4))
  gate           [R, 16] bf16 contiguous
  k_rows, v_rows one SWA layer's pool, [slots, 192] bf16 contiguous (page 64)
  page_table     [bs, max_pages] int32, unit stride: FA4's swa_page_table (the SWA
                 page of each logical page of the request)
  cache_seqlens  [bs] int32: FA4's cache_seqlens_int32 (the 4 new tokens included)
  out            [R, 16 * 192] bf16

Split: grid (bs, NUM_SPLITS, HG); CTA (r, s, g) owns logical page first_page + s of
request r's union window [max(0, L - 511), L + 4) and the 4 x 16 / HG query rows of head
group g. Before griddepcontrol.wait it reads cache_seqlens and the page table (both
filled before the graph replays) and the K / V rows earlier forwards wrote; after the
wait, Q and the rows of this step's 4 tokens, which the PDL predecessor (the fused
q/k-norm + RoPE + SWA store) writes. One QK^T, the window mask, a page-local softmax in
the log2 domain, PV; FP32 partials (O, max, sum). No page outside the window is read:
SWA eviction may have freed it.
Merge: grid (bs, 16, 3); CTA (r, h, t) loads every split's partials of head h's 4 rows
for dims [64 t, 64 t + 64) in one round, merges them in fixed split order
(deterministic), applies the gate and rounds once; it triggers its dependents at entry
so the o_proj GEMM can stream its weights while the split runs out.

Head groups: 4 (16 rows a CTA) at batch 1-2, 2 above; the merge tiled over the head
dims with 2 warps (B300, chain microbenchmark test/kernels/bench_step5_swa_attn.py: one
wave of split CTAs, and fewer loads a merge CTA, win).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

T = 4  # verify rows a request
H = 16  # query heads a rank
D = 192
D_LO = 128  # Triton tiles are powers of two: 192 = 128 + 64
D_HI = 64
PAGE = 64
WINDOW_LEFT = 511
SCALE_LOG2 = D**-0.5 * 1.4426950408889634
# the 515-key union window [L - 511, L + 4) spans at most 10 logical pages
NUM_SPLITS = (PAGE - 1 + WINDOW_LEFT + T + PAGE - 1) // PAGE
SPLITS_P2 = triton.next_power_of_2(NUM_SPLITS)
SPLIT_WARPS = 4
MERGE_DT = 64  # head dims a merge CTA reduces
MERGE_WARPS = 2


def head_groups(bs: int) -> int:
    """CTAs a split's 16 heads are spread over at batch ``bs``."""
    return 4 if bs <= 2 else 2


@triton.jit(do_not_specialize=["pt_stride"])
def _split_kernel(
    Q,
    K,
    V,
    PT,
    SEQ,
    PO,
    PM,
    PL,
    pt_stride,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    D_LO: tl.constexpr,
    D_HI: tl.constexpr,
    PAGE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    HG: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    g = tl.program_id(2)
    M: tl.constexpr = T * H
    HH: tl.constexpr = H // HG
    MG: tl.constexpr = T * HH

    # filled before the replay, so safe to read before the wait
    n_k = tl.load(SEQ + r)
    L = n_k - T
    lo = tl.maximum(L - WINDOW_LEFT, 0)
    offs_n = tl.arange(0, PAGE)
    pos = (lo // PAGE + s) * PAGE + offs_n
    in_range = (pos >= lo) & (pos < n_k)
    page = tl.load(PT + r * pt_stride + pos // PAGE, mask=in_range, other=0)
    slot = page.to(tl.int64) * PAGE + (pos % PAGE)
    d_lo = tl.arange(0, D_LO)
    d_hi = D_LO + tl.arange(0, D_HI)
    off_lo = slot[:, None] * D + d_lo[None, :]
    off_hi = slot[:, None] * D + d_hi[None, :]
    old = in_range & (pos < L)  # earlier forwards' rows
    new = in_range & (pos >= L)  # this step's rows: the predecessor stores them
    k_lo = tl.load(K + off_lo, mask=old[:, None], other=0.0)
    k_hi = tl.load(K + off_hi, mask=old[:, None], other=0.0)
    v_lo = tl.load(V + off_lo, mask=old[:, None], other=0.0)
    v_hi = tl.load(V + off_hi, mask=old[:, None], other=0.0)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    k_lo = tl.where(
        new[:, None], tl.load(K + off_lo, mask=new[:, None], other=0.0), k_lo
    )
    k_hi = tl.where(
        new[:, None], tl.load(K + off_hi, mask=new[:, None], other=0.0), k_hi
    )
    v_lo = tl.where(
        new[:, None], tl.load(V + off_lo, mask=new[:, None], other=0.0), v_lo
    )
    v_hi = tl.where(
        new[:, None], tl.load(V + off_hi, mask=new[:, None], other=0.0), v_hi
    )
    offs_g = tl.arange(0, MG)
    t_of = offs_g // HH
    offs_m = t_of * H + g * HH + offs_g % HH  # the (token, head) row of the request
    q_row = r * M + offs_m
    q_lo = tl.load(Q + q_row[:, None] * D + d_lo[None, :])
    q_hi = tl.load(Q + q_row[:, None] * D + d_hi[None, :])

    qk = tl.dot(q_lo, tl.trans(k_lo))
    qk = tl.dot(q_hi, tl.trans(k_hi), acc=qk)
    q_abs = L + t_of
    vis = (
        in_range[None, :]
        & (pos[None, :] <= q_abs[:, None])
        & (pos[None, :] >= q_abs[:, None] - WINDOW_LEFT)
    )
    x = tl.where(vis, qk * SCALE_LOG2, float("-inf"))
    m = tl.max(x, axis=1)
    p = tl.exp2(x - tl.where(m == float("-inf"), 0.0, m)[:, None])  # masked -> 0
    lsum = tl.sum(p, axis=1)
    pb = p.to(tl.bfloat16)  # FA4 feeds BF16 P to its PV product too
    o_lo = tl.dot(pb, v_lo)
    o_hi = tl.dot(pb, v_hi)

    base = (r * NUM_SPLITS + s) * M + offs_m
    tl.store(PM + base, m)
    tl.store(PL + base, lsum)
    tl.store(PO + base[:, None] * D + d_lo[None, :], o_lo)
    tl.store(PO + base[:, None] * D + d_hi[None, :], o_hi)


@triton.jit
def _merge_kernel(
    PO,
    PM,
    PL,
    GATE,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    DT: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SP: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    d = tl.program_id(2) * DT + tl.arange(0, DT)  # this CTA's slice of the head dims
    M: tl.constexpr = T * H
    rows = tl.arange(0, T) * H + h
    offs_s = tl.arange(0, SP)
    smask = offs_s < NUM_SPLITS
    # the gate is the q/k-norm kernel's output, complete before any split passed its
    # wait; this grid starts only after every split CTA exited
    gates = tl.load(GATE + r * M + rows).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()  # o_proj streams its weights meanwhile
        tl.extra.cuda.gdc_wait()
    # an empty split holds m = -inf, l = 0, O = 0: only the padded splits are masked
    idx = (r * NUM_SPLITS + offs_s[None, :]) * M + rows[:, None]
    m_s = tl.load(PM + idx, mask=smask[None, :], other=float("-inf"))
    l_s = tl.load(PL + idx, mask=smask[None, :], other=0.0)
    o = tl.load(
        PO + idx[:, :, None] * D + d[None, None, :],
        mask=smask[None, :, None],
        other=0.0,
    )
    mx = tl.max(m_s, axis=1)
    w = tl.exp2(m_s - tl.where(mx == float("-inf"), 0.0, mx)[:, None])
    denom = tl.sum(w * l_s, axis=1)
    acc = tl.sum(o * w[:, :, None], axis=1)
    sig = 1.0 / (1.0 + tl.exp2(-gates * 1.4426950408889634))
    scale = tl.where(denom > 0.0, sig / denom, 0.0)  # an empty row -> 0
    out = OUT + (r * M + rows)[:, None] * D
    tl.store(out + d[None, :], (acc * scale[:, None]).to(tl.bfloat16))


class SwaVerifyAttention:
    """Graph-safe launcher for batches 1 to ``max_bs``: the FP32 partials are allocated
    once and shared by every SWA layer (a layer's merge completes before the next
    layer's split passes its wait: the q/k-norm kernel between them is an ordinary
    stream-ordered launch), and every launch's grid depends on the batch alone."""

    def __init__(self, max_bs: int, device: torch.device, *, pdl: bool = True):
        self.max_bs, self.pdl = max_bs, pdl
        f32, m = torch.float32, T * H
        self.po = torch.empty(max_bs, NUM_SPLITS, m, D, dtype=f32, device=device)
        self.pm = torch.empty(max_bs, NUM_SPLITS, m, dtype=f32, device=device)
        self.pl = torch.empty(max_bs, NUM_SPLITS, m, dtype=f32, device=device)

    def scratch_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.po, self.pm, self.pl))

    def __call__(self, q, gate, k_rows, v_rows, page_table, cache_seqlens, out=None):
        bs = cache_seqlens.shape[0]
        if not 1 <= bs <= self.max_bs:
            raise ValueError(f"batch {bs} is outside 1-{self.max_bs}")
        if (
            q.shape != (bs * T, H, D)
            or q.dtype != torch.bfloat16
            or not q.is_contiguous()
        ):
            raise ValueError(
                f"q {tuple(q.shape)} {q.dtype}: want [{bs * T}, {H}, {D}] bf16 contiguous"
            )
        if (
            gate.shape != (bs * T, H)
            or gate.dtype != torch.bfloat16
            or not gate.is_contiguous()
        ):
            raise ValueError(
                f"gate {tuple(gate.shape)} {gate.dtype}: want [{bs * T}, {H}] bf16 contiguous"
            )
        for name, t in (("k_rows", k_rows), ("v_rows", v_rows)):
            if (
                t.ndim != 2
                or t.shape[1] != D
                or t.dtype != torch.bfloat16
                or not t.is_contiguous()
            ):
                raise ValueError(
                    f"{name} {tuple(t.shape)} {t.dtype}: want [slots, {D}] bf16 contiguous"
                )
        if (
            page_table.dtype != torch.int32
            or page_table.stride(1) != 1
            or page_table.shape[0] < bs
        ):
            raise ValueError("page_table: want [>= bs, pages] int32 with unit stride")
        if cache_seqlens.dtype != torch.int32 or cache_seqlens.ndim != 1:
            raise ValueError("cache_seqlens: want [bs] int32")
        if out is None:
            out = torch.empty(bs * T, H * D, dtype=torch.bfloat16, device=q.device)
        pdl = {"launch_pdl": True} if self.pdl else {}
        _split_kernel[(bs, NUM_SPLITS, head_groups(bs))](
            q,
            k_rows,
            v_rows,
            page_table,
            cache_seqlens,
            self.po,
            self.pm,
            self.pl,
            page_table.stride(0),
            T=T,
            H=H,
            D=D,
            D_LO=D_LO,
            D_HI=D_HI,
            PAGE=PAGE,
            NUM_SPLITS=NUM_SPLITS,
            WINDOW_LEFT=WINDOW_LEFT,
            SCALE_LOG2=SCALE_LOG2,
            HG=head_groups(bs),
            USE_PDL=self.pdl,
            num_warps=SPLIT_WARPS,
            num_stages=1,
            **pdl,
        )
        _merge_kernel[(bs, H, D // MERGE_DT)](
            self.po,
            self.pm,
            self.pl,
            gate,
            out,
            T=T,
            H=H,
            D=D,
            DT=MERGE_DT,
            NUM_SPLITS=NUM_SPLITS,
            SP=SPLITS_P2,
            USE_PDL=self.pdl,
            num_warps=MERGE_WARPS,
            num_stages=1,
            **pdl,
        )
        return out

    def warmup(self) -> None:
        """Compile every kernel variant a batch of 1 to ``max_bs`` launches (Triton
        compiles at the first launch, which a CUDA-graph capture does not allow)."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm the SWA verify kernels up before any capture")
        device = self.po.device
        k = torch.zeros(4 * PAGE, D, dtype=torch.bfloat16, device=device)
        for bs in range(1, self.max_bs + 1):
            q = torch.zeros(bs * T, H, D, dtype=torch.bfloat16, device=device)
            gate = torch.zeros(bs * T, H, dtype=torch.bfloat16, device=device)
            table = torch.zeros(bs, 16, dtype=torch.int32, device=device)
            seqlens = torch.full((bs,), T, dtype=torch.int32, device=device)
            self(q, gate, k, k, table, seqlens)
        torch.cuda.synchronize(device)
