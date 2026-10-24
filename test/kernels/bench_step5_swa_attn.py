# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot swa_attn's microbenchmark: the SWA attention of a target-verify step inside its
real per-layer chain, as one CUDA graph over N_LAYERS layer buffers (cold windows):

  step5_qknorm_rope_gated (SGLang's, with the fused SWA K / V store; it triggers its
  dependents early) -> the attention -> o_proj stand-in (torch.mm [rows, 3072] x
  [3072, 4096] BF16: 25 MB of weights a layer, so 23 other layers' weights stream through
  L2 between two reads of one layer's window, as in serving).

Arms: "none" (o_proj on a static buffer: the chain without attention), FA4 as SGLang
calls it (one split, default tile, PDL, gate epilogue), FA4 with tile_mn (128, 64), and
the slot. The attention cost per layer = the arm's per-layer wall minus "none"'s.
Prints a CSV; asserts nothing (run it with pytest -s, or as a script; under nsys with
--cuda-graph-trace=node for per-kernel times: each arm's replays are one NVTX range).
  python3 test/kernels/bench_step5_swa_attn.py --bs 1 2 4 8 --ctx 8188 131068
"""

from __future__ import annotations

import argparse

import pytest
import torch

T, H, D, PAGE, WIN = 4, 16, 192, 64, 511


def build(bs, L, layers, device):
    g = torch.Generator(device="cpu").manual_seed(bs * 7919 + L)
    n_k = L + T
    lo = max(0, L - WIN)
    win_pages = list(range(lo // PAGE, (n_k - 1) // PAGE + 1))
    phys = 1 + bs * len(win_pages)
    page_table = torch.zeros(bs, (n_k + PAGE - 1) // PAGE + 1, dtype=torch.int32)
    perm = torch.randperm(phys - 1, generator=g) + 1
    for r in range(bs):
        page_table[r, win_pages] = perm[
            r * len(win_pages) : (r + 1) * len(win_pages)
        ].to(torch.int32)

    def rows():
        return torch.randn(phys * PAGE, D, generator=g).to(torch.bfloat16).to(device)

    loc = torch.tensor(
        [
            int(page_table[r, (L + i) // PAGE]) * PAGE + (L + i) % PAGE
            for r in range(bs)
            for i in range(T)
        ],
        dtype=torch.int64,
    )
    return dict(
        kv=[(rows(), rows()) for _ in range(layers)],
        page_table=page_table.to(device),
        loc=loc.to(device),
        pos=torch.tensor(
            [L + i for _ in range(bs) for i in range(T)], dtype=torch.int64
        ).to(device),
        qkvg=[
            (torch.randn(bs * T, 3472, generator=g) * 0.5).to(torch.bfloat16).to(device)
            for _ in range(layers)
        ],
        w_o=[
            (torch.randn(3072, 4096, generator=g) * 0.02).to(torch.bfloat16).to(device)
            for _ in range(layers)
        ],
        rope=torch.randn(n_k + 16, 192, generator=g).to(torch.bfloat16).to(device),
        norm_w=torch.zeros(192, dtype=torch.float32, device=device),
        o_static=torch.zeros(bs * T, H * D, dtype=torch.bfloat16, device=device),
        cache_seqlens=torch.full((bs,), n_k, dtype=torch.int32, device=device),
        cu=torch.arange(0, bs * T + 1, T, dtype=torch.int32, device=device),
    )


def arm(kind, bs, device):
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache

    if kind == "none":
        return lambda q, gate, k, v, st: st["o_static"]
    if kind.startswith("fa4"):
        tile = {"tile_mn": (128, 64)} if kind == "fa4_tile64" else {}

        def run(q, gate, k, v, st):
            o = flash_attn_with_kvcache(
                q=q,
                k_cache=k.view(-1, PAGE, 1, D),
                v_cache=v.view(-1, PAGE, 1, D),
                page_table=st["page_table"],
                cache_seqlens=st["cache_seqlens"],
                cu_seqlens_q=st["cu"],
                max_seqlen_q=T,
                softmax_scale=D**-0.5,
                causal=True,
                window_size=(WIN, 0),
                softcap=0.0,
                num_splits=1,
                head_gate=gate,
                use_pdl=True,
                ver=4,
                **tile,
            )
            return o.view(bs * T, H * D)

        return run
    from sgl_nvidia_mega.step5.sm103.attention.swa_split import SwaVerifyAttention

    att = SwaVerifyAttention(bs, device)
    att.warmup()
    return lambda q, gate, k, v, st: att(
        q, gate, k, v, st["page_table"], st["cache_seqlens"]
    )


def chain(st, attention, layers):
    from sglang.kernels.ops.attention.step5.qknorm_rope import step5_qknorm_rope_gated

    for i in range(layers):
        k, v = st["kv"][i]
        q, _, _, gate = step5_qknorm_rope_gated(
            st["qkvg"][i],
            st["norm_w"],
            st["norm_w"],
            st["pos"],
            st["rope"],
            eps=1e-6,
            kv_write=(k, v, st["loc"]),
        )
        torch.mm(attention(q, gate, k, v, st), st["w_o"][i])


def time_arm(st, attention, layers, label, iters=50):
    chain(st, attention, layers)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        chain(st, attention, layers)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    torch.cuda.nvtx.range_push(label)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    return start.elapsed_time(end) * 1000.0 / iters / layers


def run(bss, ctxs, layers=24, arms=("none", "fa4", "fa4_tile64", "slot")):
    device = torch.device("cuda")
    print("bs,L,arm,us_per_layer,attention_us_per_layer")
    for bs in bss:
        for L in ctxs:
            st = build(bs, L, layers, device)
            base = None
            for kind in arms:
                us = time_arm(
                    st, arm(kind, bs, device), layers, f"swa|bs={bs}|L={L}|{kind}"
                )
                base = us if kind == "none" else base
                print(f"{bs},{L},{kind},{us:.2f},{us - base:.2f}", flush=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM10x GPU",
)
def test_bench_swa_attn():
    run([1, 2, 4, 8], [8188, 131068])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--ctx", type=int, nargs="+", default=[8188, 131068])
    ap.add_argument("--layers", type=int, default=24)
    a = ap.parse_args()
    run(a.bs, a.ctx, a.layers)
