# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot hop_mxfp8's entry trigger for slot attn_ll on 4 GPUs: SGLang's push-plane
all-reduce + residual + Gemma norm boundary kernel in its MXFP8 form, after a dense
layer (plain all-reduce) and after a MoE layer (deferred finalize + shared add +
all-reduce), at TP4 on the CustomAllReduceV2 push plane as M3.1 registers it (4 MiB
push slots, 512 row counters), M3.1's shapes (hidden 6144, top-4, fp32 routing
weights), followed by slot attn_ll's low-latency q|k|v|index GEMM (``mxfp8_ll``).

The served order the entry trigger (``trigger_at_entry``) is for, at 1-8 rows (needs
FlashInfer's low-latency MXFP8 GEMM): the boundary kernel signalling at entry, then
the low-latency GEMM on its pair with the pinned tactic and the two that keep every K
tile in flight, alone and followed by SGLang's q/k norm + RoPE kernel (a tactic that
keeps every K tile in flight signals the norm before the GEMM's own wait, so the norm
then launches while the boundary kernel still runs), captured in CUDA graphs and
replayed on new inputs: the boundary's outputs, the GEMM's and the norm's (in place,
FP8 q and index q), bit for bit, against the same chains without the entry trigger
and against today's kernel + FlashInfer's quantize + the same GEMM and norm called
eagerly. Rank 0 then prints the chain's time per layer (boundary, GEMM, q/k norm +
RoPE) with and without the trigger, and from torch.profiler's kernel records the
GEMM's start after the boundary kernel's start and end, and the boundary kernel's
duration, where the profiler records the kernels of graph replays.

The MXFP8 form itself, with and without the entry trigger (against today's kernel
followed by FlashInfer's mxfp8_quantize at every width the slot serves, under graph
replay, and the stock qkv|index GEMM on its pair), is SGLang's test, with the ops
(test/registered/kernels/ops/communication/test_all_reduce_residual_norm_quant.py).

Run (one process per GPU is spawned):
    python3 -m pytest -q -s test/kernels/test_minimax_m31_hop_mxfp8.py
"""

import os
import statistics
import sys
import tempfile
import time
import traceback
from functools import partial

import pytest
import torch

pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

WORLD_SIZE = 4
H = 6144
TOP_K = 4
EPS = 1e-6
N_QKV = 2560  # one TP4 rank's fused q|k|v|index_q|index_k projection
TIME_LAUNCHES = 20  # launches per captured graph
TIME_REPLAYS = 10
TIMEOUT_S = 1800  # cold JIT builds of both kernel families
# the entry trigger with slot attn_ll's low-latency q|k|v|index GEMM (1-8 rows), the
# tactics besides the pinned one: every K tile of a CTA in flight before its wait
EARLY_ROWS = (1, 4, 8)
ALL_TILES_IN_FLIGHT = ((128, 12, 1, 4), (256, 6, 1, 4))


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= WORLD_SIZE
        and torch.cuda.get_device_capability()[0] == 10
    )


def _init(rank: int, init_method: str):
    """The world group and a push-only CustomAllReduceV2 plane registered for the
    fused kernels, as test_all_reduce_residual_norm.py and M3.1 set them up."""
    import sglang.srt.distributed.parallel_state as ps
    import torch.distributed as dist
    from sglang.kernels.ops.communication import all_reduce_fusion
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.runtime_context import get_parallel

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=WORLD_SIZE,
    )
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(WORLD_SIZE)), local_rank=rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    comm = CustomAllReduceV2(
        coord.cpu_group,
        torch.device(f"cuda:{rank}"),
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=4 * 1024 * 1024,
        max_push_blocks=512,
    )
    assert not comm.disabled
    all_reduce_fusion.register_comm(comm.obj)
    return coord, comm


def _inputs(rows: int, rank: int, finalize: bool, seed: int):
    """One boundary's inputs. The residual stream, the norm weight and the routing are
    the same on every rank; the partial sums (the MoE's GEMM2 rows and shared expert
    output) are each rank's own. The norm weight's 32-column groups span 2^-12..2^11,
    so the UE8M0 scales span many exponents, and one 32-column group sums to zero on
    every rank (an all-zero group: scale 0)."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    mine = torch.Generator(device=dev).manual_seed(seed * 16 + rank + 1)
    residual = torch.randn(rows, H, generator=g, device=dev).bfloat16()
    spread = torch.exp2(torch.arange(H // 32, device=dev) % 24 - 12.0)
    gamma = (
        1 + 0.1 * torch.randn(H, generator=g, device=dev)
    ) * spread.repeat_interleave(32)
    gamma = gamma.bfloat16()
    zero = slice(64, 96)
    residual[:, zero] = 0
    if not finalize:
        partial = torch.randn(rows, H, generator=mine, device=dev) * (rank + 1) / 4
        partial = partial.bfloat16()
        partial[:, zero] = 0
        return dict(partial=partial, residual=residual, gamma=gamma)
    permuted = rows * TOP_K + 5
    gemm2 = (torch.randn(permuted, H, generator=mine, device=dev) / 4).bfloat16()
    gemm2[:, zero] = 0
    idx = torch.randperm(permuted, generator=g, device=dev)[: rows * TOP_K].int()
    weights = torch.rand(rows, TOP_K, generator=g, device=dev) * 2.0
    shared = (torch.randn(rows, H, generator=mine, device=dev) / 4).bfloat16()
    shared[:, zero] = 0
    return dict(
        gemm2=gemm2,
        idx=idx,
        weights=weights,
        shared=shared,
        residual=residual,
        gamma=gamma,
    )


def _stock_kernel(inp, finalize, geometry=None):
    """Today's kernel, as CustomAllReduceFusion launches it: (normed, residual_out)."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops

    launch = dict(zip(("two_shot", "vecs_per_thread", "cluster_size"), geometry or ()))
    residual_out = torch.empty_like(inp["residual"])
    if finalize:
        return ops.moe_finalize_all_reduce_residual_norm(
            inp["gemm2"],
            inp["idx"],
            inp["weights"],
            TOP_K,
            inp["shared"],
            inp["residual"],
            inp["gamma"],
            EPS,
            world_size=WORLD_SIZE,
            hidden_dim=H,
            residual_out=residual_out,
            **launch,
        )
    return ops.all_reduce_residual_norm(
        inp["partial"],
        inp["residual"],
        inp["gamma"],
        EPS,
        world_size=WORLD_SIZE,
        residual_out=residual_out,
        **launch,
    )


def _stock(inp, finalize, geometry=None):
    """Today's kernel, then FlashInfer's MXFP8 quantize of its normed row, as
    Fp8LinearMethod's GEMM path does."""
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    normed, residual_out = _stock_kernel(inp, finalize, geometry)
    q, sf = flashinfer_mxfp8_quantize(normed, is_sf_swizzled_layout=True, alignment=32)
    return normed, residual_out, q, sf


def _quant(
    inp, finalize, buffers, geometry=None, residual_out=None, trigger_at_entry=False
):
    """The MXFP8 form of the kernel, writing into ``buffers`` (normed, codes, scales);
    ``trigger_at_entry``: it signals the next kernel's launch at its entry."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops

    rows = inp["residual"].shape[0]
    normed, data, scales = buffers
    launch = dict(zip(("two_shot", "vecs_per_thread", "cluster_size"), geometry or ()))
    outputs = dict(
        residual_out=(
            torch.empty_like(inp["residual"]) if residual_out is None else residual_out
        ),
        out=normed[:rows],
        quantized=data[:rows],
        scales=scales,
        trigger_at_entry=trigger_at_entry,
        **launch,
    )
    if finalize:
        return ops.moe_finalize_all_reduce_residual_norm_quant(
            inp["gemm2"],
            inp["idx"],
            inp["weights"],
            TOP_K,
            inp["shared"],
            inp["residual"],
            inp["gamma"],
            EPS,
            world_size=WORLD_SIZE,
            hidden_dim=H,
            **outputs,
        )
    return ops.all_reduce_residual_norm_quant(
        inp["partial"],
        inp["residual"],
        inp["gamma"],
        EPS,
        world_size=WORLD_SIZE,
        **outputs,
    )


def _buffers(dev):
    """The slot's persistent buffers: sized for 128 rows, one 128-row scale tile."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops

    return (
        torch.empty(128, H, dtype=torch.bfloat16, device=dev),
        torch.empty(128, H, dtype=torch.float8_e4m3fn, device=dev),
        torch.empty(ops.mxfp8_scale_bytes(H), dtype=torch.uint8, device=dev),
    )


def _qkv_weight(dev):
    """One TP4 rank's qkv|index projection weight [N_QKV, H] in MXFP8, its scales in
    the 128x4 layout the layer keeps (weight_scale_inv_swizzled)."""
    from flashinfer import mxfp8_quantize

    g = torch.Generator(device=dev).manual_seed(4242)
    w = torch.randn(N_QKV, H, generator=g, device=dev) / H**0.5
    weight, weight_sf = mxfp8_quantize(w.bfloat16(), is_sf_swizzled_layout=True)
    return weight, weight_sf.reshape(-1)


def _poison(buffers):
    """Garbage in every output byte, so a byte the kernel does not write shows."""
    normed, data, scales = buffers
    normed.fill_(float("nan"))
    data.view(torch.uint8).fill_(0x5A)
    scales.fill_(0xA5)


def _differ(name, got, want):
    """'' when bit for bit equal, else what differs."""
    if got.shape != want.shape or got.dtype != want.dtype:
        return f"{name}: {tuple(got.shape)} {got.dtype} vs {tuple(want.shape)} {want.dtype}"
    a, b = got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8)
    bad = (a != b).nonzero()
    if bad.numel() == 0:
        return ""
    return f"{name}: {bad.shape[0]} of {a.numel()} bytes differ, first at {bad[0].tolist()}"


def _compare(tag, got, want, failures):
    names = ("normed", "residual_out", "e4m3 codes", "scales")
    diffs = [d for d in (_differ(n, g, w) for n, g, w in zip(names, got, want)) if d]
    if diffs:
        failures.append(f"{tag}: " + "; ".join(diffs))
    return not diffs


def _graph_time_us(fn, coord) -> float:
    """Microseconds per call of ``fn``: TIME_LAUNCHES calls captured in one CUDA graph,
    replayed TIME_REPLAYS times once the ranks line up (each push-plane launch waits
    for its peers)."""
    import torch.distributed as dist

    fn()  # warm up
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(TIME_LAUNCHES):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier(group=coord.cpu_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIME_REPLAYS):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / (TIME_LAUNCHES * TIME_REPLAYS)


# --------------------------- the entry trigger with attn_ll's low-latency q|k|v|index GEMM

BOUNDARY_KERNEL = "moe_finalize_all_reduce_kernel"
LL_GEMM_KERNEL = "LowLatencyBlockscaledGemmKernel"
ROPE_THETA = 1e7
MAX_POS = 1 << 16


def _ll_skip_reason():
    """Why FlashInfer's low-latency MXFP8 GEMM (slot attn_ll) cannot run here, else
    None."""
    try:
        import cutlass  # noqa: F401
        import flashinfer  # noqa: F401

        from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll
    except ImportError as e:
        return f"needs FlashInfer and the CuTe DSL ({e})"
    if not mxfp8_ll.supported(mxfp8_ll.sm_version()):
        return "FlashInfer has no low-latency MXFP8 GEMM on this GPU"
    return None


def _ll_tactics(rows):
    """The q|k|v|index tactics the chain runs at ``rows``: the pinned one, then those
    that keep every K tile of a CTA in flight before its PDL wait, where FlashInfer
    offers them."""
    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    pinned = mxfp8_ll.pinned_tactic(mxfp8_ll.sm_version(), N_QKV, H, rows)
    offered = mxfp8_ll.offered_tactics(N_QKV, H, mxfp8_ll.row_bucket(rows))
    return [pinned] + [t for t in ALL_TILES_IN_FLIGHT if t != pinned and t in offered]


def _ll_gemm(data, scales, weight, weight_sf, tactic):
    """Slot attn_ll's q|k|v|index GEMM on the MXFP8 pair ``(data, scales)``."""
    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    return mxfp8_ll.linear(data, scales, weight, weight_sf, tactic=tactic)


def _tiles(tactic):
    from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll

    return (
        f"{mxfp8_ll.tiles_before_wait(tactic, H)} of "
        f"{mxfp8_ll.k_tiles_per_cta(tactic, H)} K tiles before the wait"
    )


class _QkNorm:
    """SGLang's q/k norm + RoPE kernel as the attention launches it on the q|k|v|index
    GEMM's output (in place; FP8 q and index q out): the kernel after the GEMM in a
    served step, launched with PDL, reading only its norm weights and the positions
    before its own wait."""

    def __init__(self, rows, dev):
        g = torch.Generator(device=dev).manual_seed(99)
        self.groups = [
            ((0.1 * torch.randn(128, generator=g, device=dev)).bfloat16(), off, cnt)
            for off, cnt in ((0, 16), (16, 1), (18, 1), (19, 1))
        ]
        dims = torch.arange(0, 64, 2, dtype=torch.float, device=dev) / 64
        freqs = torch.outer(
            torch.arange(MAX_POS, dtype=torch.float, device=dev),
            1.0 / ROPE_THETA**dims,
        )
        self.cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()
        self.positions = torch.randint(0, MAX_POS, (rows,), device=dev, generator=g)
        self.fp8_q = torch.empty(rows, 16, 128, dtype=torch.float8_e4m3fn, device=dev)
        self.fp8_idx_q = torch.empty(
            rows, 1, 128, dtype=torch.float8_e4m3fn, device=dev
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


def _after(norm, fused, normed):
    """What a chain leaves after the boundary kernel, cloned, as ``(name, tensor)``:
    the low-latency GEMM's output; or, with the q/k norm + RoPE after it, the GEMM's
    output as the norm rewrote it in place and the norm's FP8 q and index q."""
    if not normed:
        return (("low-latency GEMM", fused.clone()),)
    return (
        ("q/k norm's q|k|v|index", fused.clone()),
        ("q/k norm's FP8 q", norm.fp8_q.clone()),
        ("q/k norm's FP8 index q", norm.fp8_idx_q.clone()),
    )


def _poison_norm(norm):
    """e4m3 NaN codes in the q/k norm's FP8 outputs, so a byte it does not write
    shows."""
    norm.fp8_q.view(torch.uint8).fill_(0x7F)
    norm.fp8_idx_q.view(torch.uint8).fill_(0x7F)


def _compare_after(tag, got, want, failures):
    diffs = [
        d
        for d in (
            _differ(f"{tag} {name}", g, w) for (name, g), (_, w) in zip(got, want)
        )
        if d
    ]
    failures.extend(diffs)
    return not diffs


def _check_early_trigger(coord, rank, failures, log):
    """The served order the entry trigger is for, captured in CUDA graphs: the
    boundary kernel from the slot's persistent buffers, then attn_ll's low-latency
    q|k|v|index GEMM on its pair (nothing between them), alone and followed by SGLang's
    q/k norm + RoPE kernel as the attention launches it (in place on the GEMM's
    output; FP8 q and index q out). A tactic that keeps every K tile of a CTA in flight
    signals the norm before the GEMM's own PDL wait, so with the entry trigger the
    norm launches while the boundary kernel still runs. Replayed on new inputs, with
    the boundary signalling at its entry and at its end; against each other and
    against today's kernel + FlashInfer's quantize + the same GEMM (and norm) called
    eagerly on the quantize's pair, bit for bit."""
    dev = torch.device("cuda")
    weight, weight_sf = _qkv_weight(dev)
    buffers = _buffers(dev)
    # (the boundary signals at its entry, the q/k norm follows the GEMM)
    arms = [(early, normed) for normed in (False, True) for early in (False, True)]
    for finalize in (False, True):
        variant = "finalize" if finalize else "plain"
        for rows in EARLY_ROWS:
            norm = _QkNorm(rows, dev)
            for tactic in _ll_tactics(rows):
                static = _inputs(rows, rank, finalize, seed=7000 + rows)
                residual_out = torch.empty_like(static["residual"])

                def step(
                    early,
                    normed,
                    static=static,
                    residual_out=residual_out,
                    tactic=tactic,
                ):
                    out = _quant(
                        static,
                        finalize,
                        buffers,
                        residual_out=residual_out,
                        trigger_at_entry=early,
                    )
                    gemm = _ll_gemm(out[2], out[3], weight, weight_sf, tactic)
                    if normed:
                        norm(gemm)  # in place; FP8 q and index q into norm's buffers
                    return out, gemm

                for arm in arms:
                    step(*arm)  # warm up: the JIT builds, the GEMM's compiled kernel
                torch.cuda.synchronize()
                graphs = {}
                for arm in arms:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = step(*arm)
                    graphs[arm] = (graph, captured)
                fresh = _inputs(rows, rank, finalize, seed=8000 + rows)
                want = _stock(fresh, finalize)
                want_gemm = _ll_gemm(want[2], want[3], weight, weight_sf, tactic)
                want_after = {False: _after(norm, want_gemm, False)}
                want_after[True] = _after(norm, norm(want_gemm.clone()), True)
                for key, value in fresh.items():
                    static[key].copy_(value)
                got = {}
                for arm, (graph, (out, gemm)) in graphs.items():
                    _poison(buffers)
                    residual_out.fill_(float("nan"))
                    gemm.fill_(float("nan"))
                    _poison_norm(norm)
                    graph.replay()
                    torch.cuda.synchronize()
                    got[arm] = ([t.clone() for t in out], _after(norm, gemm, arm[1]))
                tag = f"{variant} rows={rows} tactic={tactic} ({_tiles(tactic)})"
                ok = True
                for (early, normed), (out, after) in got.items():
                    then = " + q/k norm" if normed else ""
                    chain = f"{tag} {'entry' if early else 'end'} trigger{then}"
                    ok &= _compare(f"{chain} graph replay", out, want, failures)
                    ok &= _compare_after(
                        f"{chain}:", after, want_after[normed], failures
                    )
                for normed in (False, True):
                    then = " + q/k norm" if normed else ""
                    ok &= _compare_after(
                        f"{tag}{then}, entry vs end trigger:",
                        got[True, normed][1],
                        got[False, normed][1],
                        failures,
                    )
                log(
                    f"{tag}: boundary outputs, low-latency GEMM and the q/k norm + "
                    f"RoPE after it {'bitwise' if ok else 'DIFFER'} with the entry "
                    f"trigger, the end trigger and today's kernel + quantize"
                )


def _kernel_offsets(fn, coord, rank):
    """Rank 0, from torch.profiler's kernel records of TIME_REPLAYS replays of a graph
    of TIME_LAUNCHES calls of ``fn``: the medians, in microseconds, of the GEMM's start
    after the boundary kernel's start and after its end, and of the boundary kernel's
    duration; a string saying why where that cannot be measured. Every rank replays
    alike (each boundary launch waits for its peers); the others return None."""
    import torch.distributed as dist
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(TIME_LAUNCHES):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    prof, failure = None, None
    if rank == 0:
        try:
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.start()
        except Exception as e:  # report it; the replays below must run on every rank
            prof, failure = None, e
    dist.barrier(group=coord.cpu_group)
    for _ in range(TIME_REPLAYS):
        graph.replay()
    torch.cuda.synchronize()
    if rank != 0:
        return None
    if prof is None:
        return f"not measurable (torch.profiler: {failure!r})"
    prof.stop()
    spans = sorted(
        (e.time_range.start, e.time_range.end, e.name)
        for e in prof.events()
        if e.device_type == DeviceType.CUDA
        and (BOUNDARY_KERNEL in e.name or LL_GEMM_KERNEL in e.name)
    )
    after_start, after_end, durations, boundary = [], [], [], None
    for start, end, name in spans:
        if BOUNDARY_KERNEL in name:
            boundary = (start, end)
            durations.append(end - start)
        elif boundary is not None:
            after_start.append(start - boundary[0])
            after_end.append(start - boundary[1])
            boundary = None
    if not after_start:
        return "not measurable (no kernel records of the graph replays)"
    median = statistics.median
    return (
        f"GEMM start {median(after_start):.2f} us after the boundary's start, "
        f"{median(after_end):+.2f} us after its end; boundary "
        f"{median(durations):.2f} us ({len(after_start)} pairs, p50)"
    )


def _time_early_trigger(coord, rank, log):
    """Prints, checks nothing: per layer of the served chain (the boundary kernel,
    attn_ll's low-latency q|k|v|index GEMM, the q/k norm + RoPE after it), the time
    from CUDA graphs with the boundary signalling at its end and at its entry, and
    where torch.profiler records the kernels of graph replays, when the GEMM starts."""
    dev = torch.device("cuda")
    weight, weight_sf = _qkv_weight(dev)
    buffers = _buffers(dev)
    for finalize in (False, True):
        variant = "finalize" if finalize else "plain"
        for rows in EARLY_ROWS:
            inp = _inputs(rows, rank, finalize, seed=9000 + rows)
            residual_out = torch.empty_like(inp["residual"])
            norm = _QkNorm(rows, dev)
            for tactic in _ll_tactics(rows):

                def layer(early, inp=inp, residual_out=residual_out, tactic=tactic):
                    out = _quant(
                        inp,
                        finalize,
                        buffers,
                        residual_out=residual_out,
                        trigger_at_entry=early,
                    )
                    norm(_ll_gemm(out[2], out[3], weight, weight_sf, tactic))

                us = {
                    early: _graph_time_us(partial(layer, early), coord)
                    for early in (False, True)
                }
                where = {
                    early: _kernel_offsets(partial(layer, early), coord, rank)
                    for early in (False, True)
                }
                log(
                    f"{variant} rows={rows} tactic={tactic} ({_tiles(tactic)}): "
                    f"boundary + GEMM + q/k norm {us[False]:.2f} us per layer with the "
                    f"end trigger, {us[True]:.2f} us with the entry trigger "
                    f"({us[True] - us[False]:+.2f} us); end trigger: {where[False]}; "
                    f"entry trigger: {where[True]}"
                )


def _early_worker(rank: int, init_method: str) -> None:
    coord, comm = _init(rank, init_method)

    def log(message):
        if rank == 0:
            print(f"[hop_mxfp8 entry trigger] {message}", flush=True)

    failures = []
    try:
        with torch.inference_mode():
            _check_early_trigger(coord, rank, failures, log)
            _time_early_trigger(coord, rank, log)
        torch.cuda.synchronize()
    except BaseException:
        # No collective teardown: the other ranks may be waiting in a collective
        # for this one. Exit now, so the parent stops them at once.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    import torch.distributed as dist

    comm.close()
    dist.destroy_process_group()
    assert not failures, f"rank {rank}:\n" + "\n".join(failures)


def _spawn(worker, what: str) -> None:
    import torch.multiprocessing as mp

    # The ranks meet in a file store of their own: a free TCP port picked here is
    # released before rank 0 binds it, and another socket on the node can take it.
    with tempfile.TemporaryDirectory() as rendezvous:
        context = mp.start_processes(
            worker,
            args=(f"file://{rendezvous}/store",),
            nprocs=WORLD_SIZE,
            join=False,
            start_method="spawn",
        )
        deadline = time.monotonic() + TIMEOUT_S
        while not context.join(timeout=10):
            if time.monotonic() > deadline:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                pytest.fail(f"{what} workers did not finish in {TIMEOUT_S} s")


@pytest.mark.skipif(
    not _supported(), reason=f"needs {WORLD_SIZE} SM100-family GPUs (B200/B300/GB300)"
)
def test_early_trigger_feeds_the_low_latency_gemm():
    why = _ll_skip_reason()
    if why is not None:
        pytest.skip(why)
    _spawn(_early_worker, "hop_mxfp8 entry trigger")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
