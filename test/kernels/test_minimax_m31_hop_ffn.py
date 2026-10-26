# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot hop_ffn of the MiniMax M3.1 megakernel: SGLang's push-plane all-reduce +
residual + Gemma norm kernel at the FFN input (the plain all-reduce of the attention's
partial sum) in its quantizing forms, at TP4 on the CustomAllReduceV2 push plane as
M3.1 registers it (4 MiB push slots, 512 row counters), at M3.1's hidden width 6144.

1 GPU, the NVFP4 epilogue on rows from memory (``nvfp4_row_quant`` at every launch
geometry, in both subnormal modes) against FlashInfer's ``nvfp4_quantize`` as SGLang's
routed MoE calls it (per-token, linear scales, cute-dsl): on the build's probe rows,
random rows at magnitudes 1e-30..1e30, and rows with NaN, Inf and -0. The mode the
build picks (``hop_ffn.probe_nvfp4``) must exist and match every byte of every row;
the test prints which mode FlashInfer's kernel has.

4 GPUs (one process each):
- at every width the slot serves (1-16 and 8-128 rows), against today's kernel
  followed by FlashInfer's MXFP8 quantize (cute-dsl, 128x4, as the gate_up quantizes
  its bf16 input) and FlashInfer's per-token NVFP4 quantize of the same bf16 rows:
  residual_out, the bf16 normed rows, the e4m3 codes, every MXFP8 scale byte, the
  packed E2M1 codes, the E4M3 block scales and the per-token scales, bit for bit, with
  the outputs pre-filled with garbage, the quantized outputs alike on every rank; the
  MXFP8-only form (dense layers) likewise; both again at forced launch geometries (two
  vectors per thread);
- in a decode step's launch order: the NVFP4 form from persistent buffers followed
  directly by the shared expert's gate_up GEMM on its pair (the boundary kernel is the
  GEMM's PDL primary there), captured in one CUDA graph and replayed on new inputs,
  against today's kernel, the two quantizes and the same GEMM on the bf16 rows;
- the dense MLP's and the shared expert's gate_up GEMMs on the pair against the same
  GEMMs on the bf16 rows;
- rank 0 prints (nothing is checked) the time per launch of the NVFP4 form against
  today's kernel plus the two quantize launches it replaces, from CUDA graphs.

Every check runs the epilogue in the subnormal mode the slot's build probe picks on
this GPU (``hop_ffn.probe_nvfp4``), at the global scale the slot launches it with
(``moe_nofill.e4m3_max``).

Run (the 4-GPU test spawns one process per GPU):
    python3 -m pytest -q -s test/kernels/test_minimax_m31_hop_ffn.py
"""

import os
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
SF = H // 16
EPS = 1e-6
N_SHARED = 1536  # one TP4 rank's shared-expert gate_up (2 x 3072 / 4)
N_DENSE = 6144  # one TP4 rank's dense-MLP gate_up (2 x 12288 / 4)
# the widths the slot serves: no-spec decode steps of 1-16 rows, DSpark verify steps
# of 8-128 rows (8 a request)
ROWS = sorted(set(range(1, 17)) | set(range(8, 129, 8)))
# (rows, two_shot, vecs_per_thread, cluster_size) the default policy never picks at
# these widths: two vectors per thread
FORCED = ((8, False, 2, 3), (64, True, 2, 3), (128, True, 2, 3))
# the epilogue's (cluster_size, vecs_per_thread): the default geometries (one-shot
# below 48 rows, two-shot from 48) and every forced one
ROW_GEOMETRIES = ((6, 1), (4, 1), (3, 2), (6, 2), (4, 2))
GRAPH_ROWS = (8, 64, 128)
GEMM_ROWS = (1, 8, 16, 48, 128)
TIME_ROWS = (8, 16, 48, 64, 128)
TIME_LAUNCHES = 20  # launches per captured graph
TIME_REPLAYS = 10
TIMEOUT_S = (
    2400  # cold JIT builds of both kernel families and FlashInfer's CuTe kernels
)


def _sm100(devices: int) -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= devices
        and torch.cuda.get_device_capability()[0] == 10
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1).view(torch.uint8)


def _differ(name, got, want):
    """'' when the bytes are equal, else what differs."""
    a, b = _bits(got), _bits(want)
    if a.numel() != b.numel():
        return f"{name}: {a.numel()} vs {b.numel()} bytes"
    bad = (a != b).nonzero()
    if bad.numel() == 0:
        return ""
    return f"{name}: {bad.shape[0]} of {a.numel()} bytes differ, first at byte {bad[0].item()}"


def _global_scale_inv(dev):
    """The stock chain's torch.full of 1 / (e4m3_max * 6)."""
    from sgl_nvidia_mega.minimax_m31.moe import moe_nofill

    return torch.full(
        (1,), 1.0 / (moe_nofill.e4m3_max() * 6.0), dtype=torch.float32, device=dev
    )


def _flashinfer_nvfp4(x, gsi):
    """FlashInfer's per-token NVFP4 quantize as SGLang's routed MoE calls it:
    ``(codes [m, H / 2], block scales [m, H / 16], per-token scales [m])``."""
    from flashinfer import SfLayout, nvfp4_quantize

    fp4, sf, pts = nvfp4_quantize(
        x,
        gsi,
        sfLayout=SfLayout.layout_linear,
        per_token_activation=True,
        backend="cute-dsl",
    )
    m = x.shape[0]
    return fp4.reshape(m, -1), sf.reshape(m, -1), pts


# ------------------------------------------------------------ 1 GPU: the epilogue


def _special_rows(dev):
    """Rows with NaN, Inf and -0: a few NaN values, a 16-value block all NaN, an
    all-NaN row, +-Inf values, -0 throughout, an all-zero row."""
    g = torch.Generator(device="cpu").manual_seed(91)
    rows = torch.randn(6, H, generator=g)
    rows[0, torch.randint(0, H, (40,), generator=g)] = float("nan")
    rows[1, 160:176] = float("nan")
    rows[2] = float("nan")
    rows[3, torch.randint(0, H, (5,), generator=g)] = float("inf")
    rows[3, 7] = -float("inf")
    rows[4] = -0.0
    rows[5] = 0.0
    return rows.to(torch.bfloat16).to(dev).contiguous()


def _random_rows(dev, m, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(m, H, generator=g)
    x *= torch.logspace(-30, 30, m)[:, None] if m > 1 else 1.0
    return x.to(torch.bfloat16).to(dev).contiguous()


def _check_row_quant(failures, log):
    """The epilogue on rows from memory against FlashInfer, every geometry and mode;
    the build's mode must match every row."""
    from sglang.kernels.ops.communication.all_reduce_fusion import nvfp4_row_quant

    from sgl_nvidia_mega.minimax_m31.boundary import hop_ffn

    dev = torch.device("cuda")
    gsi = _global_scale_inv(dev)
    ftz, why = hop_ffn.probe_nvfp4(((6, 1), (4, 1)), hidden=H, device=dev)
    if ftz is None:
        failures.append(f"probe_nvfp4 found no bit-for-bit mode: {why}")
    log(
        "FlashInfer's per-token NVFP4 kernel "
        + (
            f"matches the epilogue with {hop_ffn._mode_name(ftz)}"
            if ftz is not None
            else f"matches neither mode: {why}"
        )
    )
    sets = {
        "probe rows": hop_ffn.probe_rows(H, dev),
        "NaN / Inf / -0 rows": _special_rows(dev),
        **{
            f"{m} random rows": _random_rows(dev, m, seed=m)
            for m in (1, 7, 64, 128, 1000)
        },
    }
    names = ("codes", "block scales", "per-token scales")
    for label, x in sets.items():
        want = _flashinfer_nvfp4(x, gsi)
        matched = {}
        for mode in (False, True):
            for cluster, vecs in ROW_GEOMETRIES:
                got = nvfp4_row_quant(
                    x, gsi, ftz=mode, cluster_size=cluster, vecs_per_thread=vecs
                )
                diffs = [d for d in map(_differ, names, got, want) if d]
                matched.setdefault(mode, []).append(not diffs)
                if mode == ftz and diffs:
                    failures.append(
                        f"{label}, cluster {cluster} x {vecs}: " + "; ".join(diffs)
                    )
        log(
            f"{label}: bitwise with subnormals kept at "
            f"{sum(matched[False])}/{len(ROW_GEOMETRIES)} geometries, flushed at "
            f"{sum(matched[True])}/{len(ROW_GEOMETRIES)}"
        )


@pytest.mark.skipif(not _sm100(1), reason="needs an SM100-family GPU (B200/B300/GB300)")
def test_nvfp4_epilogue_matches_flashinfer_per_token_quantize():
    failures = []
    with torch.inference_mode():
        _check_row_quant(failures, lambda message: print(f"[hop_ffn] {message}"))
    assert not failures, "\n".join(failures)


# ------------------------------------------------------------ 4 GPUs: the kernel


def _init(rank: int, init_method: str):
    """The world group and a push-only CustomAllReduceV2 plane registered for the
    fused kernels, as test_minimax_m31_hop_mxfp8.py and M3.1 set them up."""
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


def _inputs(rows: int, rank: int, seed: int):
    """One FFN-input boundary's inputs: the residual stream and the norm weight alike
    on every rank, the attention's partial sum each rank's own. The norm weight's
    32-column groups span 2^-12..2^11 (so do the scales of both formats), one
    32-column group is zero on every rank; from 3 rows on, row 1 is zero (a zero amax)
    and row 2 tiny (1e-30)."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    mine = torch.Generator(device=dev).manual_seed(seed * 16 + rank + 1)
    residual = torch.randn(rows, H, generator=g, device=dev)
    spread = torch.exp2(torch.arange(H // 32, device=dev) % 24 - 12.0)
    gamma = (1 + 0.1 * torch.randn(H, generator=g, device=dev)) * (
        spread.repeat_interleave(32)
    )
    partial_sum = torch.randn(rows, H, generator=mine, device=dev) * (rank + 1) / 4
    zero = slice(64, 96)
    residual[:, zero] = 0
    partial_sum[:, zero] = 0
    if rows >= 3:
        residual[1], partial_sum[1] = 0, 0
        residual[2] *= 1e-30
        partial_sum[2] *= 1e-30
    return dict(
        partial=partial_sum.bfloat16(),
        residual=residual.bfloat16(),
        gamma=gamma.bfloat16(),
    )


def _launch(geometry):
    return dict(zip(("two_shot", "vecs_per_thread", "cluster_size"), geometry or ()))


def _stock(inp, gsi, geometry=None, nvfp4=True):
    """Today's kernel (as CustomAllReduceFusion._ffn_input_add_norm launches it), then
    FlashInfer's MXFP8 quantize of its normed rows (the gate_up's) and, for a MoE
    layer, its per-token NVFP4 quantize (the routed experts')."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    normed, residual_out = ops.all_reduce_residual_norm(
        inp["partial"],
        inp["residual"],
        inp["gamma"],
        EPS,
        world_size=WORLD_SIZE,
        residual_out=torch.empty_like(inp["residual"]),
        **_launch(geometry),
    )
    q, sf = flashinfer_mxfp8_quantize(normed, is_sf_swizzled_layout=True, alignment=32)
    out = (normed, residual_out, q, sf)
    return out + _flashinfer_nvfp4(normed, gsi) if nvfp4 else out


def _buffers(dev):
    """The slot's persistent buffers, sized for 128 rows."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops

    return dict(
        out=torch.empty(128, H, dtype=torch.bfloat16, device=dev),
        quantized=torch.empty(128, H, dtype=torch.float8_e4m3fn, device=dev),
        scales=torch.empty(ops.mxfp8_scale_bytes(H), dtype=torch.uint8, device=dev),
        fp4=torch.empty(128, H // 2, dtype=torch.uint8, device=dev),
        fp4_scales=torch.empty(128 * SF, dtype=torch.uint8, device=dev),
        token_scales=torch.empty(128, dtype=torch.float32, device=dev),
    )


def _poison(buffers):
    """Garbage in every output byte, so a byte the kernel does not write shows."""
    for name, t in buffers.items():
        _bits(t).fill_(0xA5 if name != "out" else 0xFF)


def _slot(inp, gsi, buffers, ftz, geometry=None, residual_out=None, nvfp4=True):
    """The slot's launch: the NVFP4 form (MoE layers) or the MXFP8 form (dense
    layers), into ``buffers``."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops

    rows = inp["residual"].shape[0]
    outputs = dict(
        residual_out=(
            torch.empty_like(inp["residual"]) if residual_out is None else residual_out
        ),
        out=buffers["out"][:rows],
        quantized=buffers["quantized"][:rows],
        scales=buffers["scales"],
        **_launch(geometry),
    )
    args = (inp["partial"], inp["residual"], inp["gamma"], EPS)
    if not nvfp4:
        return ops.all_reduce_residual_norm_quant(
            *args, world_size=WORLD_SIZE, **outputs
        )
    out = ops.all_reduce_residual_norm_quant_nvfp4(
        *args,
        gsi,
        world_size=WORLD_SIZE,
        ftz=ftz,
        fp4=buffers["fp4"][:rows],
        fp4_scales=buffers["fp4_scales"][: rows * SF],
        token_scales=buffers["token_scales"][:rows],
        **outputs,
    )
    return out[:5] + (out[5].view(rows, SF), out[6])


NAMES = (
    "normed",
    "residual_out",
    "e4m3 codes",
    "MXFP8 scales",
    "e2m1 codes",
    "NVFP4 block scales",
    "per-token scales",
)


def _compare(tag, got, want, failures):
    diffs = [d for d in map(_differ, NAMES, got, want) if d]
    if diffs:
        failures.append(f"{tag}: " + "; ".join(diffs))
    return not diffs


def _same_on_every_rank(tag, tensors, coord, failures):
    import torch.distributed as dist

    for name, t in tensors:
        bits = _bits(t)
        gathered = [torch.empty_like(bits) for _ in range(coord.world_size)]
        dist.all_gather(gathered, bits, group=coord.device_group)
        for rank, other in enumerate(gathered):
            if not torch.equal(other, bits):
                failures.append(f"{tag}: {name} differs on rank {rank}")


def _check_widths(coord, rank, ftz, failures, log):
    dev = torch.device("cuda")
    gsi = _global_scale_inv(dev)
    buffers = _buffers(dev)
    for nvfp4 in (True, False):
        variant = "MoE (MXFP8 + NVFP4)" if nvfp4 else "dense (MXFP8)"
        bitwise = []
        for rows in ROWS:
            inp = _inputs(rows, rank, seed=rows)
            want = _stock(inp, gsi, nvfp4=nvfp4)
            _poison(buffers)
            got = _slot(inp, gsi, buffers, ftz, nvfp4=nvfp4)
            torch.cuda.synchronize()
            tag = f"{variant} rows={rows}"
            if _compare(tag, got, want, failures):
                bitwise.append(rows)
            quantized = [("e4m3 codes", got[2]), ("MXFP8 scales", got[3])]
            if nvfp4:
                quantized += [(n, t) for n, t in zip(NAMES[4:], got[4:])]
            _same_on_every_rank(tag, quantized, coord, failures)
        log(f"{variant}: bitwise (every output) at rows {bitwise}")
        for rows, *geometry in FORCED:
            inp = _inputs(rows, rank, seed=1000 + rows)
            want = _stock(inp, gsi, geometry, nvfp4=nvfp4)
            _poison(buffers)
            got = _slot(inp, gsi, buffers, ftz, geometry, nvfp4=nvfp4)
            torch.cuda.synchronize()
            tag = f"{variant} rows={rows} two_shot/vecs/cluster={geometry}"
            ok = _compare(tag, got, want, failures)
            log(f"{tag}: {'bitwise' if ok else 'DIFFERS'}")


def _weight(n, dev, seed):
    """One TP4 rank's gate_up weight [n, H] in MXFP8, its scales in the 128x4 layout
    the layer keeps (weight_scale_inv_swizzled)."""
    from flashinfer import mxfp8_quantize

    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(n, H, generator=g, device=dev) / H**0.5
    weight, weight_sf = mxfp8_quantize(w.bfloat16(), is_sf_swizzled_layout=True)
    return weight, weight_sf.reshape(-1)


def _gemm(x, weight, weight_sf, scales=None):
    """The gate_up GEMM as Fp8LinearMethod runs it (cute-dsl): on the MXFP8 pair
    ``(x, scales)``, or on bf16 rows ``x`` that it quantizes first."""
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )

    return flashinfer_mxfp8_blockscaled_linear(
        x, weight, weight_sf, input_scale=scales, backend="cute-dsl"
    )


def _check_graph_replay(coord, rank, ftz, failures, log):
    """The NVFP4 form from its persistent buffers and the shared expert's gate_up on
    its pair, captured in one CUDA graph as a decode step's graph holds them (the
    boundary kernel the GEMM's PDL primary) and replayed on new inputs copied into the
    captured ones; against today's kernel, the two quantizes and the same GEMM on the
    bf16 rows."""
    dev = torch.device("cuda")
    gsi = _global_scale_inv(dev)
    buffers = _buffers(dev)
    weight, weight_sf = _weight(N_SHARED, dev, 4242)
    for rows in GRAPH_ROWS:
        static = _inputs(rows, rank, seed=2000 + rows)
        residual_out = torch.empty_like(static["residual"])

        def step():
            out = _slot(static, gsi, buffers, ftz, residual_out=residual_out)
            return out, _gemm(out[2], weight, weight_sf, scales=out[3])

        step()  # warm up: the JIT builds and the GEMM's compiled kernel
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_gemm = step()
        fresh = _inputs(rows, rank, seed=3000 + rows)
        want = _stock(fresh, gsi)
        want_gemm = _gemm(want[0], weight, weight_sf)
        for key, value in fresh.items():
            static[key].copy_(value)
        _poison(buffers)
        residual_out.fill_(float("nan"))
        captured_gemm.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        tag = f"rows={rows} graph replay"
        ok = _compare(tag, captured, want, failures)
        diff = _differ(f"{tag} shared gate_up GEMM", captured_gemm, want_gemm)
        if diff:
            failures.append(diff)
        log(
            f"{tag}: boundary kernel {'bitwise' if ok else 'DIFFERS'}, gate_up GEMM on "
            f"its pair {'bitwise' if not diff else 'DIFFERS'}"
        )


def _check_gemm(coord, rank, ftz, failures, log):
    """The dense MLP's and the shared expert's gate_up (Fp8LinearMethod's MXFP8 path)
    in a decode step's launch order: with the slot, the boundary kernel then the GEMM
    on its pair, nothing between them; stock, today's kernel then the same GEMM on its
    bf16 normed rows, which quantizes them first."""
    dev = torch.device("cuda")
    gsi = _global_scale_inv(dev)
    buffers = _buffers(dev)
    for name, n, nvfp4 in (
        ("shared expert", N_SHARED, True),
        ("dense MLP", N_DENSE, False),
    ):
        weight, weight_sf = _weight(n, dev, 77 + n)
        for rows in GEMM_ROWS:
            # build the GEMM for these rows first, so that below the GEMM launches
            # straight after the boundary kernel
            _gemm(
                torch.zeros(rows, H, dtype=torch.bfloat16, device=dev),
                weight,
                weight_sf,
            )
            inp = _inputs(rows, rank, seed=5000 + rows)
            _poison(buffers)
            torch.cuda.synchronize()
            out = _slot(inp, gsi, buffers, ftz, nvfp4=nvfp4)
            got = _gemm(out[2], weight, weight_sf, scales=out[3])
            want = _gemm(_stock(inp, gsi, nvfp4=False)[0], weight, weight_sf)
            torch.cuda.synchronize()
            diff = _differ(f"{name} rows={rows} gate_up GEMM", got, want)
            if diff:
                failures.append(diff)
            log(
                f"{name} rows={rows} gate_up GEMM on the pair: {'bitwise' if not diff else 'DIFFERS'}"
            )


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


def _time_kernels(coord, rank, ftz, log):
    """Prints, checks nothing: the NVFP4 form's time per launch against today's
    kernel plus the launches it replaces (the shared expert's MXFP8 quantize and the
    routed experts' per-token NVFP4 quantize, as slot moe_nofill launches it)."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    from sgl_nvidia_mega.minimax_m31.moe import moe_nofill

    dev = torch.device("cuda")
    gsi = _global_scale_inv(dev)
    buffers = _buffers(dev)
    quant = moe_nofill.PerTokenQuant(max_rows=128, hidden=H, device=dev)
    for rows in TIME_ROWS:
        inp = _inputs(rows, rank, seed=6000 + rows)
        residual_out = torch.empty_like(inp["residual"])

        def stock():
            normed, _ = ops.all_reduce_residual_norm(
                inp["partial"],
                inp["residual"],
                inp["gamma"],
                EPS,
                world_size=WORLD_SIZE,
                residual_out=residual_out,
            )
            flashinfer_mxfp8_quantize(normed, is_sf_swizzled_layout=True, alignment=32)
            quant(normed)

        stock_us = _graph_time_us(stock, coord)
        slot_us = _graph_time_us(
            partial(_slot, inp, gsi, buffers, ftz, residual_out=residual_out), coord
        )
        log(
            f"rows={rows}: today's kernel + two quantizes {stock_us:.2f} us, NVFP4 "
            f"form {slot_us:.2f} us ({stock_us - slot_us:+.2f} us saved per MoE layer)"
        )


def _worker(rank: int, init_method: str) -> None:
    coord, comm = _init(rank, init_method)

    def log(message):
        if rank == 0:
            print(f"[hop_ffn] {message}", flush=True)

    failures = []
    try:
        from sgl_nvidia_mega.minimax_m31.boundary import hop_ffn

        with torch.inference_mode():
            ftz, why = hop_ffn.probe_nvfp4(((6, 1), (4, 1)), hidden=H)
            if ftz is None:
                failures.append(f"probe_nvfp4 found no bit-for-bit mode: {why}")
                ftz = False
            log(f"epilogue mode: {hop_ffn._mode_name(ftz)}")
            _check_widths(coord, rank, ftz, failures, log)
            _check_graph_replay(coord, rank, ftz, failures, log)
            _check_gemm(coord, rank, ftz, failures, log)
            _time_kernels(coord, rank, ftz, log)
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


@pytest.mark.skipif(
    not _sm100(WORLD_SIZE),
    reason=f"needs {WORLD_SIZE} SM100-family GPUs (B200/B300/GB300)",
)
def test_hop_ffn_matches_stock_kernel_and_flashinfer_quantizes():
    import torch.multiprocessing as mp

    # The ranks meet in a file store of their own: a free TCP port picked here is
    # released before rank 0 binds it, and another socket on the node can take it.
    with tempfile.TemporaryDirectory() as rendezvous:
        context = mp.start_processes(
            _worker,
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
                pytest.fail(f"hop_ffn workers did not finish in {TIMEOUT_S} s")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
