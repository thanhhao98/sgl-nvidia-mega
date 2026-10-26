# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of the M3.1 megakernel slot attn_ll (the attention's q|k|v|index MXFP8
GEMM, and o_proj's under the test knob, on FlashInfer's low-latency GEMM at 1-8
rows): the pinned tactics and the K tiles they keep in flight before the PDL wait,
the declared tolerances, the GEMMs knob, the build's per-layer preconditions,
per-width plan, build-time launch and cross-rank agreement, the step's fallbacks to
the stock module, a call trace of SGLang's real MiniMaxM3Attention / Fp8LinearMethod /
RowParallelLinear forward against the slot's with only the CUDA kernels faked, the
decision that lets slot hop_mxfp8's boundary kernel signal the q|k|v|index GEMM at its
entry, and the PDL order the kernels around the slot's GEMMs rely on."""

import contextlib
import inspect
import os
import unittest
from collections import Counter
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.nn import Parameter

pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

from sglang.srt.layers.linear import RowParallelLinear  # noqa: E402
from sglang.srt.layers.quantization import fp8_utils  # noqa: E402
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod  # noqa: E402
from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend  # noqa: E402
from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput  # noqa: E402

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind  # noqa: E402
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import NumericsKind  # noqa: E402

from sglang.srt.models.minimax_m3 import (  # noqa: E402
    MiniMaxM3Attention,
    _FusedQKVIndexProj,
)
from sglang.test.test_utils import CustomTestCase  # noqa: E402

from sgl_nvidia_mega.minimax_m31 import attention as m31_attention  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import boundary as m31_boundary  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import plan as m31_plan  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import runtime as m31_runtime  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import spec as m31_spec  # noqa: E402
from sgl_nvidia_mega.minimax_m31.runtime import agree as m31_agree  # noqa: E402
from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll  # noqa: E402

QKV, O_PROJ = mxfp8_ll.QKV_INDEX, mxfp8_ll.O_PROJ
HIDDEN, ATTN_OUT = QKV[1], O_PROJ[1]
E4M3 = torch.float8_e4m3fn
QKV_ONLY = ("qkv_index",)  # the slot's default GEMMs
BOTH = ("qkv_index", "o_proj")  # with o_proj, under the test knob


# --------------------------------------------------------------- fakes of the layer


def _mxfp8_method(backend=Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL):
    """SGLang's Fp8LinearMethod as an MXFP8 checkpoint on Blackwell resolves it."""
    qm = Fp8LinearMethod.__new__(Fp8LinearMethod)
    qm.use_marlin = False
    qm.use_mxfp8 = True
    qm.convert_mxfp8_to_block = False
    qm.block_fp8_as_mxfp8 = False
    qm.mxfp8_dense_backend = backend
    qm.w8a8_mxfp8_linear = partial(
        fp8_utils.flashinfer_mxfp8_blockscaled_linear, backend="cute-dsl"
    )
    qm.mxfp8_prefill_autotune_min_tokens = None
    return qm


_WEIGHTS: dict = {}


def _weights(n, k):
    """An MXFP8 weight [N, K] and its 128x4 scale bytes (shared by the fake layers)."""
    if (n, k) not in _WEIGHTS:
        gen = torch.Generator().manual_seed(n + k)
        w = (torch.randn(n, k, generator=gen) * 0.5).to(E4M3)
        s = torch.randint(
            120, 128, (mxfp8_ll.scale_bytes(n, k),), dtype=torch.uint8, generator=gen
        )
        _WEIGHTS[n, k] = (w, s)
    return _WEIGHTS[n, k]


def _set_weights(module, shape):
    w, s = _weights(*shape)
    module.weight = Parameter(w, requires_grad=False)
    module.weight_scale_inv_swizzled = Parameter(s, requires_grad=False)


def _fused_proj():
    p = _FusedQKVIndexProj.__new__(_FusedQKVIndexProj)
    torch.nn.Module.__init__(p)
    p._qm = _mxfp8_method()
    p.input_scale = None
    _set_weights(p, QKV)
    return p


def _o_proj():
    o = RowParallelLinear.__new__(RowParallelLinear)
    torch.nn.Module.__init__(o)
    o.quant_method = _mxfp8_method()
    o.tp_group = SimpleNamespace(rank_in_group=0, world_size=4)
    o.parallel_group = "attn_tp"
    o.input_is_parallel = True
    o.reduce_results = False
    o.skip_bias_add = False
    o.use_dp_attention_reduce = False
    o.use_decode_attn_tp = False
    o.register_parameter("bias", None)
    _set_weights(o, O_PROJ)
    return o


def _desc(t):
    """A tensor's layout and bytes (or the value itself)."""
    if not isinstance(t, torch.Tensor):
        return t
    raw = t.detach().contiguous().view(-1).view(torch.uint8).clone()
    return ("tensor", tuple(t.shape), tuple(t.stride()), t.dtype, raw)


def _layout(t):
    """A tensor's layout only (an output buffer's bytes are not yet written)."""
    return (tuple(t.shape), tuple(t.stride()), t.dtype)


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.shape == b.shape
            and torch.equal(a, b)
        )
    if isinstance(a, (tuple, list)):
        return (
            isinstance(b, (tuple, list))
            and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b))
        )
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


class _Trace:
    def __init__(self):
        self.calls = []

    def add(self, name, *payload):
        self.calls.append((name, payload))

    def named(self, name):
        return [payload for n, payload in self.calls if n == name]

    def names(self):
        return [n for n, _ in self.calls]


class _FakeRadixAttention(torch.nn.Module):
    """The MSA backend's call: records its arguments, returns [T, 2048] BF16."""

    def __init__(self, trace):
        super().__init__()
        self.trace = trace
        self.out_dtype = torch.bfloat16

    def forward(self, q, k, v, forward_batch, idx_q=None, idx_k=None, idx_v=None):
        self.trace.add(
            "attn",
            *(_desc(t) for t in (q, k, v)),
            id(forward_batch),
            _desc(idx_q),
            _desc(idx_k),
            idx_v,
        )
        rows = q.shape[0]
        out = q.float().reshape(rows, -1) * 0.5 + v.float().reshape(rows, 1, -1).expand(
            rows, 16, -1
        ).reshape(rows, -1)
        out = out + k.float().reshape(rows, -1).sum(-1, keepdim=True) * 1e-3
        out = out + idx_k.float().reshape(rows, -1).mean(-1, keepdim=True)
        return None, out.to(self.out_dtype).contiguous()


class _FakeAttention(MiniMaxM3Attention):
    """A MiniMaxM3Attention of the M3.1 TP4 rank (the real class and forward), with
    fake submodules where kernels run."""

    def __init__(self, trace):
        torch.nn.Module.__init__(self)
        self.is_sparse_attention_layer = True
        self.disable_index_value = True
        self.index_o_proj = None
        self.num_heads, self.num_kv_heads, self.head_dim = 16, 1, 128
        self.num_idx_heads, self.idx_head_dim, self.idx_replica_size = 1, 128, 1
        self.q_size, self.kv_size = 2048, 128
        self._fused_main_size = self.q_size + 2 * self.kv_size
        self._can_use_aiter_fused_qknorm_static = False
        self._combined_qknorm_ok = True
        self._fp8_q_epilogue_ok = True
        self._qknorm_group_meta = ((0, 16), (16, 1), (18, 1), (19, 1))
        gen = torch.Generator().manual_seed(3)
        for name in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
            setattr(
                self,
                name,
                SimpleNamespace(
                    weight=torch.randn(128, generator=gen), variance_epsilon=1e-6
                ),
            )
        self.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.randn(4096, 64, generator=gen)
        )
        self.add_module("fused_qkv_index_proj", _fused_proj())
        self._fused_qkv_index = self.fused_qkv_index_proj
        self.o_proj = _o_proj()
        self.attn = _FakeRadixAttention(trace)


def _layer(trace):
    return SimpleNamespace(self_attn=_FakeAttention(trace))


def _model(trace, num_layers=3):
    layers = [_layer(trace) for _ in range(num_layers)]
    return SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=num_layers, layers=layers)
    )


# ------------------------------------------------------------- fakes of the kernels


def _fake_mm(a, w):
    return (a.float() @ w.float().t()).to(torch.bfloat16)


def _fake_quantize(trace):
    def quantize(input, is_sf_swizzled_layout=True, alignment=32, backend="cute-dsl"):
        trace.add(
            "quantize",
            _desc(input),
            dict(is_sf_swizzled_layout=is_sf_swizzled_layout, alignment=alignment),
            backend,
        )
        rows, k = input.shape
        q = input.to(E4M3)
        s = (torch.arange(mxfp8_ll.scale_bytes(rows, k)) % 7 + 124).to(torch.uint8)
        return q, s

    return quantize


def _fake_mm_mxfp8(trace):
    """FlashInfer mm_mxfp8 as SGLang's flashinfer_mm_mxfp8 op calls it."""

    def mm(q, weight_t, x_scale, weight_scale_t, out_dtype, use_8x4_sf_layout, backend):
        trace.add(
            "gemm",
            _desc(q),
            _desc(x_scale),
            id(weight_t._base if weight_t._base is not None else weight_t),
            id(weight_scale_t),
            dict(out_dtype=out_dtype, use_8x4_sf_layout=use_8x4_sf_layout),
            backend,
        )
        return _fake_mm(q, weight_t.t())

    return mm


def _fake_ll_linear(trace):
    def linear(a, a_scale, weight, weight_scale, *, tactic, out=None):
        trace.add(
            "ll_gemm",
            _desc(a),
            _desc(a_scale),
            id(weight),
            id(weight_scale),
            tactic,
        )
        return _fake_mm(a, weight)

    return linear


def _fake_qknorm(trace):
    def qknorm(
        qkv, groups, cos_sin_cache, positions, eps, *, fp8_a, group_a, fp8_b, group_b
    ):
        trace.add(
            "qknorm",
            _desc(qkv),
            [(id(w), off, cnt) for w, off, cnt in groups],
            id(cos_sin_cache),
            id(positions),
            eps,
            _layout(fp8_a),
            group_a,
            _layout(fp8_b),
            group_b,
        )
        qkv.mul_(0.5).add_(0.25)
        fp8_a.copy_(qkv[:, :2048].reshape(fp8_a.shape).to(E4M3))
        fp8_b.copy_(qkv[:, 2304:2432].reshape(fp8_b.shape).to(E4M3))
        return qkv

    return qknorm


_TP_GROUP = SimpleNamespace(world_size=4, rank_in_group=0)
_PARALLEL = SimpleNamespace(tp_group=_TP_GROUP, attn_tp_group=_TP_GROUP)


@contextlib.contextmanager
def _kernels_faked(trace, *, sp_active=False):
    """SGLang's CUDA kernels, collectives and runtime groups faked alike for both
    forwards."""
    from sglang.kernels.ops.attention import minimax_qknorm_rope
    from sglang.srt import runtime_context
    from sglang.srt.distributed.device_communicators import pynccl_allocator
    from sglang.srt.layers import layernorm_sp, linear

    def use_symmetric_memory(group, disabled=False):
        trace.add("allocation", id(group), disabled)
        return contextlib.nullcontext()

    def gbar_matmul(layer, x, bias):
        trace.add("reduce_scatter")
        return layer.quant_method.apply(layer, x, bias=bias)

    def all_reduce(x):
        trace.add("all_reduce")
        return x

    forward = SimpleNamespace(sp_active=sp_active)
    with contextlib.ExitStack() as stack:
        for target, name, value in (
            (fp8_utils, "flashinfer_mxfp8_quantize", _fake_quantize(trace)),
            (fp8_utils, "flashinfer_mm_mxfp8", _fake_mm_mxfp8(trace)),
            (mxfp8_ll, "linear", _fake_ll_linear(trace)),
            (
                minimax_qknorm_rope,
                "minimax_qknorm_rope_grouped_fp8",
                _fake_qknorm(trace),
            ),
            (linear, "get_parallel", lambda: _PARALLEL),
            (runtime_context, "get_parallel", lambda: _PARALLEL),
            (linear, "get_forward", lambda: forward),
            (runtime_context, "get_forward", lambda: forward),
            (linear, "use_symmetric_memory", use_symmetric_memory),
            (pynccl_allocator, "use_symmetric_memory", use_symmetric_memory),
            (layernorm_sp, "row_parallel_gbar_matmul", gbar_matmul),
            (linear, "tensor_model_parallel_all_reduce", all_reduce),
        ):
            stack.enter_context(patch.object(target, name, value, create=True))
        yield


def _bound(layers, rows=(1, 2, 3, 4, 5, 6, 7, 8), sm=103, gemms=QKV_ONLY):
    """The build result of attn_ll binding these layers with ``gemms`` on the
    low-latency kernel (fake tactics lookup)."""

    def linear(layer, name):
        module = (
            layer.self_attn.fused_qkv_index_proj
            if name == "qkv_index"
            else layer.self_attn.o_proj
        )
        shape = m31_attention.SHAPES[name]
        return m31_attention.LowLatencyLinear(
            module, {w: mxfp8_ll.pinned_tactic(sm, *shape, w) for w in rows}
        )

    bound = {
        i: m31_attention.LayerGemms(**{name: linear(layer, name) for name in gemms})
        for i, layer in enumerate(layers)
    }
    return m31_attention.AttnLowLatency(
        layers=bound,
        widths=tuple(rows),
        tactics={},
        reasons=[],
        skipped={},
        gemms=gemms,
    )


_FORWARD_BATCH = SimpleNamespace(name="forward_batch")
_POSITIONS: dict[int, torch.Tensor] = {}


def _state(layer_id, rows, built, *, slot=True):
    if rows not in _POSITIONS:
        _POSITIONS[rows] = torch.arange(rows, dtype=torch.long) + 100
    return SimpleNamespace(
        slots=m31_plan.Slots(attn_ll=slot),
        bound={"attention": built},
        layer_id=layer_id,
        rows=rows,
        positions=_POSITIONS[rows],
        fb=_FORWARD_BATCH,
    )


def _inputs(rows, seed=0):
    gen = torch.Generator().manual_seed(seed + rows)
    return (torch.randn(rows, HIDDEN, generator=gen) * 2).to(torch.bfloat16)


# ------------------------------------------------------------------------- tests


class TestTactics(CustomTestCase):
    def test_row_buckets(self):
        self.assertEqual(
            [mxfp8_ll.row_bucket(r) for r in range(1, 9)], [1, 2, 4, 4, 8, 8, 8, 8]
        )

    def test_every_served_row_count_has_a_pinned_tactic_on_b200_b300_gb300(self):
        for sm in (100, 103):
            for n, k in (QKV, O_PROJ):
                for rows in range(1, 9):
                    self.assertIsNotNone(mxfp8_ll.pinned_tactic(sm, n, k, rows))
                for rows in (0, 9, 16):
                    self.assertIsNone(mxfp8_ll.pinned_tactic(sm, n, k, rows))
            self.assertIsNone(mxfp8_ll.pinned_tactic(sm, 4096, 6144, 8))
        self.assertIsNone(mxfp8_ll.pinned_tactic(90, *QKV, 8))

    def test_pinned_tactics_fit_one_wave_and_the_kernel_limits(self):
        for (sm, n, k, bucket), tactic in mxfp8_ll.PINNED_TACTICS.items():
            cta_k, stages, sfb_stages, split_k = tactic
            self.assertIn(bucket, mxfp8_ll.ROW_BUCKETS)
            self.assertIn(split_k, (1, 2, 4, 8))
            self.assertEqual(sfb_stages, 1)  # the direct SFB copy path
            self.assertTrue(1 <= stages <= 12)
            # MX scale atoms are 128-K wide; whole K tiles in every split
            self.assertEqual(cta_k % 128, 0)
            self.assertEqual(k % cta_k, 0)
            self.assertEqual((k // cta_k) % split_k, 0)
            # one CTA per 128 output columns and split: one wave on 148 SMs (the
            # measured sm_103 o_proj pins run 48 CTAs: fewer, longer K slices won)
            self.assertLessEqual(n // 128 * split_k, 148, tactic)
            # FlashInfer's shared-memory estimate (A, B, SFA, SFB, padded SFB per
            # stage) within sm_100's 227 KB
            per_stage = 136 * cta_k + 264 * (cta_k // 32)
            mailbox = (split_k - 1) * 128 * (8 // split_k) * 4
            self.assertLessEqual(stages * per_stage + mailbox + 1024, 232448, tactic)

    def test_k_tiles_in_flight_before_the_pdl_wait(self):
        """A CTA's weight warp fills each free stage before the PDL wait: the pinned
        q|k|v|index tactics keep 8 (1 row) and 10 (2-8 rows) of a CTA's 12 K tiles in
        flight there, (128, 12, 1, 4) and (256, 6, 1, 4) all of them."""
        k = QKV[1]
        for tactic, tiles, before in (
            ((128, 8, 1, 4), 12, 8),
            ((128, 10, 1, 4), 12, 10),
            ((128, 12, 1, 4), 12, 12),
            ((256, 6, 1, 4), 6, 6),
            ((256, 6, 1, 2), 12, 6),
            ((128, 6, 1, 8), 6, 6),
        ):
            self.assertEqual(mxfp8_ll.k_tiles_per_cta(tactic, k), tiles, tactic)
            self.assertEqual(mxfp8_ll.tiles_before_wait(tactic, k), before, tactic)
        # o_proj (K 2048)
        self.assertEqual(mxfp8_ll.tiles_before_wait((128, 11, 1, 1), O_PROJ[1]), 11)
        self.assertEqual(mxfp8_ll.k_tiles_per_cta((128, 11, 1, 1), O_PROJ[1]), 16)
        self.assertEqual(mxfp8_ll.tiles_before_wait((128, 8, 1, 2), O_PROJ[1]), 8)

    def test_all_k_tiles_in_flight_candidates_fit_flashinfers_estimate(self):
        """FlashInfer's own shared-memory estimate (_smem_bytes, transcribed below):
        the served (128, 10, 1, 4) takes the 188,032 B nsys showed, the candidates
        that keep every K tile in flight 224,896 B, within sm_100's 232,448 B."""
        capacity = 232448
        self.assertEqual(_flashinfer_smem_bytes((128, 10, 1, 4)), 188032)
        self.assertEqual(_flashinfer_smem_bytes((128, 8, 1, 4)), 151040)
        for tactic in ((128, 12, 1, 4), (256, 6, 1, 4)):
            self.assertEqual(_flashinfer_smem_bytes(tactic), 224896, tactic)
            self.assertLessEqual(_flashinfer_smem_bytes(tactic), capacity)
            self.assertEqual(
                mxfp8_ll.tiles_before_wait(tactic, QKV[1]),
                mxfp8_ll.k_tiles_per_cta(tactic, QKV[1]),
            )
        # one stage more does not fit
        self.assertGreater(_flashinfer_smem_bytes((128, 13, 1, 4)), capacity)
        self.assertGreater(_flashinfer_smem_bytes((256, 7, 1, 4)), capacity)
        for tactic in mxfp8_ll.PINNED_TACTICS.values():
            self.assertLessEqual(_flashinfer_smem_bytes(tactic), capacity, tactic)

    def test_scale_bytes(self):
        self.assertEqual(mxfp8_ll.scale_bytes(8, 6144), 128 * 192)
        self.assertEqual(mxfp8_ll.scale_bytes(1, 2048), 128 * 64)
        self.assertEqual(mxfp8_ll.scale_bytes(2560, 6144), 2560 * 192)
        self.assertEqual(mxfp8_ll.scale_bytes(6144, 2048), 6144 * 64)

    def test_operand_checks(self):
        w, s = _weights(*QKV)
        self.assertEqual(mxfp8_ll.weight_reasons(w, s, QKV), [])
        self.assertTrue(mxfp8_ll.weight_reasons(w, s, O_PROJ))
        self.assertTrue(mxfp8_ll.weight_reasons(w.to(torch.bfloat16), s, QKV))
        self.assertTrue(mxfp8_ll.weight_reasons(w.t(), s, (QKV[1], QKV[0])))
        self.assertTrue(mxfp8_ll.weight_reasons(w, s[:-4], QKV))
        self.assertTrue(mxfp8_ll.weight_reasons(w, None, QKV))
        self.assertTrue(mxfp8_ll.weight_reasons(None, s, QKV))
        a = torch.zeros(8, HIDDEN, dtype=E4M3)
        a_sf = torch.zeros(mxfp8_ll.scale_bytes(8, HIDDEN), dtype=torch.uint8)
        self.assertEqual(mxfp8_ll.activation_reasons(a, a_sf, HIDDEN), [])
        self.assertEqual(mxfp8_ll.activation_reasons(a[:3], a_sf, HIDDEN), [])
        for bad in (
            (torch.zeros(9, HIDDEN, dtype=E4M3), a_sf),
            (a.to(torch.bfloat16), a_sf),
            (a[:, :2048], a_sf),
            (a, a_sf[:-1]),
            (torch.zeros(8, 2 * HIDDEN, dtype=E4M3)[:, ::2], a_sf),
        ):
            self.assertTrue(mxfp8_ll.activation_reasons(*bad, HIDDEN), bad[0].shape)


def _flashinfer_smem_bytes(tactic) -> int:
    """FlashInfer 0.7.0's ``low_latency_blockscaled_gemm._smem_bytes`` for an MXFP8
    tactic (e4m3 x e4m3, one UE8M0 scale per 32): barriers and the split-K mailbox,
    then the A, B, SFA, SFB and padded SFB stages, each 128-byte aligned."""
    cta_k, stages, sfb_stages, split_k = tactic

    def align(v, a):
        return -(-v // a) * a

    mailbox = max(split_k - 1, 0) * 128 * (8 // split_k)
    cursor = 16 * stages + 16 * sfb_stages + 16 + 8 + 8 + 4
    cursor = align(cursor, 8) + 8 + mailbox * 4
    sf_k = cta_k // 32
    for size in (
        128 * cta_k * stages,
        8 * cta_k * stages,
        128 * sf_k * stages,
        8 * sf_k * stages,
        128 * sf_k * stages,
    ):
        cursor = align(cursor, 128) + size
    return cursor


class TestNumerics(CustomTestCase):
    def test_attn_ll_makes_the_spec_tolerance_with_its_declared_quantities(self):
        self.assertNotIn("attn_ll", m31_runtime.BITWISE_SLOTS)
        self.assertIn("attn_ll", m31_runtime.OPT_IN_SLOTS)
        tol = m31_runtime.tolerances(frozenset({"attn_ll"}))
        gemms = ("attn.qkv_index", "attn.o_proj")
        self.assertEqual(
            set(tol),
            {f"{gemm}.{what}" for gemm in gemms for what in ("rel", "norm_rel")}
            | {
                "attn.fp8_q.moved_fraction",
                "attn.fp8_q.max_e4m3_steps",
                "attn.kv.nvfp4.moved_fraction",
                "attn.kv.scale.moved_fraction",
                "attn.kv.nvfp4.block_top_steps",
                "attn.index.topk_moved_fraction",
            },
        )
        for gemm in gemms:
            # one BF16 rounding flip of the largest output is 2^-7 of it
            self.assertTrue(2**-7 < tol[f"{gemm}.rel"] <= 2e-2)
            # flips on a fraction of the elements; a wrong K block or row: 1e-2 or more
            self.assertTrue(0 < tol[f"{gemm}.norm_rel"] <= 1e-3)
        for key, value in tol.items():
            if key.endswith("moved_fraction"):
                self.assertTrue(0 < value <= 0.1, key)
        # a BF16 difference crosses at most one rounding boundary of a coarser code
        self.assertEqual(tol["attn.fp8_q.max_e4m3_steps"], 1)
        self.assertEqual(tol["attn.kv.nvfp4.block_top_steps"], 1)
        with patch.object(m31_spec, "SLOTS", frozenset({"attn_ll"})):
            numerics = m31_spec._numerics()
        self.assertEqual(numerics.kind, NumericsKind.TOLERANCE)
        self.assertEqual(dict(numerics.tolerances), tol)


class _FakeKernels:
    def __init__(self, sm=103, reasons=(), refuse=(), fail=()):
        self._sm, self._reasons, self._refuse = sm, list(reasons), set(refuse)
        self._fail = set(fail)
        self.compiled, self.asked, self.prepared = [], [], []

    def sm(self):
        return self._sm

    def reasons(self, sm):
        return list(self._reasons)

    def offered(self, n, k, rows):
        self.asked.append((n, k, rows))
        return [
            t
            for (sm, pn, pk, b), t in mxfp8_ll.PINNED_TACTICS.items()
            if (pn, pk) == (n, k) and (n, k, rows) not in self._refuse
        ] + [(128, 8, 1, 1)]

    def prepare(self, tactic, linear, rows):
        self.prepared.append((tactic, linear, rows))
        if tactic in self._fail:
            raise RuntimeError(f"cute.compile failed for {tactic}")
        self.compiled.append(tactic)


class TestBuild(CustomTestCase):
    def test_not_requested(self):
        self.assertIsNone(m31_attention.build(None, (8,), frozenset()))
        self.assertFalse(m31_attention.serves(None, 8))

    def test_verify_steps_serve_one_request_with_qkv_only(self):
        """By default only the q|k|v|index GEMM is on the low-latency kernel: no o_proj
        tactic is asked for, compiled, launched or needed to serve a width."""
        model = _model(_Trace())
        kernels = _FakeKernels()
        qkv8 = mxfp8_ll.PINNED_TACTICS[(103, *QKV, 8)]
        built = m31_attention.build_low_latency(model, tuple(range(8, 129, 8)), kernels)
        attn = model.model.layers[1].self_attn
        self.assertEqual(built.gemms, QKV_ONLY)
        self.assertIs(built.layers[1].qkv_index.linear, attn.fused_qkv_index_proj)
        self.assertIsNone(built.layers[1].o_proj)
        self.assertEqual(built.widths, (8,))
        self.assertEqual(sorted(built.layers), [0, 1, 2])
        self.assertEqual(built.skipped, {})
        self.assertEqual(built.tactics, {"qkv_index": {8: qkv8}})
        self.assertEqual(kernels.compiled, [qkv8])
        self.assertEqual(kernels.asked, [(*QKV, 8)])
        # the tactic launched once at build, on the first bound layer's weights
        attn0 = model.model.layers[0].self_attn
        self.assertEqual(kernels.prepared, [(qkv8, attn0.fused_qkv_index_proj, 8)])
        self.assertTrue(m31_attention.serves(built, 8))
        self.assertFalse(m31_attention.serves(built, 16))
        gemms = built.layers[1]
        self.assertEqual(gemms.qkv_index.tactics, {8: qkv8})
        self.assertEqual((gemms.qkv_index.n, gemms.qkv_index.k), QKV)

    def test_verify_steps_serve_one_request_with_o_proj(self):
        model = _model(_Trace())
        kernels = _FakeKernels()
        qkv8 = mxfp8_ll.PINNED_TACTICS[(103, *QKV, 8)]
        o8 = mxfp8_ll.PINNED_TACTICS[(103, *O_PROJ, 8)]
        built = m31_attention.build_low_latency(
            model, tuple(range(8, 129, 8)), kernels, gemms=BOTH
        )
        attn = model.model.layers[1].self_attn
        self.assertEqual(built.gemms, BOTH)
        self.assertIs(built.layers[1].qkv_index.linear, attn.fused_qkv_index_proj)
        self.assertIs(built.layers[1].o_proj.linear, attn.o_proj)
        self.assertEqual(built.widths, (8,))
        self.assertEqual(sorted(built.layers), [0, 1, 2])
        self.assertEqual(built.tactics, {"qkv_index": {8: qkv8}, "o_proj": {8: o8}})
        self.assertEqual(sorted(kernels.compiled), sorted([o8, qkv8]))
        # each tactic launched once at build, on the first bound layer's weights
        attn0 = model.model.layers[0].self_attn
        self.assertEqual(
            kernels.prepared,
            [
                (qkv8, attn0.fused_qkv_index_proj, 8),
                (o8, attn0.o_proj, 8),
            ],
        )
        gemms = built.layers[1]
        self.assertEqual((gemms.qkv_index.n, gemms.qkv_index.k), QKV)
        self.assertEqual((gemms.o_proj.n, gemms.o_proj.k), O_PROJ)

    def test_decode_steps_serve_one_to_eight_rows_by_bucket(self):
        for gemms, launched in (
            (QKV_ONLY, [((256, 6, 1, 4), 8)]),
            (BOTH, [((256, 6, 1, 4), 8), ((256, 4, 1, 2), 8)]),
        ):
            with self.subTest(gemms=gemms):
                kernels = _FakeKernels(sm=100)
                built = m31_attention.build_low_latency(
                    _model(_Trace(), 1), tuple(range(1, 17)), kernels, gemms=gemms
                )
                self.assertEqual(built.widths, tuple(range(1, 9)))
                self.assertEqual(set(built.tactics), set(gemms))
                for name in gemms:
                    self.assertEqual(set(built.tactics[name]), set(range(1, 9)))
                # one launch per GEMM and tactic, at the most rows the tactic serves
                self.assertEqual(
                    [(t, rows) for t, _, rows in kernels.prepared], launched
                )
                # FlashInfer is asked once per GEMM's shape and row bucket
                shapes = [m31_attention.SHAPES[name] for name in gemms]
                self.assertEqual(
                    sorted(kernels.asked),
                    sorted((n, k, b) for n, k in shapes for b in (1, 2, 4, 8)),
                )

    def test_a_tactic_flashinfer_does_not_offer_leaves_its_bucket_stock(self):
        for refused, gemms, widths in (
            ((*QKV, 4), QKV_ONLY, (1, 2, 5, 6, 7, 8)),
            ((*O_PROJ, 4), BOTH, (1, 2, 5, 6, 7, 8)),
            # o_proj's tactics do not matter while it stays SGLang's
            ((*O_PROJ, 4), QKV_ONLY, tuple(range(1, 9))),
        ):
            with self.subTest(refused=refused, gemms=gemms):
                kernels = _FakeKernels(refuse={refused})
                built = m31_attention.build_low_latency(
                    _model(_Trace(), 1), tuple(range(1, 17)), kernels, gemms=gemms
                )
                self.assertEqual(built.widths, widths)

    def test_a_tactic_that_fails_at_build_leaves_its_widths_stock(self):
        # a tactic of one row bucket fails to compile or launch: that bucket's
        # widths run the stock module, the others the slot
        one_row = (128, 12, 1, 4)
        with patch.dict(mxfp8_ll.PINNED_TACTICS, {(103, *QKV, 1): one_row}):
            kernels = _FakeKernels(fail={one_row})
            built = m31_attention.build_low_latency(
                _model(_Trace(), 1), tuple(range(1, 17)), kernels, gemms=BOTH
            )
        self.assertEqual(built.widths, tuple(range(2, 9)))
        self.assertEqual(sorted(built.layers), [0])
        # each other tactic launched once, at the most served rows it covers
        pinned = mxfp8_ll.PINNED_TACTICS
        o_proj_rows = {}
        for w in range(2, 9):
            t = pinned[(103, *O_PROJ, mxfp8_ll.row_bucket(w))]
            o_proj_rows[t] = max(o_proj_rows.get(t, 0), w)
        qkv_rows = {}
        for w in range(2, 9):
            t = pinned[(103, *QKV, mxfp8_ll.row_bucket(w))]
            qkv_rows[t] = max(qkv_rows.get(t, 0), w)
        self.assertEqual(
            sorted((t, rows) for t, _, rows in kernels.prepared),
            sorted([(one_row, 1), *qkv_rows.items(), *o_proj_rows.items()]),
        )
        # every bucket's o_proj tactic fails: no width with o_proj on the kernel,
        # with the error in the reasons; every width without it
        o_proj_tactics = {
            t for (sm, *nk, _), t in pinned.items() if sm == 103 and tuple(nk) == O_PROJ
        }
        built = m31_attention.build_low_latency(
            _model(_Trace(), 1),
            tuple(range(1, 17)),
            _FakeKernels(fail=o_proj_tactics),
            gemms=BOTH,
        )
        self.assertEqual(built.widths, ())
        self.assertEqual(built.layers, {})
        self.assertIn("failed at build", "; ".join(built.reasons))
        self.assertIn("cute.compile failed", "; ".join(built.reasons))
        built = m31_attention.build_low_latency(
            _model(_Trace(), 1), tuple(range(1, 17)), _FakeKernels(fail=o_proj_tactics)
        )
        self.assertEqual(built.widths, tuple(range(1, 9)))

    def test_ranks_that_build_different_plans_stop(self):
        built = m31_attention.build_low_latency(
            _model(_Trace(), 2), (8, 16), _FakeKernels()
        )
        plan = (built.widths, (0, 1), built.tactics)
        with patch.object(
            m31_agree, "gather_tp", side_effect=lambda x, group=None: [x] * 4
        ) as gather:
            m31_attention._agree(built)
        self.assertEqual(gather.call_args.args[0], plan)
        other = ((), (), {"qkv_index": {}, "o_proj": {}})
        with (
            patch.object(
                m31_agree,
                "gather_tp",
                side_effect=lambda x, group=None: [x, x, other, x],
            ),
            self.assertRaisesRegex(RuntimeError, "slot attn_ll plans"),
        ):
            m31_attention._agree(built)

    def test_an_unsupported_gpu_serves_no_width(self):
        for kernels in (
            _FakeKernels(reasons=["no low-latency GEMM"]),
            _FakeKernels(sm=90),
        ):
            built = m31_attention.build_low_latency(
                _model(_Trace(), 1), (8, 16), kernels
            )
            self.assertEqual(built.widths, ())
            self.assertEqual(built.layers, {})
            self.assertTrue(built.reasons)
            self.assertEqual(kernels.compiled, [])
        built = m31_attention.build_low_latency(
            _model(_Trace(), 1), (16, 24), _FakeKernels()
        )
        self.assertEqual(built.widths, ())

    def test_layers_off_the_branch_stay_stock(self):
        trace = _Trace()
        model = _model(trace, num_layers=10)
        layers = model.model.layers

        def bias(attn):
            attn.o_proj.bias = Parameter(torch.zeros(6144))

        def deepgemm(attn):
            attn.o_proj.quant_method.mxfp8_dense_backend = (
                Mxfp8DenseGemmBackend.DEEP_GEMM
            )

        def no_scales(attn):
            del attn.fused_qkv_index_proj.weight_scale_inv_swizzled

        def overridden(attn):
            attn.forward_core = lambda s: None

        def fp8_q_off(attn):
            attn._fp8_q_epilogue_ok = False

        def index_value(attn):
            attn.disable_index_value = False

        def unfused(attn):
            attn._fused_qkv_index = None

        def reduces(attn):
            attn.o_proj.reduce_results = True

        def logged_apply(attn):
            # SGLANG_KERNEL_API_LOGLEVEL: LinearBase wraps its quant method's apply
            # on the instance, and the stock call would log through it
            from sglang.kernels import kernel_api_logging

            with patch.object(kernel_api_logging, "_KERNEL_API_LOG_LEVEL", 1):
                kernel_api_logging.wrap_method_with_debug_kernel_once(
                    attn.fused_qkv_index_proj._qm,
                    "apply",
                    op_name="sglang.quant_method.Fp8LinearMethod.apply",
                )
            self.assertIn("apply", vars(attn.fused_qkv_index_proj._qm))

        changes = (
            bias,
            deepgemm,
            no_scales,
            overridden,
            fp8_q_off,
            index_value,
            unfused,
            reduces,
            logged_apply,
        )
        for i, change in enumerate(changes):
            attn = layers[i].self_attn
            change(attn)
            self.assertTrue(m31_attention.layer_reasons(layers[i]), change.__name__)
        self.assertIn(
            "apply is replaced on the instance",
            "; ".join(m31_attention.layer_reasons(layers[len(changes) - 1])),
        )
        clean = len(changes)
        self.assertEqual(m31_attention.layer_reasons(layers[clean]), [])
        built = m31_attention.build_low_latency(model, (8,), _FakeKernels())
        self.assertEqual(built.widths, (8,))
        self.assertEqual(sorted(built.layers), [clean])
        self.assertEqual(sorted(built.skipped), list(range(clean)))

    def test_no_layer_on_the_branch_serves_no_width(self):
        model = _model(_Trace(), num_layers=2)
        for layer in model.model.layers:
            layer.self_attn._combined_qknorm_ok = False
        built = m31_attention.build_low_latency(model, (8,), _FakeKernels())
        self.assertEqual(built.widths, ())
        self.assertIn("no layer", "; ".join(built.reasons))

    def test_build_plan_runs_the_slot_only_at_its_widths(self):
        model = _model(_Trace(), num_layers=2)
        ctx = SimpleNamespace(
            widths=tuple(range(8, 129, 8)), tokens_per_request=8, binding=None
        )
        one_rank = patch.object(
            m31_agree, "gather_tp", side_effect=lambda x, group=None: [x]
        )
        with (
            patch.object(m31_runtime, "SLOTS", frozenset({"attn_ll"})),
            patch.object(m31_attention, "_GpuKernels", _FakeKernels),
            one_rank,
        ):
            built = m31_runtime.build(model, ctx)
        self.assertEqual(built.slot_plan[8], m31_plan.Slots(attn_ll=True))
        for w in range(16, 129, 8):
            self.assertEqual(built.slot_plan[w], m31_plan.Slots())
        self.assertIn("attn_ll at [8] rows", built.summary())
        with (
            patch.object(m31_runtime, "SLOTS", frozenset({"attn_ll"})),
            patch.object(
                m31_attention, "_GpuKernels", lambda: _FakeKernels(reasons=["no"])
            ),
            patch.object(m31_agree, "gather_tp", side_effect=lambda x, group=None: [x]),
            self.assertRaisesRegex(RuntimeError, "attn_ll"),
        ):
            m31_runtime.build(model, ctx)


class TestStepMatchesSGLangsAttention(CustomTestCase):
    """The slot's step against SGLang's MiniMaxM3Attention.forward on the same layer:
    the same quantize call (so the same MXFP8 GEMM inputs), the same q/k norm + RoPE,
    the same attention backend call and the same output, with the GEMMs replaced."""

    def _run(self, layer, x, rows, *, slot, built=None, sp_active=False):
        trace = _Trace()
        layer.self_attn.attn.trace = trace
        st = _state(0, rows, built or _bound([layer]), slot=slot)
        with _kernels_faked(trace, sp_active=sp_active):
            out = m31_attention.step(layer, x, st)
        return trace, out

    def _check_against_stock(self, x, rows, gemms=QKV_ONLY):
        """The slot's step with ``gemms`` on the low-latency kernel against SGLang's."""
        layer = _layer(_Trace())
        stock, want = self._run(layer, x, rows, slot=False)
        built = _bound([layer], gemms=gemms)
        ll, got = self._run(layer, x, rows, slot=True, built=built)
        self.assertNotIn("ll_gemm", stock.names())
        stock_gemms = stock.named("gemm")
        self.assertEqual(len(stock_gemms), 2)
        # in call order the q|k|v|index GEMM, then o_proj's: each on the low-latency
        # kernel where the slot runs it there, else SGLang's own call
        names = tuple(m31_attention.SHAPES)
        self.assertEqual(
            [n for n in ll.names() if n in ("gemm", "ll_gemm")],
            ["ll_gemm" if name in gemms else "gemm" for name in names],
        )
        ll_gemms, ll_stock = iter(ll.named("ll_gemm")), iter(ll.named("gemm"))
        linears = (layer.self_attn.fused_qkv_index_proj, layer.self_attn.o_proj)
        for name, linear, stock_call in zip(names, linears, stock_gemms):
            q, s, w, ws, kw, backend = stock_call
            self.assertEqual(
                kw, dict(out_dtype=torch.bfloat16, use_8x4_sf_layout=False)
            )
            self.assertEqual(backend, "cute-dsl")
            if name not in gemms:
                # SGLang's GEMM, called as the stock forward calls it
                self.assertTrue(_same(next(ll_stock), stock_call), name)
                continue
            # the stock GEMM's operands are the low-latency GEMM's operands
            a, a_s, lw, lws, tactic = next(ll_gemms)
            self.assertTrue(_same(q, a))
            self.assertTrue(_same(s, a_s))
            self.assertEqual((w, ws), (lw, lws))
            self.assertEqual(
                (lw, lws), (id(linear.weight), id(linear.weight_scale_inv_swizzled))
            )
            self.assertEqual(
                tactic,
                mxfp8_ll.pinned_tactic(103, *m31_attention.SHAPES[name], rows),
            )
        # everything else, call for call
        for name in ("quantize", "qknorm", "attn", "allocation"):
            self.assertTrue(_same(stock.named(name), ll.named(name)), name)
        self.assertEqual(
            [n for n in stock.names() if n != "gemm"],
            [n for n in ll.names() if n not in ("gemm", "ll_gemm")],
        )
        attn = layer.self_attn
        (qknorm,) = ll.named("qknorm")
        self.assertEqual(
            qknorm[1], [(id(w), off, cnt) for w, off, cnt in attn._qknorm_groups()]
        )
        self.assertEqual(
            qknorm[2:5], (id(attn.rotary_emb.cos_sin_cache), id(_POSITIONS[rows]), 1e-6)
        )
        self.assertEqual((qknorm[6], qknorm[8]), (0, 2))
        (call,) = ll.named("attn")
        self.assertEqual(call[3], id(_FORWARD_BATCH))
        self.assertIsNone(call[6])
        self.assertEqual(ll.named("allocation"), [(id(_TP_GROUP), False)])
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertEqual(tuple(got.shape), (rows, 6144))
        self.assertTrue(torch.equal(got, want))
        return stock, ll

    def test_bf16_input_at_every_served_width(self):
        for gemms in (QKV_ONLY, BOTH, ("o_proj",)):
            for rows in range(1, 9):
                with self.subTest(gemms=gemms, rows=rows):
                    stock, _ = self._check_against_stock(_inputs(rows), rows, gemms)
                    quantize = stock.named("quantize")
                    # the q|k|v|index input, then o_proj's
                    self.assertEqual(
                        [q[0][1] for q in quantize], [(rows, HIDDEN), (rows, ATTN_OUT)]
                    )
                    self.assertEqual(
                        quantize[0][1], dict(is_sf_swizzled_layout=True, alignment=32)
                    )

    def test_mxfp8_input_skips_the_quantize(self):
        for gemms in (QKV_ONLY, BOTH):
            for rows in (1, 8):
                with self.subTest(gemms=gemms, rows=rows):
                    x = _inputs(rows)
                    q = x.to(E4M3)
                    s = (torch.arange(mxfp8_ll.scale_bytes(rows, HIDDEN)) % 5 + 125).to(
                        torch.uint8
                    )
                    pair = Mxfp8SwizzledInput(q, s)
                    stock, ll = self._check_against_stock(pair, rows, gemms)
                    # only o_proj's input is quantized
                    self.assertEqual(len(stock.named("quantize")), 1)
                    self.assertEqual(stock.named("quantize")[0][0][1], (rows, ATTN_OUT))
                    # the low-latency q|k|v|index GEMM is the step's first kernel, on
                    # the pair as given: nothing runs between the pair's producer and
                    # it (what lets hop_mxfp8's kernel signal it at entry)
                    self.assertEqual(ll.names()[0], "ll_gemm")
                    self.assertTrue(_same(ll.named("ll_gemm")[0][0], _desc(q)))
                    self.assertTrue(_same(ll.named("ll_gemm")[0][1], _desc(s)))

    def test_slot_off_runs_the_stock_module_call(self):
        calls = []

        class Attn(torch.nn.Module):
            def forward(self, positions, hidden_states, forward_batch):
                calls.append((positions, hidden_states, forward_batch))
                return "stock"

        layer = SimpleNamespace(self_attn=Attn())
        st = _state(0, 8, None, slot=False)
        x = _inputs(8)
        self.assertEqual(m31_attention.step(layer, x, st), "stock")
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], st.positions)
        self.assertIs(calls[0][1], x)
        self.assertIs(calls[0][2], st.fb)

    def test_fallbacks_to_the_stock_module(self):
        """A layer the build left out, or a step whose input or forward state the
        slot does not reproduce, runs SGLang's attention (no low-latency GEMM)."""
        rows = 8
        x = _inputs(rows)

        def unbound(layer):
            return dict(built=_bound([]))

        def no_tactic(layer):
            return dict(built=_bound([layer], rows=(1, 2)))

        def sequence_parallel(layer):
            return dict(sp_active=True)

        def decode_attention_tp(layer):
            layer.self_attn.o_proj.use_decode_attn_tp = True
            return {}

        def attention_hook(layer):
            layer.self_attn.register_forward_hook(lambda *a: None)
            return {}

        def linear_pre_hook(layer):
            layer.self_attn.o_proj.register_forward_pre_hook(lambda *a: None)
            return {}

        def short_scales(layer):
            # an MXFP8 input whose scales fall short of one 128-row atom
            q = _inputs(rows).to(E4M3)
            return dict(x=Mxfp8SwizzledInput(q, torch.zeros(64, dtype=torch.uint8)))

        for case in (
            unbound,
            no_tactic,
            sequence_parallel,
            decode_attention_tp,
            attention_hook,
            linear_pre_hook,
            short_scales,
        ):
            layer = _layer(_Trace())
            kw = case(layer)
            trace, _ = self._run(layer, kw.pop("x", x), rows, slot=True, **kw)
            self.assertNotIn("ll_gemm", trace.names(), case.__name__)
            self.assertEqual(trace.names().count("gemm"), 2, case.__name__)
            if case is sequence_parallel:
                self.assertIn("reduce_scatter", trace.names())
            if case is decode_attention_tp:
                self.assertIn("all_reduce", trace.names())

    def test_o_proj_falls_back_alone_when_the_attention_output_is_not_bf16(self):
        rows = 4
        layer = _layer(_Trace())
        layer.self_attn.attn.out_dtype = torch.float16
        built = _bound([layer], gemms=BOTH)
        trace, _ = self._run(layer, _inputs(rows), rows, slot=True, built=built)
        self.assertEqual(trace.names().count("ll_gemm"), 1)
        self.assertEqual(trace.names().count("gemm"), 1)
        # o_proj ran as SGLang's RowParallelLinear, under its allocation context
        self.assertEqual(trace.named("allocation"), [(id(_TP_GROUP), False)])


class TestGemmsKnob(CustomTestCase):
    """The slot runs only the q|k|v|index GEMM on the low-latency kernel by default;
    the test-only GEMMS_ENV adds o_proj (or runs o_proj alone), and a name it does not
    know stops the build."""

    def test_qkv_only_by_default_and_o_proj_by_the_knob(self):
        env = m31_attention.GEMMS_ENV
        self.assertEqual(m31_attention.DEFAULT_GEMMS, QKV_ONLY)
        for raw, want in (
            (None, QKV_ONLY),
            ("", QKV_ONLY),
            (" , ", QKV_ONLY),
            ("qkv_index", QKV_ONLY),
            ("o_proj,qkv_index", BOTH),
            (" qkv_index , o_proj ", BOTH),
            ("o_proj", ("o_proj",)),
        ):
            with self.subTest(raw=raw):
                environ = {} if raw is None else {env: raw}
                self.assertEqual(m31_attention.requested_gemms(environ), want)
        with self.assertRaisesRegex(ValueError, f"{env}: unknown GEMM 'qkv'"):
            m31_attention.requested_gemms({env: "qkv,o_proj"})

    def _build(self, environ):
        one_rank = patch.object(
            m31_agree, "gather_tp", side_effect=lambda x, group=None: [x]
        )
        with (
            patch.object(m31_attention, "_GpuKernels", _FakeKernels),
            one_rank,
            patch.dict(os.environ, environ),
        ):
            for name in (m31_attention.GEMMS_ENV, m31_attention.LAYERS_ENV):
                if name not in environ:
                    os.environ.pop(name, None)
            return m31_attention.build(
                _model(_Trace(), 3), (8, 16), frozenset({"attn_ll"})
            )

    def test_the_build_reads_the_knob(self):
        with self.assertNoLogs(m31_attention.logger, level="WARNING"):
            built = self._build({})
        self.assertEqual(built.gemms, QKV_ONLY)
        self.assertEqual(set(built.tactics), {"qkv_index"})
        self.assertTrue(all(g.o_proj is None for g in built.layers.values()))
        with self.assertLogs(m31_attention.logger, level="WARNING") as logs:
            built = self._build({m31_attention.GEMMS_ENV: "qkv_index,o_proj"})
        self.assertEqual(built.gemms, BOTH)
        self.assertTrue(all(g.o_proj is not None for g in built.layers.values()))
        self.assertIn("GEMMs ['qkv_index', 'o_proj']", "\n".join(logs.output))
        with self.assertLogs(m31_attention.logger, level="WARNING"):
            built = self._build({m31_attention.GEMMS_ENV: "o_proj"})
        self.assertEqual(built.gemms, ("o_proj",))
        self.assertTrue(all(g.qkv_index is None for g in built.layers.values()))
        with self.assertRaisesRegex(ValueError, "unknown GEMM"):
            self._build({m31_attention.GEMMS_ENV: "q_proj"})
        # the layers knob still applies on top
        with self.assertLogs(m31_attention.logger, level="WARNING"):
            built = self._build({m31_attention.LAYERS_ENV: "0,2"})
        self.assertEqual(sorted(built.layers), [0, 2])


def _pair(rows, *, scale_bytes=None):
    """An MXFP8 pair as slot hop_mxfp8 hands it over: e4m3 rows and one 128-row atom
    of UE8M0 scales."""
    q = _inputs(rows).to(E4M3)
    n = mxfp8_ll.scale_bytes(8, HIDDEN) if scale_bytes is None else scale_bytes
    return Mxfp8SwizzledInput(q, (torch.arange(n) % 5 + 125).to(torch.uint8))


class TestEarlyTriggerDecision(CustomTestCase):
    """attention.low_latency_qkv, the step's own decision, asked by hop_mxfp8's
    boundary kernel before it launches: on an MXFP8 pair it is True exactly where
    step() then launches the low-latency q|k|v|index GEMM first, with nothing before
    it, so the boundary may signal that GEMM at its entry."""

    def _decide_and_run(self, layer, built, rows, *, x=None, slot=True, **faked):
        x = _pair(rows) if x is None else x
        st = _state(0, rows, built, slot=slot)
        trace = _Trace()
        layer.self_attn.attn.trace = trace
        with _kernels_faked(trace, **faked):
            decided = m31_attention.low_latency_qkv(layer, x, st)
            m31_attention.step(layer, x, st)
        kernels = [n for n in trace.names() if n != "allocation"]
        return decided, kernels[0]

    def test_true_where_the_step_launches_the_low_latency_gemm_first(self):
        for gemms in (QKV_ONLY, BOTH):
            for rows in range(1, 9):
                with self.subTest(gemms=gemms, rows=rows):
                    layer = _layer(_Trace())
                    decided, first = self._decide_and_run(
                        layer, _bound([layer], gemms=gemms), rows
                    )
                    self.assertTrue(decided)
                    self.assertEqual(first, "ll_gemm")

    def test_false_wherever_the_step_launches_anything_else_first(self):
        rows = 8

        def o_proj_only(layer):
            return dict(built=_bound([layer], gemms=("o_proj",)))

        def slot_off_for_the_width(layer):
            return dict(slot=False)

        def unbound_layer(layer):
            return dict(built=_bound([]))

        def layers_knob(layer):
            built = _bound([layer])
            built.layers = {}  # what LAYERS_ENV leaves of the build
            return dict(built=built)

        def no_tactic_for_the_rows(layer):
            return dict(built=_bound([layer], rows=(1, 2)))

        def attention_hook(layer):
            layer.self_attn.register_forward_hook(lambda *a: None)
            return {}

        def sequence_parallel(layer):
            return dict(sp_active=True)

        def decode_attention_tp(layer):
            layer.self_attn.o_proj.use_decode_attn_tp = True
            return {}

        def short_scales(layer):
            return dict(x=_pair(rows, scale_bytes=64))

        def too_many_rows(layer):
            return dict(x=_pair(9))

        for case in (
            o_proj_only,
            slot_off_for_the_width,
            unbound_layer,
            layers_knob,
            no_tactic_for_the_rows,
            attention_hook,
            sequence_parallel,
            decode_attention_tp,
            short_scales,
            too_many_rows,
        ):
            with self.subTest(case.__name__):
                layer = _layer(_Trace())
                kw = case(layer)
                built = kw.pop("built", None) or _bound([layer])
                x = kw.pop("x", None)
                step_rows = rows if x is None else int(x.data.shape[0])
                decided, first = self._decide_and_run(
                    layer, built, step_rows, x=x, **kw
                )
                self.assertFalse(decided)
                self.assertEqual(first, "gemm")

    def test_a_bf16_input_is_quantized_first(self):
        """The boundary asks only about the pair it writes; on a bf16 input the
        slot's own quantize comes before the GEMM."""
        layer = _layer(_Trace())
        decided, first = self._decide_and_run(layer, _bound([layer]), 8, x=_inputs(8))
        self.assertTrue(decided)
        self.assertEqual(first, "quantize")

    def test_the_boundary_asks_the_attention_with_its_pair(self):
        """boundary.attn_input hands slot hop_mxfp8's build a predicate; on the pair
        it writes, that predicate answers what the attention step then does."""
        for attn_ll, gemms, early in (
            (True, QKV_ONLY, True),
            (True, ("o_proj",), False),
            (False, QKV_ONLY, False),
        ):
            with self.subTest(attn_ll=attn_ll, gemms=gemms):
                layer = _layer(_Trace())
                pair = _pair(8)
                seen = {}

                def take(layer_id, layer_, x, fb, captured, tally, early_trigger):
                    seen["early"] = early_trigger(pair)
                    return pair

                st = _state(0, 8, _bound([layer], gemms=gemms), slot=attn_ll)
                st.slots = m31_plan.Slots(attn_ll=attn_ll, hop_mxfp8=True)
                st.bound["boundary"] = SimpleNamespace(attn_input=take)
                st.scratch = {}
                trace = _Trace()
                layer.self_attn.attn.trace = trace
                with _kernels_faked(trace):
                    x = m31_boundary.attn_input(layer, "stream output", st, None)
                    m31_attention.step(layer, x, st)
                self.assertIs(x, pair)
                self.assertIs(seen["early"], early)
                first = next(n for n in trace.names() if n != "allocation")
                self.assertEqual(first, "ll_gemm" if early else "gemm")
                self.assertEqual(st.scratch[m31_boundary.TALLY], Counter())


class TestPdlOrder(CustomTestCase):
    """The low-latency GEMM lets the next kernel start before its own PDL wait
    (mxfp8_ll): the kernels after the slot's two GEMMs must read nothing written
    before the GEMM ahead of their own wait. Tripwires on SGLang's code."""

    def test_qknorm_reads_the_qkv_row_only_after_its_wait(self):
        import sglang

        src = (
            Path(sglang.__file__).parent
            / "kernels/jit/csrc/minimax/fused_gemma_qknorm_rope.cuh"
        ).read_text()
        body = src[src.index("const auto input = params.qkv") :]
        before_wait = body[: body.index("PDLWaitPrimary")]
        # only the norm weights and the position (a static graph input) before it
        self.assertNotIn("input[", before_wait)
        self.assertIn("weight[idx_0]", before_wait)
        self.assertIn("params.positions", before_wait)
        self.assertIn("input[idx_0]", body[body.index("PDLWaitPrimary") :])

    def test_the_boundary_before_the_gemm_signals_at_entry_only_on_request(self):
        """all_reduce_fusion.cuh: the MXFP8 form's entry trigger is the kernel's first
        PDL instruction, before its wait and before the bumper cluster branches off,
        so every CTA executes it; only the non-mHC kQuant entries take the flag, off
        by default; the end trigger after the epilogue stays."""
        import sglang

        src = (
            Path(sglang.__file__).parent
            / "kernels/jit/csrc/distributed/all_reduce_fusion.cuh"
        ).read_text()
        self.assertIn("  bool trigger_at_entry = false;\n};", src)
        body = src[src.index("void moe_finalize_all_reduce_kernel(") :]
        body = body[: body.index("// --- host")]
        entry = body.index(
            "if (params.trigger_at_entry) PDLTriggerSecondary<kUsePDL>();"
        )
        self.assertLess(entry, body.index("PDLWaitPrimary"))
        self.assertLess(entry, body.index("if (row_idx == params.num_tokens)"))
        self.assertIn(
            "if constexpr (kQuant && !kMhc) {\n"
            "    if (params.trigger_at_entry) PDLTriggerSecondary<kUsePDL>();",
            body,
        )
        # the bumper's and the epilogue's triggers are unchanged
        self.assertEqual(body.count("PDLTriggerSecondary<kUsePDL>()"), 3)
        self.assertIn("return PDLTriggerSecondary<kUsePDL>();", body)
        # the host: the two MXFP8 entries pass the flag, everything else the default
        for entry_name in ("run_residual_norm_quant", "run_plain_quant"):
            start = src.index(f"static void {entry_name}(")
            text = src[start : src.index("\n  }\n", start)]
            self.assertIn("bool trigger_at_entry) {", text)
            self.assertTrue(
                text.rstrip().endswith("std::nullopt,\n        trigger_at_entry);"),
                entry_name,
            )
        self.assertEqual(src.count("trigger_at_entry);"), 2)
        self.assertIn("bool trigger_at_entry = false) {", src)
        self.assertIn(".trigger_at_entry = trigger_at_entry,", src)

    def test_the_boundary_after_o_proj_waits_first(self):
        import sglang
        from sglang.kernels.ops.communication import all_reduce_fusion
        from sglang.srt.layers.layer_boundary.fusions import custom_all_reduce_fusion

        # the attention -> FFN boundary of every layer: all-reduce + residual + norm
        self.assertIs(
            custom_all_reduce_fusion.all_reduce_residual_norm,
            all_reduce_fusion.all_reduce_residual_norm,
        )
        # it, and its quantizing forms that slot hop_ffn launches (boundary/hop_ffn.py), take
        # no prefetch_metadata ...
        for op in (
            all_reduce_fusion.all_reduce_residual_norm,
            all_reduce_fusion.all_reduce_residual_norm_quant,
            all_reduce_fusion.all_reduce_residual_norm_quant_nvfp4,
        ):
            self.assertNotIn(
                "prefetch_metadata", inspect.signature(op).parameters, op.__name__
            )
        # ... and their kernel entries (all_reduce_fusion.cuh) pass it false to
        # run_impl, its ninth argument after norm_weight and eps: the kernel waits
        # before it reads
        src = (
            Path(sglang.__file__).parent
            / "kernels/jit/csrc/distributed/all_reduce_fusion.cuh"
        ).read_text()
        for entry_name in ("run_plain", "run_plain_quant", "run_plain_quant_nvfp4"):
            start = src.index(f"static void {entry_name}(")
            text = src[start : src.index("\n  }\n", start)]
            call = text[text.index("run_impl(") + len("run_impl(") :]
            args = [arg.strip() for arg in call[: call.index(");")].split(",")]
            self.assertEqual(args[6:9], ["norm_weight", "eps", "false"], entry_name)
        # the only launcher that may read before its wait does not by default
        prefetch = inspect.signature(all_reduce_fusion.moe_finalize_all_reduce)
        self.assertIs(prefetch.parameters["prefetch_metadata"].default, False)


class TestLowLatencyLinear(CustomTestCase):
    def test_input_checks(self):
        lin = m31_attention.LowLatencyLinear(_fused_proj(), {8: (256, 6, 1, 4)})
        self.assertEqual(lin.reasons(_inputs(8)), [])
        self.assertTrue(lin.reasons(_inputs(4)))  # no tactic
        self.assertTrue(lin.reasons(_inputs(8).float()))
        self.assertTrue(lin.reasons(_inputs(8)[:, :2048]))
        self.assertTrue(lin.reasons((torch.zeros(1), torch.zeros(1))))
        q = torch.zeros(8, HIDDEN, dtype=E4M3)
        s = torch.zeros(mxfp8_ll.scale_bytes(8, HIDDEN), dtype=torch.uint8)
        self.assertEqual(lin.reasons(Mxfp8SwizzledInput(q, s)), [])
        self.assertTrue(lin.reasons(Mxfp8SwizzledInput(q[:4], s)))
        self.assertTrue(lin.reasons(Mxfp8SwizzledInput(q, s[:100])))


if __name__ == "__main__":
    unittest.main()
