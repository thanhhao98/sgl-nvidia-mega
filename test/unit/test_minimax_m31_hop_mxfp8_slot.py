# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of the M3.1 megakernel's slot hop_mxfp8 (the attention-input boundary
kernel that also writes the MXFP8 of its normed output): the layers and widths it
takes, its attention input against SGLang's own ``attn_boundary.prepare`` on stage
boundaries built by SGLang's factories (the kernels faked alike), its fallbacks to
the stock prepare and the per-width log of why it fell back, and the stock attention
module taking the MXFP8 pair without quantizing again. The quant ops' plumbing and the
kernel's scale-layout arithmetic are SGLang's tests, with the ops."""

import contextlib
import unittest
from collections import Counter
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import pytest
import torch

pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

from sglang.kernels.ops.communication import all_reduce_fusion as ops  # noqa: E402
from sglang.srt import batch_invariant_ops, runtime_context  # noqa: E402
from sglang.srt.layers.aux_hidden_states import (  # noqa: E402
    AuxHiddenStateList,
    AuxHiddenStatePacker,
)
from sglang.srt.layers.layer_boundary import (  # noqa: E402
    PLAIN_ADD,
    SumGroup,
    UnreducedOutput,
)
from sglang.srt.layers.layer_boundary.contracts import BatchVariant  # noqa: E402
from sglang.srt.layers.layer_boundary.fusions import (  # noqa: E402
    custom_all_reduce_fusion as custom,
)
from sglang.srt.layers.layer_boundary.fusions.moe_finalize import (  # noqa: E402
    MoeDeferredFinalize,
)
from sglang.srt.layers.layer_boundary.prepare import _run_entry  # noqa: E402
from sglang.srt.layers.layer_boundary.residual import (  # noqa: E402
    batch as residual_batch,
)
from sglang.srt.layers.layer_boundary.residual.stream import (  # noqa: E402
    ResidualStream,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm  # noqa: E402
from sglang.srt.layers.quantization import (  # noqa: E402
    mxfp4_flashinfer_trtllm_moe as moe_quant,
)
from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.test.boundary_fixtures import make_test_stages, planning  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

from sgl_nvidia_mega.minimax_m31 import boundary  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import plan as m31_plan  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import runtime as m31_runtime  # noqa: E402
from sgl_nvidia_mega.minimax_m31.runtime import agree as m31_agree  # noqa: E402
from sgl_nvidia_mega.minimax_m31.runtime import executor as m31_executor  # noqa: E402

if not hasattr(ops, "load_residual_norm_quant"):
    pytest.skip(
        "this SGLang has no MXFP8 epilogue of the push-plane boundary kernel",
        allow_module_level=True,
    )

H = 64  # a hidden width the CPU fakes can carry; the slot's kernel sees 6144
TOP_K = 4
TP = SimpleNamespace(name="tp", ranks=[0, 1, 2, 3])
PARALLEL = SimpleNamespace(
    tp_size=4,
    tp_rank=0,
    pp_size=1,
    attn_dp_size=1,
    attn_tp_size=4,
    attn_tp_rank=0,
    attn_cp_size=1,
    attn_cp_rank=0,
    moe_dense_tp_size=None,
    enable_prefill_cp=False,
    enable_cp_tp_group_sharing=False,
    moe_ep_size=1,
    moe_tp_size=4,
    moe_dp_size=1,
    dwdp_size=1,
    attn_dp_enabled=False,
    enable_attn_tp_input_scattered=False,
    disable_attn_tp_gather=False,
    tp_group=TP,
    attn_tp_group=SimpleNamespace(name="attn_tp", ranks=[0, 1, 2, 3]),
    attn_cp_group=SimpleNamespace(name="attn_cp", ranks=[0]),
)


@contextlib.contextmanager
def tp4(*, fusable=True, counters=512):
    """What building and running the stage boundaries reads at TP4 without attention
    DP, CP or an a2a backend (SGLang's own ``planning`` fixture), and the process
    state CustomAllReduceFusion gates on: the registered push plane (TP4,
    ``counters`` row counters and 4 MiB slots, as M3.1 registers it) and the fusion's
    question to it (``fusable``; None: the real ``can_fuse_all_reduce`` on that
    plane)."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(planning(PARALLEL))
        for target, name, value in (
            (runtime_context, "get_parallel", lambda: PARALLEL),
            (custom, "get_parallel", lambda: PARALLEL),
            (
                custom,
                "get_exec",
                lambda: SimpleNamespace(
                    comm=SimpleNamespace(enable_quant_communications=False)
                ),
            ),
            (
                custom,
                "get_attn_tp_context",
                lambda: SimpleNamespace(input_scattered=False),
            ),
            (custom, "is_in_breakable_cuda_graph", lambda: False),
            (batch_invariant_ops, "is_batch_invariant_mode_enabled", lambda: False),
            (moe_quant, "get_parallel", lambda: PARALLEL),
            (
                moe_quant,
                "_fused_all_reduce_comm",
                SimpleNamespace(
                    world_size=4,
                    max_push_size=4 << 20,
                    config=SimpleNamespace(num_push_blocks=counters),
                ),
            ),
        ):
            stack.enter_context(patch.object(target, name, value))
        if fusable is not None:
            stack.enter_context(
                patch.object(custom, "can_fuse_all_reduce", lambda m, h: fusable)
            )
        yield


def _counter_rule(rows):
    """The slot's padding gate before it asked can_fuse_all_reduce: the registered
    plane's row counters, read directly (the kernel's own host check)."""
    comm = moe_quant._fused_all_reduce_comm
    return comm is not None and (
        rows == ops.MXFP8_QUANT_MAX_TOKENS or rows < comm.config.num_push_blocks
    )


def _norm(dtype=torch.bfloat16):
    norm = GemmaRMSNorm(H).to(dtype=dtype)
    torch.add(torch.linspace(-0.5, 0.5, H), 1.0, out=norm.gemma_weight)
    return norm


def _mxfp8_projection(backend="cute-dsl", use_mxfp8=True, k=H):
    """An attention whose fused qkv|index projection runs a FlashInfer MXFP8 GEMM."""
    qm = SimpleNamespace(
        use_mxfp8=use_mxfp8,
        use_marlin=False,
        block_fp8_as_mxfp8=False,
        mxfp8_dense_backend=SimpleNamespace(
            is_flashinfer_cutlass=lambda: backend == "cutlass",
            is_flashinfer_cutedsl=lambda: backend == "cute-dsl",
            is_deep_gemm=lambda: backend == "deep_gemm",
        ),
    )
    proj = SimpleNamespace(_qm=qm, weight=torch.empty(40, k))
    return SimpleNamespace(_fused_qkv_index=proj, fused_qkv_index_proj=proj)


def _layer(*, sparse_before=True, first=False, fusions=True, norm=None, attn=None):
    """One decoder layer's stage boundaries from SGLang's factories (after a MoE
    layer, or after a dense one), the push-plane fusion on both stages as
    MiniMaxM3DecoderLayer installs it."""
    fusion = (
        custom.CustomAllReduceFusion(hidden_size=H, defers_finalize=True)
        if fusions
        else None
    )
    with tp4():
        stages = make_test_stages(
            attention_norm=norm or _norm(),
            ffn_norm=_norm(),
            first=first,
            sparse=True,
            previous_sparse=sparse_before,
            **({"fusions": fusion} if fusions else {}),
        )
    return SimpleNamespace(
        attn_boundary=stages.attn,
        ffn_boundary=stages.ffn,
        self_attn=attn or _mxfp8_projection(),
    )


def _hop(layer, layer_id=1, widths=(8, 16)):
    hop, why = boundary.layer_hop(layer, hidden=H)
    assert hop is not None, why
    return boundary.HopMxfp8(
        widths=widths, hops={layer_id: hop}, refused={}, hidden=H, device="cpu"
    )


def _keeping(layer, asked=None):
    """``layer`` with its attention entry certifying that the following stage keeps
    the residual (``capture_preserves_residual``, as ``_carry_capture`` binds it),
    so a DSpark capture adopts the updated residual without a copy; ``asked`` (a
    list) records the residuals the predicate is asked about."""
    paths = layer.attn_boundary.plan.paths
    steps = paths[BatchVariant.ORDINARY]

    def keeps(value, forward_batch):
        if asked is not None:
            asked.append(value)
        return True

    paths[BatchVariant.ORDINARY] = msgspec.structs.replace(
        steps,
        entry=msgspec.structs.replace(steps.entry, capture_preserves_residual=keeps),
    )
    return layer


def _moe_output(rows, seed, *, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    return MoeDeferredFinalize(
        routed_output=torch.randn(rows * TOP_K + 3, H, generator=g).to(dtype),
        expert_weights=torch.rand(rows, TOP_K, generator=g),
        permuted_indices=torch.randperm(rows * TOP_K, generator=g)
        .int()
        .view(rows, TOP_K),
        gated_shared_output=torch.randn(rows, H, generator=g).bfloat16(),
        m=rows,
        finish=Mock(side_effect=AssertionError("the fused kernel takes the handoff")),
    )


def _dense_output(rows, seed, *, dtype=torch.bfloat16, group=TP, **kw):
    g = torch.Generator().manual_seed(seed)
    return UnreducedOutput(torch.randn(rows, H, generator=g).to(dtype), group, **kw)


def _batch(rows, owed, seed, *, mode=ForwardMode.TARGET_VERIFY, residual=True):
    """A forward batch whose residual stream holds ``owed``, the previous layer's
    pending output; returns the batch and the model-facing handle of that output."""
    g = torch.Generator().manual_seed(seed + 1)
    stream = ResidualStream(
        torch.randn(rows, H, generator=g).bfloat16() if residual else None
    )
    x = stream.record(owed, PLAIN_ADD)
    fb = SimpleNamespace(
        forward_mode=mode,
        can_run_tbo=False,
        input_ids=torch.zeros(rows, dtype=torch.long),
        attn_cp_metadata=None,
        residual_stream=stream,
    )
    return fb, x


def _normed(residual):
    return (residual.float() * 3 - 1).bfloat16()


def _stock_kernel(*args, **kw):
    """SGLang's push-plane kernel, faked: a fresh normed row, the updated residual
    into ``residual_out``."""
    residual = args[5] if len(args) > 4 else args[1]
    kw["residual_out"].copy_(residual + 1)
    return _normed(residual), kw["residual_out"]


def _quant_kernel(*args, **kw):
    """Its MXFP8 form, faked alike, writing the codes and scales it was given."""
    normed, residual_out = _stock_kernel(*args, **kw)
    kw["out"].copy_(normed)
    kw["quantized"].copy_(normed.float().to(torch.float8_e4m3fn))
    kw["scales"].fill_(len(args))
    return kw["out"], residual_out, kw["quantized"], kw["scales"]


@contextlib.contextmanager
def kernels():
    """The stock kernels (as CustomAllReduceFusion calls them) and the quant ops (as
    the slot calls them), faked and recorded."""
    calls = SimpleNamespace(
        stock_finalize=Mock(side_effect=_stock_kernel),
        stock_plain=Mock(side_effect=_stock_kernel),
        quant_finalize=Mock(side_effect=_quant_kernel),
        quant_plain=Mock(side_effect=_quant_kernel),
    )
    with (
        patch.object(
            custom, "moe_finalize_all_reduce_residual_norm", calls.stock_finalize
        ),
        patch.object(custom, "all_reduce_residual_norm", calls.stock_plain),
        patch.object(
            ops, "moe_finalize_all_reduce_residual_norm_quant", calls.quant_finalize
        ),
        patch.object(ops, "all_reduce_residual_norm_quant", calls.quant_plain),
    ):
        yield calls


def _same(a, b) -> bool:
    """Equal values of the same type; tensors bit for bit."""
    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and a.shape == b.shape
            and a.dtype == b.dtype
            and torch.equal(_bits(a), _bits(b))
        )
    return a == b


def _bits(t: torch.Tensor) -> torch.Tensor:
    # torch.equal has no fp8 kernel on CPU
    return t.view(torch.uint8) if t.element_size() == 1 and t.is_floating_point() else t


class TestAttentionInputMatchesStockPrepare(CustomTestCase):
    """The slot's attention input makes the stock kernel call in its MXFP8 form and
    leaves the residual stream and the DSpark capture as attn_boundary.prepare does."""

    def _run(self, *, moe, rows, capture, slot, keeps=False):
        layer = _layer(sparse_before=moe)
        if keeps:
            _keeping(layer)
        owed = _moe_output(rows, 7) if moe else _dense_output(rows, 7)
        fb, x = _batch(rows, owed, 7)
        stream, pending, residual = fb.residual_stream, fb.residual_stream.pending, None
        residual = stream.residual
        aux = AuxHiddenStateList() if capture else None
        with tp4(), kernels() as calls:
            if slot:
                hop = _hop(layer, widths=(rows,))
                got = hop.attn_input(1, layer, x, fb, aux)
            else:
                got = layer.attn_boundary.prepare(x, fb, capture_gathered=aux)
        stock = calls.stock_finalize if moe else calls.stock_plain
        quant = calls.quant_finalize if moe else calls.quant_plain
        call = (quant if slot else stock).call_args
        self.assertEqual(
            (stock.call_count, quant.call_count), (0, 1) if slot else (1, 0)
        )
        return SimpleNamespace(
            got=got,
            call=call,
            fb=fb,
            stream=stream,
            pending=pending,
            residual=residual,
            aux=aux,
            hop=hop if slot else None,
        )

    def _check(self, *, moe, rows, capture, keeps=False):
        want = self._run(moe=moe, rows=rows, capture=capture, slot=False, keeps=keeps)
        got = self._run(moe=moe, rows=rows, capture=capture, slot=True, keeps=keeps)
        # the same kernel arguments; the quant form adds its three outputs and the
        # entry trigger, off without an early-trigger predicate
        self.assertEqual(len(got.call.args), len(want.call.args))
        for a, b in zip(got.call.args, want.call.args):
            self.assertTrue(_same(a, b), (a, b))
        extra = {"out", "quantized", "scales", "trigger_at_entry"}
        self.assertEqual(set(got.call.kwargs) - extra, set(want.call.kwargs))
        self.assertIs(got.call.kwargs["trigger_at_entry"], False)
        for key, value in want.call.kwargs.items():
            if key == "residual_out":
                continue
            self.assertEqual(got.call.kwargs[key], value, key)
        for run in (want, got):
            # a fresh residual_out, not the residual updated in place
            residual_out = run.call.kwargs["residual_out"]
            self.assertNotEqual(residual_out.data_ptr(), run.residual.data_ptr())
            self.assertEqual(residual_out.shape, run.residual.shape)
            # the stream is written with it and the pending output consumed
            self.assertIs(run.fb.residual_stream, run.stream)
            self.assertIsNone(run.stream.pending)
            self.assertIs(run.stream.residual, residual_out)
            self.assertIsNone(run.pending.value)
            self.assertIsNone(run.pending.owed)
            if capture:
                # one capture of the updated residual: a copy (later layers reuse
                # it), or the residual itself where the following stage keeps it
                self.assertEqual(len(run.aux), 1)
                self.assertTrue(torch.equal(run.aux[0], residual_out))
                if keeps:
                    self.assertIs(run.aux[0], residual_out)
                else:
                    self.assertNotEqual(run.aux[0].data_ptr(), residual_out.data_ptr())
        self.assertTrue(torch.equal(got.stream.residual, want.stream.residual))
        if capture:
            self.assertTrue(torch.equal(got.aux[0], want.aux[0]))
        # the attention gets the MXFP8 pair of the stock kernel's normed row, in the
        # slot's buffers
        self.assertIsInstance(got.got, Mxfp8SwizzledInput)
        data, scales = got.got
        self.assertEqual(data.data_ptr(), got.hop.data.data_ptr())
        self.assertEqual(tuple(data.shape), (rows, H))
        self.assertEqual(data.dtype, torch.float8_e4m3fn)
        self.assertIs(scales, got.hop.scales)
        self.assertEqual(got.call.kwargs["quantized"].data_ptr(), data.data_ptr())
        self.assertEqual(got.call.kwargs["out"].data_ptr(), got.hop.normed.data_ptr())
        self.assertTrue(torch.equal(got.call.kwargs["out"], want.got))
        self.assertTrue(
            torch.equal(data.float(), want.got.float().to(data.dtype).float())
        )

    def test_after_a_moe_layer(self):
        for rows, capture in ((8, False), (8, True), (1, True), (16, False)):
            with self.subTest(rows=rows, capture=capture):
                self._check(moe=True, rows=rows, capture=capture)

    def test_after_a_dense_layer(self):
        for rows, capture in ((8, False), (3, True), (16, True)):
            with self.subTest(rows=rows, capture=capture):
                self._check(moe=False, rows=rows, capture=capture)

    def test_a_capture_the_following_stage_keeps_is_adopted_as_in_stock(self):
        for moe in (True, False):
            with self.subTest(moe=moe):
                self._check(moe=moe, rows=8, capture=True, keeps=True)

    def test_the_executor_hands_the_pair_to_the_stock_attention(self):
        layer = _layer()
        fb, x = _batch(8, _moe_output(8, 3), 3)
        seen = []
        layer.self_attn = lambda positions, hidden_states, forward_batch: (
            seen.append(hidden_states) or "attention partial"
        )
        layer.attn_boundary = SimpleNamespace(
            plan=layer.attn_boundary.plan,
            entry=layer.attn_boundary.entry,
            prepare=Mock(side_effect=AssertionError("stock prepare ran")),
            finish=lambda out, fb: out,
        )
        layer.ffn_boundary = SimpleNamespace(
            prepare=lambda out, fb: out,
            exit=lambda fb: contextlib.nullcontext(SimpleNamespace(finish=lambda o: o)),
        )
        layer.mlp = lambda x, forward_batch: x
        layer.is_layer_sparse = True
        hop, why = boundary.layer_hop(
            SimpleNamespace(
                attn_boundary=SimpleNamespace(plan=layer.attn_boundary.plan),
                self_attn=_mxfp8_projection(),
            ),
            hidden=H,
        )
        self.assertIsNone(why)
        built = boundary.HopMxfp8(
            widths=(8,),
            hops={1: hop._replace(layer=layer)},
            refused={},
            hidden=H,
            device="cpu",
        )
        st = SimpleNamespace(
            fb=fb,
            positions=torch.zeros(8),
            rows=8,
            layer_id=1,
            slots=m31_plan.Slots(hop_mxfp8=True),
            bound={"boundary": built},
            scratch={},
        )
        with tp4(), kernels():
            out = m31_executor.MegakernelDecode._layer(st, layer, x, None)
        self.assertEqual(out, "attention partial")
        (pair,) = seen
        self.assertIsInstance(pair, Mxfp8SwizzledInput)
        self.assertEqual(pair.data.data_ptr(), built.data.data_ptr())
        self.assertEqual(st.scratch[boundary.TALLY], Counter({boundary.TAKEN: 1}))


class TestFallsBackToStockPrepare(CustomTestCase):
    """Where the stock prepare would not run the push-plane kernel, the slot declines
    before touching anything, and attn_input runs the stock prepare."""

    def _declines(
        self, layer, x, fb, why, *, hop=None, layer_id=1, fusable=True, counters=512
    ):
        """The slot declines, having done nothing; ``why`` is the reason the step's
        tally gets (None: a layer the slot was not built for, which it does not
        count)."""
        hop = hop or _hop(layer)
        stream = fb.residual_stream
        pending, residual = stream.pending, stream.residual
        aux = AuxHiddenStateList()
        tally = Counter()
        with tp4(fusable=fusable, counters=counters), kernels() as calls:
            self.assertIsNone(hop.attn_input(layer_id, layer, x, fb, aux, tally))
        self.assertEqual(tally, Counter({why: 1}) if why else Counter())
        self.assertEqual(calls.quant_finalize.call_count, 0)
        self.assertEqual(calls.quant_plain.call_count, 0)
        self.assertIs(fb.residual_stream, stream)
        self.assertIs(stream.pending, pending)
        self.assertIs(stream.residual, residual)
        self.assertEqual(len(aux), 0)

    def test_batches_the_push_plane_kernel_does_not_take(self):
        layer = _layer()
        why = "a batch the push-plane kernel does not take"
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            with self.subTest(mode=mode):
                fb, x = _batch(8, _moe_output(8, 1), 1, mode=mode)
                self._declines(layer, x, fb, why)
        # a plane too small for the rows
        fb, x = _batch(8, _moe_output(8, 1), 1)
        self._declines(layer, x, fb, why, fusable=False)

    def test_outputs_the_push_plane_kernel_does_not_take(self):
        layer = _layer(sparse_before=False)
        other = SimpleNamespace(name="tp", ranks=[0, 1])
        dense = "a partial sum the push-plane kernel does not take"
        for owed, why in (
            (_dense_output(8, 2, dtype=torch.float16), dense),
            (_dense_output(8, 2, group=other), dense),
            (_dense_output(8, 2, group=None, reduce_to_dp_local=lambda p: p), dense),
            (_dense_output(8, 2, reduce_to_dp_local=lambda p: p), dense),
            (
                _moe_output(8, 2, dtype=torch.float16),
                "a MoE handoff the push-plane kernel does not take",
            ),
        ):
            with self.subTest(owed=type(owed).__name__):
                fb, x = _batch(8, owed, 2)
                self._declines(layer, x, fb, why)

    def test_a_declared_sum_or_an_empty_stream(self):
        layer = _layer()
        fb, _ = _batch(8, _moe_output(8, 4), 4)
        stream = ResidualStream(fb.residual_stream.residual)
        x = stream.record(
            torch.zeros(8, H, dtype=torch.bfloat16),
            PLAIN_ADD,
            declared_sum=SumGroup.ATTN_TP,
        )
        fb.residual_stream = stream
        self._declines(
            layer, x, fb, "an output the push-plane kernel does not complete"
        )
        # the first layer's stream: nothing pending, no residual yet
        fb.residual_stream = ResidualStream()
        self._declines(
            layer,
            torch.zeros(8, H, dtype=torch.bfloat16),
            fb,
            "no plain add pending on a written residual",
        )

    def test_rows_past_the_buffers_another_path_or_another_layer(self):
        layer = _layer()
        hop = _hop(layer, widths=(8,))
        fb, x = _batch(16, _moe_output(16, 5), 5)
        self._declines(layer, x, fb, "rows past the slot's buffers", hop=hop)
        fb, x = _batch(8, _moe_output(8, 5), 5)
        # layers the slot was not built for are not counted
        self._declines(_layer(), x, fb, None, hop=hop)
        self._declines(layer, x, fb, None, hop=hop, layer_id=2)
        other = msgspec.structs.replace(
            layer.attn_boundary.plan.paths[BatchVariant.ORDINARY].entry,
            input_move=lambda **kw: kw["hidden_states"],
        )
        with patch.object(layer.attn_boundary, "entry", lambda fb, steps=None: other):
            self._declines(layer, x, fb, "another batch path", hop=hop)

    def test_a_plane_with_no_counter_left_for_the_scale_padding(self):
        """Below 128 rows the MXFP8 form's trailing cluster zeros the scale tile's
        padding rows, which needs a row counter past the rows; the stock kernel also
        runs when the rows take every counter, the MXFP8 form there declines."""
        why = "no push-plane counter left to zero the scale tile's padding"
        for moe in (True, False):
            with self.subTest(moe=moe):
                layer = _layer(sparse_before=moe)
                output = _moe_output if moe else _dense_output
                fb, x = _batch(8, output(8, 9), 9)
                self._declines(layer, x, fb, why, counters=8)
        # no plane registered (the stock fusion declines too)
        with patch.object(moe_quant, "_fused_all_reduce_comm", None):
            self.assertFalse(boundary._pads_scale_tile(8))
        # a counter past the rows, or a full tile with no padding: the slot takes them
        for rows, counters in ((8, 9), (128, 128)):
            with self.subTest(rows=rows, counters=counters):
                layer = _layer()
                fb, x = _batch(rows, _moe_output(rows, 9), 9)
                tally = Counter()
                with tp4(counters=counters), kernels() as calls:
                    got = _hop(layer, widths=(rows,)).attn_input(
                        1, layer, x, fb, None, tally
                    )
                self.assertIsInstance(got, Mxfp8SwizzledInput)
                self.assertEqual(calls.quant_finalize.call_count, 1)
                self.assertEqual(tally, Counter({boundary.TAKEN: 1}))

    def test_pads_scale_tile_matches_the_counter_rule(self):
        """The gate asks the registered plane (can_fuse_all_reduce) for the rows and
        one more, and decides as the counter rule wherever the stock fusion's
        _eligible lets the slot ask, i.e. the plane takes the rows: on M3.1's plane
        (4 MiB slots) at every row count and counter count. On a plane whose slot
        holds fewer rows it may decline more, never take more."""
        compared = Counter()
        for counters in (512, 129, 128, 64):
            with tp4(counters=counters):
                for rows in range(1, 129):
                    if not moe_quant.can_fuse_all_reduce(rows, boundary.HIDDEN):
                        continue
                    compared[counters] += 1
                    with self.subTest(rows=rows, counters=counters):
                        self.assertEqual(
                            boundary._pads_scale_tile(rows), _counter_rule(rows)
                        )
        # every row the plane takes: all 128 rows on 128 counters or more
        self.assertEqual(compared, Counter({512: 128, 129: 128, 128: 128, 64: 64}))
        # slots of 96 rows: the plane takes 1-96 rows, and the gate leaves the last
        # one, which has no room for one more, to the stock prepare
        small = SimpleNamespace(
            world_size=4,
            max_push_size=96 * boundary.HIDDEN * 2,
            config=SimpleNamespace(num_push_blocks=512),
        )
        with tp4(), patch.object(moe_quant, "_fused_all_reduce_comm", small):
            gate = [r for r in range(1, 97) if boundary._pads_scale_tile(r)]
            rule = [r for r in range(1, 97) if _counter_rule(r)]
        self.assertEqual(gate, list(range(1, 96)))
        self.assertEqual(rule, list(range(1, 97)))

    def test_a_full_tile_on_a_short_plane_declines_at_eligible(self):
        """128 rows on a plane of 64 counters: the counter rule would launch (a full
        tile has no padding rows) and the new gate would not, but neither is asked,
        since the stock fusion's _eligible (the real predicate) declines the batch
        first, as it does for the stock prepare."""
        with tp4(counters=64):
            self.assertTrue(_counter_rule(128))
            self.assertFalse(boundary._pads_scale_tile(128))
        layer = _layer()
        fb, x = _batch(128, _moe_output(128, 9), 9)
        self._declines(
            layer,
            x,
            fb,
            "a batch the push-plane kernel does not take",
            hop=_hop(layer, widths=(128,)),
            fusable=None,
            counters=64,
        )

    def test_take_leaves_the_same_stream_on_the_batch(self):
        """The slot writes the batch's residual stream in place and never re-binds
        forward_batch.residual_stream; the stock prepare re-binds it to the same
        object. Either way the batch holds the stream it held before."""
        rebound = []

        class Batch(SimpleNamespace):
            """A forward batch that records the attributes set on it."""

            def __setattr__(self, name, value):
                rebound.append((name, value))
                super().__setattr__(name, value)

        for slot in (True, False):
            with self.subTest(slot=slot):
                layer = _layer()
                fb, x = _batch(8, _moe_output(8, 3), 3)
                stream = fb.residual_stream
                fb = Batch(**vars(fb))
                rebound.clear()
                with tp4(), kernels() as calls:
                    if slot:
                        got = _hop(layer).attn_input(1, layer, x, fb, None)
                        self.assertIsInstance(got, Mxfp8SwizzledInput)
                    else:
                        layer.attn_boundary.prepare(x, fb, capture_gathered=None)
                self.assertIs(fb.residual_stream, stream)
                self.assertEqual(rebound, [] if slot else [("residual_stream", stream)])
                # written in place: the pending output consumed, the kernel's residual
                kernel = calls.quant_finalize if slot else calls.stock_finalize
                self.assertIsNone(stream.pending)
                self.assertIs(stream.residual, kernel.call_args.kwargs["residual_out"])

    def test_an_output_of_another_stream_raises_as_in_stock(self):
        layer = _layer()
        fb, _ = _batch(8, _moe_output(8, 6), 6)
        _, stranger = _batch(8, _moe_output(8, 6), 6)
        with (
            tp4(),
            kernels() as calls,
            self.assertRaisesRegex(
                RuntimeError, "does not belong to this residual stream"
            ),
        ):
            _hop(layer).attn_input(1, layer, stranger, fb, None)
        self.assertEqual(calls.quant_finalize.call_count, 0)

    def test_attn_input_runs_the_stock_prepare_when_declined_or_off(self):
        layer = SimpleNamespace(attn_boundary=SimpleNamespace(prepare=Mock()))
        declining = SimpleNamespace(attn_input=Mock(return_value=None))
        for slots, bound in (
            (m31_plan.Slots(), {"boundary": declining}),
            (m31_plan.Slots(hop_mxfp8=True), {"boundary": declining}),
            (m31_plan.Slots(hop_mxfp8=True), {"boundary": None}),
        ):
            layer.attn_boundary.prepare.reset_mock()
            st = SimpleNamespace(
                fb="fb", layer_id=5, slots=slots, bound=bound, scratch={}
            )
            out = boundary.attn_input(layer, "x", st, "captured")
            self.assertIs(out, layer.attn_boundary.prepare.return_value)
            layer.attn_boundary.prepare.assert_called_once_with(
                "x", "fb", capture_gathered="captured"
            )
            # only a step the slot serves keeps a tally
            self.assertEqual(
                boundary.TALLY in st.scratch,
                bound["boundary"] is declining and slots.hop_mxfp8,
            )
        # with the slot off the build is never asked
        self.assertEqual(declining.attn_input.call_count, 1)
        call = declining.attn_input.call_args
        self.assertEqual(call.args, (5, layer, "x", "fb", "captured", Counter()))
        self.assertEqual(set(call.kwargs), {"early_trigger"})


class TestEarlyTrigger(CustomTestCase):
    """Where the attention's next kernel on the pair is slot attn_ll's low-latency
    q|k|v|index GEMM, the boundary kernel signals its launch at entry: the slot asks
    a predicate on the very pair it hands the attention, before the launch, and
    passes the answer to the kernel (``trigger_at_entry``); everything else is the
    same launch."""

    def _take(self, *, moe, rows, predicate, aux=None, keeps=None):
        """The slot's launch at a layer; ``aux``, the DSpark capture collector of a
        capture layer; ``keeps``, a list: the following stage keeps the residual
        (``_keeping``), the list records what that predicate is asked about."""
        layer = _layer(sparse_before=moe)
        if keeps is not None:
            _keeping(layer, keeps)
        owed = _moe_output(rows, 11) if moe else _dense_output(rows, 11)
        fb, x = _batch(rows, owed, 11)
        hop = _hop(layer, widths=(rows,))
        tally = Counter()
        asked = []
        with tp4(), kernels() as calls:
            quant = calls.quant_finalize if moe else calls.quant_plain

            def early(pair):
                # asked before the kernel launches
                asked.append((pair, quant.call_count))
                return predicate(pair)

            got = hop.attn_input(
                1,
                layer,
                x,
                fb,
                aux,
                tally,
                early_trigger=None if predicate is None else early,
            )
        return SimpleNamespace(
            hop=hop, got=got, call=quant.call_args, asked=asked, tally=tally, aux=aux
        )

    def test_the_kernel_signals_at_entry_where_the_predicate_says(self):
        for moe in (True, False):
            for rows in (1, 8):
                for predicate, early in (
                    (None, False),
                    (lambda pair: False, False),
                    (lambda pair: True, True),
                ):
                    with self.subTest(moe=moe, rows=rows, early=early):
                        run = self._take(moe=moe, rows=rows, predicate=predicate)
                        self.assertIs(run.call.kwargs["trigger_at_entry"], early)
                        self.assertEqual(
                            run.tally,
                            Counter({boundary.TAKEN: 1, boundary.EARLY: int(early)}),
                        )
                        self.assertIsInstance(run.got, Mxfp8SwizzledInput)
                        if predicate is None:
                            self.assertEqual(run.asked, [])
                            continue
                        # asked once, before the launch, with the very pair the
                        # kernel writes and the attention gets
                        ((pair, launched),) = run.asked
                        self.assertEqual(launched, 0)
                        self.assertIs(pair, run.got)
                        self.assertEqual(pair.data.data_ptr(), run.hop.data.data_ptr())
                        self.assertEqual(tuple(pair.data.shape), (rows, H))
                        self.assertIs(pair.scales, run.hop.scales)
                        self.assertIs(run.call.kwargs["quantized"], pair.data)
                        self.assertIs(run.call.kwargs["scales"], pair.scales)

    def test_the_trigger_changes_nothing_else_in_the_launch(self):
        for moe in (True, False):
            with self.subTest(moe=moe):
                late = self._take(moe=moe, rows=8, predicate=lambda p: False)
                early = self._take(moe=moe, rows=8, predicate=lambda p: True)
                self.assertEqual(len(late.call.args), len(early.call.args))
                for a, b in zip(late.call.args, early.call.args):
                    self.assertTrue(_same(a, b))
                self.assertEqual(set(late.call.kwargs), set(early.call.kwargs))
                for key in ("world_size",):
                    self.assertEqual(late.call.kwargs[key], early.call.kwargs[key])
                self.assertTrue(
                    torch.equal(_bits(late.got.data), _bits(early.got.data))
                )

    def test_a_capture_layer_that_copies_the_residual_signals_at_the_end(self):
        """At a DSpark capture layer whose capture copies the updated residual (the
        served verify graph's L4/18/32/46: no following stage certifies that it keeps
        the residual), the copy runs between the kernel and the GEMM and the GEMM
        cannot start early: the kernel signals at its end, the predicate is not asked
        and the tally counts no early trigger."""
        for moe in (True, False):
            with self.subTest(moe=moe):
                aux = AuxHiddenStateList()
                run = self._take(moe=moe, rows=8, predicate=lambda p: True, aux=aux)
                self.assertIs(run.call.kwargs["trigger_at_entry"], False)
                self.assertEqual(run.asked, [])
                self.assertEqual(run.tally, Counter({boundary.TAKEN: 1}))
                self.assertIsInstance(run.got, Mxfp8SwizzledInput)
                (copy,) = aux
                residual_out = run.call.kwargs["residual_out"]
                self.assertTrue(torch.equal(copy, residual_out))
                self.assertNotEqual(copy.data_ptr(), residual_out.data_ptr())

    def test_a_capture_layer_that_adopts_the_residual_asks_as_any_other(self):
        """Where the following stage keeps the residual, the executor's capture list
        adopts the updated residual as it is, so nothing runs between the kernel and
        the GEMM: the predicate decides as at any other layer. The stage's predicate
        is asked once, about the residual_out the kernel then writes. A packer copies
        even what it owns: no early trigger there."""
        for answer in (True, False):
            with self.subTest(answer=answer):
                aux, keeps = AuxHiddenStateList(), []
                run = self._take(
                    moe=True, rows=8, predicate=lambda p: answer, aux=aux, keeps=keeps
                )
                residual_out = run.call.kwargs["residual_out"]
                self.assertIs(run.call.kwargs["trigger_at_entry"], answer)
                self.assertEqual(len(run.asked), 1)
                self.assertEqual(
                    run.tally,
                    Counter({boundary.TAKEN: 1, boundary.EARLY: int(answer)}),
                )
                (asked,) = keeps
                self.assertIs(asked, residual_out)
                (adopted,) = aux
                self.assertIs(adopted, residual_out)
        packer, keeps = AuxHiddenStatePacker(1), []
        run = self._take(
            moe=True, rows=8, predicate=lambda p: True, aux=packer, keeps=keeps
        )
        self.assertIs(run.call.kwargs["trigger_at_entry"], False)
        self.assertEqual(run.asked, [])
        self.assertEqual(run.tally, Counter({boundary.TAKEN: 1}))
        self.assertEqual(len(keeps), 1)
        self.assertTrue(torch.equal(packer.finalize(), run.call.kwargs["residual_out"]))

    def test_whether_a_capture_copies(self):
        self.assertFalse(boundary._capture_copies(None, False))
        self.assertTrue(boundary._capture_copies(AuxHiddenStateList(), False))
        self.assertFalse(boundary._capture_copies(AuxHiddenStateList(), True))
        for owned in (False, True):
            self.assertTrue(boundary._capture_copies(AuxHiddenStatePacker(1), owned))

    def test_a_declined_layer_asks_nothing(self):
        layer = _layer()
        hop = _hop(layer, widths=(8,))
        fb, x = _batch(16, _moe_output(16, 12), 12)
        asked = []
        with tp4(), kernels() as calls:
            got = hop.attn_input(
                1, layer, x, fb, None, Counter(), early_trigger=asked.append
            )
        self.assertIsNone(got)
        self.assertEqual(asked, [])
        self.assertEqual(calls.quant_finalize.call_count, 0)

    def test_attn_input_asks_the_attention_step_only_where_attn_ll_serves(self):
        from sgl_nvidia_mega.minimax_m31 import attention

        for attn_ll, answer in ((True, True), (True, False), (False, True)):
            with self.subTest(attn_ll=attn_ll, answer=answer):
                layer = _layer()
                fb, x = _batch(8, _moe_output(8, 13), 13)
                built = _hop(layer, layer_id=1, widths=(8,))
                st = SimpleNamespace(
                    fb=fb,
                    layer_id=1,
                    rows=8,
                    slots=m31_plan.Slots(hop_mxfp8=True, attn_ll=attn_ll),
                    bound={"boundary": built, "attention": "attention build"},
                    scratch={},
                )
                with (
                    tp4(),
                    kernels() as calls,
                    patch.object(
                        attention, "low_latency_qkv", return_value=answer
                    ) as ask,
                ):
                    pair = boundary.attn_input(layer, x, st, None)
                self.assertIsInstance(pair, Mxfp8SwizzledInput)
                if attn_ll:
                    ask.assert_called_once_with(layer, pair, st)
                else:
                    ask.assert_not_called()
                early = attn_ll and answer
                kw = calls.quant_finalize.call_args.kwargs
                self.assertIs(kw["trigger_at_entry"], early)
                self.assertEqual(
                    st.scratch[boundary.TALLY],
                    Counter({boundary.TAKEN: 1, boundary.EARLY: int(early)}),
                )


class TestDeclineReport(CustomTestCase):
    """The first step of each width logs the layers the slot left to the stock prepare
    and why (a warning), and once every served width has run, the widths at which it
    took every layer it was built for; final() hands it the step's tally."""

    def test_a_line_per_width_and_one_summary(self):
        built = boundary.HopMxfp8(
            widths=(16, 8, 24),
            hops={1: None, 2: None, 3: None},
            refused={0: "the first layer"},
            hidden=H,
            device="cpu",
        )
        with self.assertLogs(boundary.logger, level="INFO") as logs:
            built.report(8, Counter({boundary.TAKEN: 3}))
            built.report(16, Counter({boundary.TAKEN: 1, "another batch path": 2}))
            # a later step of a width already logged, a width the slot does not serve
            built.report(16, Counter({"another batch path": 3}))
            built.report(12, Counter({"another batch path": 3}))
            built.report(24, Counter({boundary.TAKEN: 3}))
        warning, summary = logs.records
        self.assertEqual(warning.levelname, "WARNING")
        self.assertIn(
            "a step of 16 rows took 1 of the 3 attention inputs", warning.getMessage()
        )
        self.assertIn("(another batch path: 2 layers)", warning.getMessage())
        self.assertEqual(summary.levelname, "INFO")
        self.assertIn(
            "took all 3 attention inputs it was built for at rows [8, 24]; fewer at "
            "rows [16]",
            summary.getMessage(),
        )
        self.assertEqual(built.reported, {8: 3, 16: 1, 24: 3})
        with self.assertNoLogs(boundary.logger):
            built.report(24, Counter({"another batch path": 3}))

    def test_early_triggers_are_not_declines_and_the_summary_counts_them(self):
        built = boundary.HopMxfp8(
            widths=(1, 8, 16),
            hops={1: None, 2: None, 3: None},
            refused={},
            hidden=H,
            device="cpu",
        )
        with self.assertLogs(boundary.logger, level="INFO") as logs:
            built.report(1, Counter({boundary.TAKEN: 3, boundary.EARLY: 3}))
            built.report(8, Counter({boundary.TAKEN: 3, boundary.EARLY: 2}))
            built.report(16, Counter({boundary.TAKEN: 3}))
        (summary,) = logs.records
        self.assertEqual(summary.levelname, "INFO")
        self.assertIn("at rows [1, 8, 16]", summary.getMessage())
        self.assertIn(
            "q|k|v|index GEMM at entry at {1: 3, 8: 2} (rows: layers)",
            summary.getMessage(),
        )
        self.assertEqual(built.early, {1: 3, 8: 2, 16: 0})

    def test_final_hands_over_the_steps_tally(self):
        built = Mock()
        tally = Counter({boundary.TAKEN: 2, "another batch path": 1})
        model = SimpleNamespace(
            norm="norm", finalize_norm="finalize", end_layer=3, layers_to_capture=[]
        )
        st = SimpleNamespace(
            scratch={boundary.TALLY: tally},
            bound={"boundary": built},
            rows=8,
            model=model,
            fb="fb",
        )
        aux = SimpleNamespace(capture="capture")
        with patch.object(residual_batch, "final_norm", return_value="normed") as norm:
            self.assertEqual(boundary.final(st, "x", aux), "normed")
        built.report.assert_called_once_with(8, tally)
        norm.assert_called_once_with(
            "x", "fb", "norm", finalize_norm="finalize", capture=None, skip_empty=True
        )
        # a step the slot does not serve keeps no tally
        st.scratch = {}
        with patch.object(residual_batch, "final_norm", return_value="normed"):
            boundary.final(st, "x", aux)
        built.report.assert_called_once()


class TestLayersAndWidths(CustomTestCase):
    def test_layers_whose_input_the_slot_takes(self):
        self.assertIsNone(boundary.layer_hop(_layer(), hidden=H)[1])
        self.assertIsNone(boundary.layer_hop(_layer(sparse_before=False), hidden=H)[1])
        cases = {
            "first": _layer(first=True),
            "no fusion": _layer(fusions=False),
            "hidden": None,
            "fp16 norm": _layer(norm=_norm(torch.float16)),
            "no projection": _layer(attn=SimpleNamespace(_fused_qkv_index=None)),
            "deep_gemm projection": _layer(attn=_mxfp8_projection("deep_gemm")),
            "unquantized projection": _layer(attn=_mxfp8_projection(use_mxfp8=False)),
            "projection width": _layer(attn=_mxfp8_projection(k=2 * H)),
        }
        for name, layer in cases.items():
            with self.subTest(name):
                hidden = 2 * H if name == "hidden" else H
                hop, why = boundary.layer_hop(layer or _layer(), hidden=hidden)
                self.assertIsNone(hop)
                self.assertTrue(why)

        def prepare_with(entry, **keywords):
            # the layer's own entry with its prepare's keywords changed
            own = entry.prepare
            return dict(
                prepare=partial(_run_entry, **{**own.keywords, **keywords}),
            )

        def reordered(entry):
            carried = entry.prepare.keywords["carried_fusions"]
            return prepare_with(entry, carried_fusions=(carried[2], *carried[:2]))

        for name, change, why_part in (
            (
                "input move",
                lambda e: dict(input_move=lambda **kw: kw["hidden_states"]),
                "moves rows",
            ),
            ("declared sum", lambda e: dict(declared_sum=SumGroup.ATTN_TP), "sum"),
            (
                "adapter",
                lambda e: dict(attn_input_adapter=lambda h, fb, f: h),
                "moves rows",
            ),
            ("fusion order", reordered, "first fused completion"),
            (
                "update capabilities",
                lambda e: dict(prepare=partial(lambda *a, **k: None, paths={})),
                "first fused completion",
            ),
            (
                "carried read fusion",
                lambda e: prepare_with(e, carried_read_fusions=(print,)),
                "read carries fusions or binds per batch",
            ),
            (
                "read bound per batch",
                lambda e: prepare_with(e, read=SimpleNamespace(bind=print)),
                "read carries fusions or binds per batch",
            ),
        ):
            with self.subTest(name):
                layer = _layer()
                path = layer.attn_boundary.plan.paths[BatchVariant.ORDINARY]
                layer.attn_boundary.plan.paths[BatchVariant.ORDINARY] = (
                    msgspec.structs.replace(
                        path,
                        entry=msgspec.structs.replace(path.entry, **change(path.entry)),
                    )
                )
                hop, why = boundary.layer_hop(layer, hidden=H)
                self.assertIsNone(hop)
                self.assertIn(why_part, why)
        layer = _layer()
        layer.attn_boundary.plan.qkv_latent_func = print
        self.assertIsNone(boundary.layer_hop(layer, hidden=H)[0])

    def _model(self, layers):
        return SimpleNamespace(
            model=SimpleNamespace(layers=layers, start_layer=0, end_layer=len(layers))
        )

    def test_build_takes_every_layer_but_the_first_at_every_width(self):
        self.assertIsNone(boundary.build(None, (8, 16), frozenset()))
        self.assertFalse(boundary.serves(None, 8))
        layers = [_layer(first=True), _layer(sparse_before=False), _layer()]
        widths = tuple(range(8, 129, 8))
        with (
            tp4(),
            patch.object(boundary, "HIDDEN", H),
            patch.object(boundary, "layer_hop", partial(boundary.layer_hop, hidden=H)),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(ops, "load_residual_norm_quant") as load,
        ):
            built = boundary.build(
                self._model(layers), widths, frozenset({"hop_mxfp8"})
            )
        self.assertEqual(sorted(built.hops), [1, 2])
        self.assertEqual(list(built.refused), [0])
        self.assertEqual(built.widths, widths)
        self.assertEqual(built.max_rows, 128)
        self.assertEqual(
            (built.data.shape, built.data.dtype), ((128, H), torch.float8_e4m3fn)
        )
        self.assertEqual(
            (built.normed.shape, built.normed.dtype), ((128, H), torch.bfloat16)
        )
        self.assertEqual(
            (built.scales.shape, built.scales.dtype),
            ((ops.mxfp8_scale_bytes(H),), torch.uint8),
        )
        self.assertTrue(all(boundary.serves(built, w) for w in widths))
        self.assertFalse(boundary.serves(built, 12))
        # each width's plain and finalize kernels are compiled at build
        self.assertEqual(load.call_count, 2 * len(widths))
        load.assert_any_call(4, 8, H)
        load.assert_any_call(4, 128, H, top_k=TOP_K, weight_dtype=torch.float32)

    def test_a_model_with_no_layer_to_take_serves_nothing(self):
        with (
            tp4(),
            patch.object(boundary, "layer_hop", partial(boundary.layer_hop, hidden=H)),
        ):
            built = boundary.build(
                self._model([_layer(first=True), _layer(fusions=False)]),
                (8, 16),
                frozenset({"hop_mxfp8"}),
            )
        self.assertEqual(built.hops, {})
        self.assertFalse(boundary.serves(built, 8))
        self.assertEqual(sorted(built.refused), [0, 1])

    def test_the_plan_marks_the_widths_the_slot_serves(self):
        served = SimpleNamespace(serves=lambda rows: rows <= 64)
        ctx = SimpleNamespace(
            widths=tuple(range(8, 129, 8)), tokens_per_request=8, binding=None
        )
        with (
            patch.object(m31_runtime, "SLOTS", frozenset({"hop_mxfp8"})),
            patch.object(boundary, "build", return_value=served),
            patch.object(m31_agree, "gather_tp", lambda obj: [obj]),
        ):
            built = m31_runtime.build(SimpleNamespace(), ctx)
            self.assertEqual(
                [w for w, s in built.slot_plan.items() if s.hop_mxfp8],
                list(range(8, 65, 8)),
            )
            self.assertIn("hop_mxfp8 at", built.summary())
        nothing = SimpleNamespace(serves=lambda rows: False)
        with (
            patch.object(m31_runtime, "SLOTS", frozenset({"hop_mxfp8"})),
            patch.object(boundary, "build", return_value=nothing),
            patch.object(m31_agree, "gather_tp", lambda obj: [obj]),
            self.assertRaisesRegex(RuntimeError, "serve no width"),
        ):
            m31_runtime.build(SimpleNamespace(), ctx)


class TestStockAttentionTakesThePair(CustomTestCase):
    """MiniMaxM3Attention's projection takes the MXFP8 pair as its input and runs the
    same GEMM call as on the bf16 row, without the quantize."""

    def test_forward_prepare_reads_the_input_only_through_the_projection(self):
        from sglang.srt.models.minimax_m3 import MiniMaxM3Attention

        rows, heads, head_dim = 8, 16, 128
        fused = torch.randn(rows, 2560).bfloat16()
        proj = Mock(return_value=fused)
        attn = SimpleNamespace(
            _fused_qkv_index=object(),
            fused_qkv_index_proj=proj,
            _fused_main_size=2304,
            _aiter_sparse_qknorm_cache=lambda *a: None,
            _combined_qknorm_ok=True,
            _fp8_q_epilogue_ok=True,
            _qknorm_groups=lambda: [],
            rotary_emb=SimpleNamespace(cos_sin_cache=torch.zeros(4)),
            q_norm=SimpleNamespace(variance_epsilon=1e-6),
            num_heads=heads,
            head_dim=head_dim,
            num_idx_heads=1,
            idx_head_dim=128,
            disable_index_value=True,
            q_size=heads * head_dim,
            kv_size=128,
        )
        attn._split_index_qkv = MiniMaxM3Attention._split_index_qkv.__get__(attn)
        pair = Mxfp8SwizzledInput(
            torch.zeros(rows, 6144, dtype=torch.float8_e4m3fn),
            torch.zeros(24576, dtype=torch.uint8),
        )
        with patch(
            "sglang.kernels.ops.attention.minimax_qknorm_rope."
            "minimax_qknorm_rope_grouped_fp8"
        ) as qknorm:
            _, _, inner = MiniMaxM3Attention.forward_prepare(
                attn, torch.zeros(rows, dtype=torch.long), pair, "fb"
            )
        proj.assert_called_once_with(pair)
        qknorm.assert_called_once()
        self.assertIs(qknorm.call_args.args[0], fused)
        self.assertEqual(inner[-1], "fb")

    def test_the_projection_and_its_linear_method_pass_the_pair_to_the_gemm(self):
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
        from sglang.srt.models.minimax_m3 import _FusedQKVIndexProj

        qm = Mock(convert_mxfp8_to_block=False)
        proj = _FusedQKVIndexProj(
            qm,
            torch.zeros(40, H, dtype=torch.float8_e4m3fn),
            torch.zeros(40, H // 32, dtype=torch.uint8),
            H,
            [32, 8],
            torch.bfloat16,
        )
        pair = Mxfp8SwizzledInput(
            torch.zeros(8, H, dtype=torch.float8_e4m3fn),
            torch.zeros(ops.mxfp8_scale_bytes(H), dtype=torch.uint8),
        )
        self.assertIs(proj(pair), qm.apply.return_value)
        qm.apply.assert_called_once_with(proj, pair, None)

        method = Fp8LinearMethod.__new__(Fp8LinearMethod)
        method.use_marlin = False
        method.use_mxfp8 = True
        method.block_fp8_as_mxfp8 = False
        method.mxfp8_prefill_autotune_min_tokens = None
        method.mxfp8_dense_backend = SimpleNamespace(
            is_flashinfer_cutlass=lambda: False,
            is_flashinfer_cutedsl=lambda: True,
            is_gfx95=lambda: False,
        )
        method.w8a8_mxfp8_linear = Mock()
        layer = SimpleNamespace(weight=proj.weight, weight_scale_inv_swizzled="sf")
        self.assertIs(method.apply(layer, pair), method.w8a8_mxfp8_linear.return_value)
        method.w8a8_mxfp8_linear.assert_called_once_with(
            input=pair.data,
            weight=layer.weight,
            weight_scale="sf",
            input_scale=pair.scales,
            bias=None,
        )

    def test_the_gemm_call_is_the_bf16_rows_less_its_quantize(self):
        from sglang.srt.layers.quantization import fp8_utils

        rows = 8
        x = torch.randn(rows, H).bfloat16()
        q = torch.zeros(rows, H, dtype=torch.float8_e4m3fn)
        sf = torch.zeros(ops.mxfp8_scale_bytes(H), dtype=torch.uint8)
        weight = torch.zeros(40, H, dtype=torch.float8_e4m3fn)
        wsf = torch.zeros(160, dtype=torch.uint8)
        gemm = Mock(side_effect=lambda a, b, *r, **k: torch.zeros(a.shape[0], 40))
        quantize = Mock(return_value=(q, sf))
        with (
            patch.object(fp8_utils, "flashinfer_mm_mxfp8", gemm, create=True),
            patch.object(fp8_utils, "flashinfer_mxfp8_quantize", quantize, create=True),
        ):
            stock = fp8_utils.flashinfer_mxfp8_blockscaled_linear(
                x, weight, wsf, backend="cute-dsl"
            )
            paired = fp8_utils.flashinfer_mxfp8_blockscaled_linear(
                q, weight, wsf, input_scale=sf, backend="cute-dsl"
            )
        # the bf16 row is quantized once, as FlashInfer's 128x4 MXFP8 ...
        quantize.assert_called_once()
        self.assertEqual(quantize.call_args.args[0].data_ptr(), x.data_ptr())
        self.assertEqual(
            quantize.call_args.kwargs, dict(is_sf_swizzled_layout=True, alignment=32)
        )
        # ... and the GEMM gets the same arguments from it as from the pair
        (stock_call, paired_call) = gemm.call_args_list
        self.assertEqual(len(stock_call.args), len(paired_call.args))
        for a, b in zip(stock_call.args, paired_call.args):
            self.assertTrue(_same(a, b) and a.data_ptr() == b.data_ptr())
        self.assertEqual(paired_call.args[0].data_ptr(), q.data_ptr())
        self.assertIs(paired_call.args[2], sf)
        self.assertEqual(stock_call.kwargs, paired_call.kwargs)
        self.assertEqual(paired_call.kwargs["out_dtype"], torch.bfloat16)
        self.assertEqual((stock.dtype, paired.dtype), (torch.bfloat16, torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
