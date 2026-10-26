# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of the M3.1 megakernel's slot hop_ffn (the FFN-input boundary kernel that
also writes the FFN's quantized inputs): its FFN input against SGLang's own
``ffn_boundary.prepare`` on stage boundaries built by SGLang's factories (the kernels
faked alike), its fallbacks to the stock prepare and the per-width log of them, the
layers and widths it takes and its build (the NVFP4 probe agreed on every rank), the
FFN step handing the MXFP8 pair to the gate_up and the NVFP4 to moe_nofill's routed
experts, the stock modules taking the pair, the probe rows telling the two subnormal
modes, the block scale's associations and rcp(6) one ulp off apart at the probe's
global scales (a model of the arithmetic), and the probe on faked kernels. The quant
ops' plumbing and arch gate and the NVFP4 epilogue's index arithmetic and reduction
order are SGLang's tests, with the ops. The kernels are checked on GPU
(test/kernels/test_minimax_m31_hop_ffn.py)."""

import contextlib
import sys
import unittest
from collections import Counter
from fractions import Fraction
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import numpy as np
import pytest
import torch

pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

# the stage-boundary context and the MoE fakes of the sibling slots' CPU tests
import test_minimax_m31_hop_mxfp8_slot as hop_tests  # noqa: E402
import test_minimax_m31_moe_nofill_slot as nofill_tests  # noqa: E402
from sglang.kernels.ops.communication import all_reduce_fusion as ops  # noqa: E402
from sglang.srt import runtime_context  # noqa: E402
from sglang.srt.layers import linear as linear_mod  # noqa: E402
from sglang.srt.layers.layer_boundary import PLAIN_ADD, UnreducedOutput  # noqa: E402
from sglang.srt.layers.layer_boundary.contracts import BatchVariant  # noqa: E402
from sglang.srt.layers.layer_boundary.fusions import (  # noqa: E402
    custom_all_reduce_fusion as custom,
)
from sglang.srt.layers.layer_boundary.prepare import _run_entry  # noqa: E402
from sglang.srt.layers.layer_boundary.residual import (  # noqa: E402
    batch as residual_batch,
)
from sglang.srt.layers.layer_boundary.residual.stream import (  # noqa: E402
    ResidualStream,
)
from sglang.srt.layers.linear import (  # noqa: E402
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)
from sglang.srt.layers.quantization import fp8_utils  # noqa: E402
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod  # noqa: E402
from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.models.minimax_m3 import MiniMaxM3MLP, MiniMaxM3MoE  # noqa: E402
from sglang.test.boundary_fixtures import make_test_stages  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

from sgl_nvidia_mega.minimax_m31 import boundary  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import moe as m31_moe  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import plan as m31_plan  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import runtime as m31_runtime  # noqa: E402
from sgl_nvidia_mega.minimax_m31.boundary import hop_ffn  # noqa: E402
from sgl_nvidia_mega.minimax_m31.moe import moe_nofill as mn  # noqa: E402
from sgl_nvidia_mega.minimax_m31.runtime import agree as m31_agree  # noqa: E402
from sgl_nvidia_mega.minimax_m31.runtime import executor as m31_executor  # noqa: E402

if not hasattr(ops, "all_reduce_residual_norm_quant_nvfp4"):
    pytest.skip(
        "this SGLang has no NVFP4 epilogue of its push-plane boundary kernel",
        allow_module_level=True,
    )

H = hop_tests.H  # a hidden width the CPU fakes can carry; the slot's kernel sees 6144
SF = H // 16
tp4 = hop_tests.tp4


def _layer(*, moe=True, fusions=True, norm=None, mlp=None, layer_id=1):
    """One decoder layer's stage boundaries from SGLang's factories, the push-plane
    fusion on both stages as MiniMaxM3DecoderLayer installs it; the FFN a MoE (layers
    3-59) or the dense MLP (layers 0-2)."""
    fusion = (
        custom.CustomAllReduceFusion(hidden_size=H, defers_finalize=moe)
        if fusions
        else None
    )
    with tp4():
        stages = make_test_stages(
            attention_norm=hop_tests._norm(),
            ffn_norm=norm or hop_tests._norm(),
            sparse=moe,
            previous_sparse=True,
            **({"fusions": fusion} if fusions else {}),
        )
    return SimpleNamespace(
        attn_boundary=stages.attn,
        ffn_boundary=stages.ffn,
        is_layer_sparse=moe,
        mlp=mlp,
        layer_id=layer_id,
    )


def _hop(layer, *, layer_id=1, widths=(8, 16), ftz=False):
    hop, why = hop_ffn.ffn_hop(layer, hidden=H)
    assert hop is not None, why
    return hop_ffn.HopFfn(
        widths=widths,
        hops={layer_id: hop},
        refused={},
        hidden=H,
        device="cpu",
        ftz=ftz,
    )


def _batch(layer, rows, seed, *, mode=ForwardMode.TARGET_VERIFY, dtype=torch.bfloat16):
    """A forward batch whose stream holds a written residual and the attention's
    partial sum as ``attn_boundary.finish`` records it; returns the batch and the
    model-facing handle of that output."""
    g = torch.Generator().manual_seed(seed)
    fb = SimpleNamespace(
        forward_mode=mode,
        can_run_tbo=False,
        input_ids=torch.zeros(rows, dtype=torch.long),
        attn_cp_metadata=None,
        residual_stream=ResidualStream(torch.randn(rows, H, generator=g).bfloat16()),
    )
    partial_sum = torch.randn(rows, H, generator=g).to(dtype)
    with tp4():
        out = layer.attn_boundary.finish(partial_sum, fb)
    return fb, out


def _normed(residual):
    return (residual.float() * 3 - 1).bfloat16()


def _stock_kernel(input, residual, gamma, eps, **kw):
    """SGLang's push-plane kernel, faked: a fresh normed row, the updated residual
    into ``residual_out``."""
    kw["residual_out"].copy_(residual + 1)
    return _normed(residual), kw["residual_out"]


def _quant_kernel(input, residual, gamma, eps, **kw):
    """Its MXFP8 form, faked alike, writing the codes and scales it was given (the
    op allocates the normed row)."""
    normed, residual_out = _stock_kernel(input, residual, gamma, eps, **kw)
    kw["quantized"].copy_(normed.float().to(torch.float8_e4m3fn))
    kw["scales"].fill_(7)
    return normed, residual_out, kw["quantized"], kw["scales"]


def _nvfp4_kernel(input, residual, gamma, eps, global_scale_inv, **kw):
    """Its MXFP8 + NVFP4 form, faked alike."""
    normed, residual_out, data, scales = _quant_kernel(
        input, residual, gamma, eps, **kw
    )
    kw["fp4"].fill_(3)
    kw["fp4_scales"].fill_(5)
    kw["token_scales"].copy_(normed.float().abs().amax(1) * global_scale_inv)
    return (
        normed,
        residual_out,
        data,
        scales,
        kw["fp4"],
        kw["fp4_scales"],
        kw["token_scales"],
    )


@contextlib.contextmanager
def kernels(config=""):
    """The stock kernel (as CustomAllReduceFusion calls it) and the quant ops (as the
    slot calls them), faked and recorded; ``config`` what ``config_reason`` says."""
    calls = SimpleNamespace(
        stock=Mock(side_effect=_stock_kernel),
        quant=Mock(side_effect=_quant_kernel),
        nvfp4=Mock(side_effect=_nvfp4_kernel),
    )
    with (
        patch.object(custom, "all_reduce_residual_norm", calls.stock),
        patch.object(ops, "all_reduce_residual_norm_quant", calls.quant),
        patch.object(ops, "all_reduce_residual_norm_quant_nvfp4", calls.nvfp4),
        patch.object(hop_ffn, "config_reason", lambda: config),
    ):
        yield calls


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1).view(torch.uint8)


class TestFfnInputMatchesStockPrepare(CustomTestCase):
    """The slot's FFN input makes the stock kernel call in its quantizing form and
    leaves the residual stream as ffn_boundary.prepare does."""

    def _run(self, *, moe, rows, slot, ftz=False, config=""):
        layer = _layer(moe=moe)
        fb, out = _batch(layer, rows, 7)
        stream, pending = fb.residual_stream, fb.residual_stream.pending
        residual, partial_sum = stream.residual, pending.value
        tally = Counter()
        hop = None
        with tp4(), kernels(config) as calls:
            if slot:
                hop = _hop(layer, widths=(rows,), ftz=ftz)
                got = hop.ffn_input(1, layer, out, fb, tally)
            else:
                got = layer.ffn_boundary.prepare(out, fb)
        return SimpleNamespace(
            got=got,
            calls=calls,
            fb=fb,
            stream=stream,
            pending=pending,
            residual=residual,
            partial=partial_sum,
            hop=hop,
            tally=tally,
        )

    def _check(self, *, moe, rows, ftz=False, config="", nvfp4=True):
        want = self._run(moe=moe, rows=rows, slot=False)
        got = self._run(moe=moe, rows=rows, slot=True, ftz=ftz, config=config)
        (stock,) = want.calls.stock.call_args_list
        launched = got.calls.nvfp4 if nvfp4 else got.calls.quant
        other = got.calls.quant if nvfp4 else got.calls.nvfp4
        self.assertEqual(
            (got.calls.stock.call_count, launched.call_count, other.call_count),
            (0, 1, 0),
        )
        (call,) = launched.call_args_list
        # the stock arguments: the attention's partial, the residual, the gamma, eps
        self.assertIs(stock.args[0], want.partial)
        self.assertIs(call.args[0], got.partial)
        for a, b in zip(call.args[1:4], stock.args[1:4]):
            self.assertTrue(hop_tests._same(a, b), (a, b))
        self.assertEqual(call.kwargs["world_size"], stock.kwargs["world_size"])
        extra = {"quantized", "scales"}
        if nvfp4:
            extra |= {"ftz", "fp4", "fp4_scales", "token_scales"}
            self.assertIs(call.args[4], got.hop.global_scale_inv[mn.e4m3_max()])
            self.assertIs(call.kwargs["ftz"], ftz)
            self.assertEqual(len(call.args), 5)
        else:
            self.assertEqual(len(call.args), 4)
        self.assertEqual(set(call.kwargs) - extra, set(stock.kwargs))
        for run, kw in ((want, stock.kwargs), (got, call.kwargs)):
            # a fresh residual_out, not the residual updated in place
            residual_out = kw["residual_out"]
            self.assertNotEqual(residual_out.data_ptr(), run.residual.data_ptr())
            self.assertEqual(residual_out.shape, run.residual.shape)
            # the stream is written with it and the pending output consumed
            self.assertIs(run.fb.residual_stream, run.stream)
            self.assertIsNone(run.stream.pending)
            self.assertIs(run.stream.residual, residual_out)
            self.assertIsNone(run.pending.value)
            self.assertIsNone(run.pending.owed)
        self.assertTrue(torch.equal(got.stream.residual, want.stream.residual))
        # the FFN input: the stock normed rows, their pair in the slot's buffers
        prepared, hop = got.got, got.hop
        self.assertIsInstance(prepared, hop_ffn.FfnInput)
        self.assertTrue(torch.equal(prepared.x, want.got))
        self.assertIsInstance(prepared.mxfp8, Mxfp8SwizzledInput)
        data, scales = prepared.mxfp8
        self.assertEqual(data.data_ptr(), hop.data.data_ptr())
        self.assertEqual((tuple(data.shape), data.dtype), ((rows, H), hop.data.dtype))
        self.assertIs(scales, hop.scales)
        self.assertEqual(call.kwargs["quantized"].data_ptr(), data.data_ptr())
        if not nvfp4:
            self.assertIsNone(prepared.nvfp4)
            return got
        # the NVFP4 as PerTokenQuant hands it to the routed experts, in the buffers
        fp4, hs_scale, token_scales = prepared.nvfp4
        self.assertEqual(
            (fp4.shape, hs_scale.shape, hs_scale.dtype, token_scales.shape),
            ((rows, H // 2), (rows, SF), torch.float8_e4m3fn, (rows,)),
        )
        self.assertEqual(fp4.data_ptr(), hop.fp4.data_ptr())
        self.assertEqual(hs_scale.data_ptr(), hop.fp4_scales.data_ptr())
        self.assertEqual(token_scales.data_ptr(), hop.token_scales.data_ptr())
        self.assertEqual(call.kwargs["fp4_scales"].shape, (rows * SF,))
        self.assertTrue(torch.equal(_bits(hs_scale), torch.full((rows * SF,), 5)))
        return got

    def test_a_moe_layer_gets_the_mxfp8_pair_and_the_nvfp4(self):
        for rows, ftz in ((8, False), (1, True), (16, False)):
            with self.subTest(rows=rows, ftz=ftz):
                got = self._check(moe=True, rows=rows, ftz=ftz)
                self.assertEqual(got.tally, Counter({hop_ffn.TAKEN: 1}))

    def test_a_dense_layer_gets_the_mxfp8_pair(self):
        for rows in (8, 3, 16):
            with self.subTest(rows=rows):
                got = self._check(moe=False, rows=rows, nvfp4=False)
                self.assertEqual(got.tally, Counter({hop_ffn.TAKEN: 1}))
                self.assertEqual(got.hop.fp4.numel(), 0)

    def test_without_the_nvfp4_a_moe_layer_gets_the_pair_only(self):
        # the build found no bit-for-bit NVFP4
        got = self._check(moe=True, rows=8, ftz=None, nvfp4=False)
        self.assertEqual(got.tally, Counter({hop_ffn.TAKEN: 1}))
        self.assertFalse(got.hop.nvfp4)
        # FlashInfer's configuration changed since the build
        why = "FLASHINFER_NVFP4_4OVER6 selects the 4over6 quantize"
        got = self._check(moe=True, rows=8, config=why, nvfp4=False)
        self.assertEqual(
            got.tally, Counter({f"{hop_ffn.TAKEN} without the NVFP4 ({why})": 1})
        )

    def test_the_executor_hands_the_ffn_input_to_the_ffn(self):
        layer = _layer(moe=False)
        seen = []
        layer.mlp = lambda x, forward_batch: seen.append(x) or "dense partial"
        fb, _ = _batch(layer, 8, 3)
        stream = fb.residual_stream
        fb.residual_stream = ResidualStream(stream.residual)
        hop = _hop(layer, widths=(8,))
        st = SimpleNamespace(
            fb=fb,
            positions=torch.zeros(8),
            rows=8,
            layer_id=1,
            slots=m31_plan.Slots(hop_ffn=True),
            bound={"hop_ffn": hop},
            scratch={},
        )
        partial_sum = torch.randn(8, H).bfloat16()
        exits = []

        @contextlib.contextmanager
        def ffn_exit(forward_batch):
            exit_ = SimpleNamespace(finish=lambda o: exits.append(o) or o)
            yield exit_

        attn_boundary = layer.attn_boundary
        layer.attn_boundary = SimpleNamespace(
            prepare=lambda x, fb, capture_gathered=None: "attention input",
            finish=attn_boundary.finish,
        )
        layer.self_attn = lambda positions, hidden_states, forward_batch: partial_sum
        ffn_boundary = layer.ffn_boundary
        layer.ffn_boundary = SimpleNamespace(
            plan=ffn_boundary.plan,
            entry=ffn_boundary.entry,
            prepare=Mock(side_effect=AssertionError("stock prepare ran")),
            exit=ffn_exit,
        )
        with (
            tp4(),
            kernels(),
            patch.object(hop_ffn, "mlp_input", lambda mlp, prepared: prepared.mxfp8),
        ):
            out = m31_executor.MegakernelDecode._layer(st, layer, "x", None)
        self.assertEqual(out, "dense partial")
        (pair,) = seen
        self.assertIsInstance(pair, Mxfp8SwizzledInput)
        self.assertEqual(pair.data.data_ptr(), hop.data.data_ptr())
        self.assertEqual(exits, ["dense partial"])
        self.assertEqual(st.scratch[hop_ffn.TALLY], Counter({hop_ffn.TAKEN: 1}))


class TestFallsBackToStockPrepare(CustomTestCase):
    """Where the stock prepare would not run the push-plane kernel, the slot declines
    before touching anything, and ffn_input runs the stock prepare."""

    def _declines(
        self, layer, out, fb, why, *, hop=None, layer_id=1, fusable=True, counters=512
    ):
        """The slot declines, having done nothing; ``why`` is the reason the step's
        tally gets (None: a layer the slot was not built for, which it does not
        count)."""
        hop = hop or _hop(layer)
        stream = fb.residual_stream
        state = (stream.pending, stream.residual) if stream is not None else None
        tally = Counter()
        with tp4(fusable=fusable, counters=counters), kernels() as calls:
            self.assertIsNone(hop.ffn_input(layer_id, layer, out, fb, tally))
        self.assertEqual(tally, Counter({why: 1}) if why else Counter())
        self.assertEqual((calls.quant.call_count, calls.nvfp4.call_count), (0, 0))
        self.assertIs(fb.residual_stream, stream)
        if stream is not None:
            self.assertIs(stream.pending, state[0])
            self.assertIs(stream.residual, state[1])

    def test_batches_the_push_plane_kernel_does_not_take(self):
        layer = _layer()
        why = "a batch the push-plane kernel does not take"
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            with self.subTest(mode=mode):
                fb, out = _batch(layer, 8, 1, mode=mode)
                self._declines(layer, out, fb, why)
        fb, out = _batch(layer, 8, 1)
        self._declines(layer, out, fb, why, fusable=False)

    def test_a_partial_sum_the_kernel_does_not_take(self):
        layer = _layer()
        fb, out = _batch(layer, 8, 2, dtype=torch.float16)
        self._declines(layer, out, fb, "no bf16 norm gamma, residual or partial sum")

    def test_rows_past_the_buffers_a_scale_tile_without_counters(self):
        layer = _layer()
        fb, out = _batch(layer, 16, 3)
        self._declines(
            layer, out, fb, "rows past the slot's buffers", hop=_hop(layer, widths=(8,))
        )
        fb, out = _batch(layer, 8, 3)
        self._declines(
            layer,
            out,
            fb,
            "no push-plane counter left to zero the scale tile's padding",
            counters=8,
        )

    def test_another_batch_path_or_another_layer(self):
        layer = _layer()
        hop = _hop(layer)
        fb, out = _batch(layer, 8, 4)
        other = msgspec.structs.replace(
            layer.ffn_boundary.plan.paths[BatchVariant.ORDINARY].entry,
            input_move=lambda **kw: kw["hidden_states"],
        )
        with patch.object(layer.ffn_boundary, "entry", lambda fb, steps=None: other):
            self._declines(layer, out, fb, "another batch path", hop=hop)
        # layers the slot was not built for are not counted
        self._declines(_layer(), out, fb, None, hop=hop)
        self._declines(layer, out, fb, None, hop=hop, layer_id=2)

    def test_a_stream_without_the_attentions_partial(self):
        layer = _layer()
        fb, out = _batch(layer, 8, 5)
        stream = fb.residual_stream
        # an attention output already completed (pending, owing nothing)
        completed = ResidualStream(stream.residual)
        handle = completed.record(torch.zeros(8, H, dtype=torch.bfloat16), PLAIN_ADD)
        fb.residual_stream = completed
        self._declines(
            layer,
            handle,
            fb,
            "the attention output does not owe its attention-TP sum",
        )
        # an unreduced output instead of the declared sum
        unreduced = ResidualStream(stream.residual)
        handle = unreduced.record(
            UnreducedOutput(torch.zeros(8, H, dtype=torch.bfloat16), hop_tests.TP),
            PLAIN_ADD,
        )
        fb.residual_stream = unreduced
        self._declines(
            layer,
            handle,
            fb,
            "the attention output does not owe its attention-TP sum",
        )
        # nothing pending, no residual written
        fb.residual_stream = ResidualStream()
        self._declines(
            layer,
            torch.zeros(8, H),
            fb,
            "no plain add pending on a written residual",
        )
        fb.residual_stream = None
        self._declines(layer, out, fb, "no residual stream")

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
                fb, out = _batch(layer, 8, 3)
                stream = fb.residual_stream
                fb = Batch(**vars(fb))
                rebound.clear()
                with tp4(), kernels() as calls:
                    if slot:
                        got = _hop(layer).ffn_input(1, layer, out, fb)
                        self.assertIsInstance(got, hop_ffn.FfnInput)
                    else:
                        layer.ffn_boundary.prepare(out, fb)
                self.assertIs(fb.residual_stream, stream)
                self.assertEqual(rebound, [] if slot else [("residual_stream", stream)])
                # written in place: the pending output consumed, the kernel's residual
                kernel = calls.nvfp4 if slot else calls.stock
                self.assertIsNone(stream.pending)
                self.assertIs(stream.residual, kernel.call_args.kwargs["residual_out"])

    def test_an_output_of_another_stream_raises_as_in_stock(self):
        layer = _layer()
        fb, _ = _batch(layer, 8, 6)
        _, stranger = _batch(layer, 8, 6)
        with (
            tp4(),
            kernels() as calls,
            self.assertRaisesRegex(
                RuntimeError, "does not belong to this residual stream"
            ),
        ):
            _hop(layer).ffn_input(1, layer, stranger, fb, None)
        self.assertEqual((calls.quant.call_count, calls.nvfp4.call_count), (0, 0))

    def test_ffn_input_runs_the_stock_prepare_when_declined_or_off(self):
        declining = SimpleNamespace(ffn_input=Mock(return_value=None))
        layer = SimpleNamespace(
            attn_boundary=SimpleNamespace(finish=lambda out, fb: ("owed", out)),
            ffn_boundary=SimpleNamespace(prepare=Mock()),
        )
        for slots, bound in (
            (m31_plan.Slots(), {"hop_ffn": declining}),
            (m31_plan.Slots(hop_ffn=True), {"hop_ffn": declining}),
            (m31_plan.Slots(hop_ffn=True), {"hop_ffn": None}),
            (m31_plan.Slots(hop_ffn=True), {}),
        ):
            layer.ffn_boundary.prepare.reset_mock()
            st = SimpleNamespace(
                fb="fb", layer_id=5, slots=slots, bound=bound, scratch={}
            )
            out = boundary.ffn_input(layer, "x", st)
            self.assertIs(out, layer.ffn_boundary.prepare.return_value)
            layer.ffn_boundary.prepare.assert_called_once_with(("owed", "x"), "fb")
            # only a step the slot serves keeps a tally
            self.assertEqual(
                hop_ffn.TALLY in st.scratch,
                bound.get("hop_ffn") is declining and slots.hop_ffn,
            )
        # with the slot off, or not built, the build is never asked
        self.assertEqual(declining.ffn_input.call_count, 1)
        declining.ffn_input.assert_called_with(5, layer, ("owed", "x"), "fb", Counter())


class TestReport(CustomTestCase):
    def test_a_line_per_width_and_one_summary(self):
        built = hop_ffn.HopFfn(
            widths=(16, 8),
            hops={1: None, 2: None},
            refused={},
            hidden=H,
            device="cpu",
        )
        nvfp4_left = f"{hop_ffn.TAKEN} without the NVFP4 (4over6)"
        with self.assertLogs(hop_ffn.logger, level="INFO") as logs:
            built.report(8, Counter({hop_ffn.TAKEN: 2}))
            built.report(16, Counter({hop_ffn.TAKEN: 1, nvfp4_left: 1}))
            built.report(16, Counter({"another batch path": 2}))
            built.report(12, Counter({"another batch path": 2}))
        warning, summary = logs.records
        self.assertEqual(warning.levelname, "WARNING")
        self.assertIn("took 1 of the 2 FFN inputs", warning.getMessage())
        self.assertIn(f"{nvfp4_left}: 1 layers", warning.getMessage())
        self.assertIn("at rows [8]; fewer at rows [16]", summary.getMessage())

    def test_final_hands_over_both_tallies(self):
        hop_mxfp8, ffn_hop = Mock(), Mock()
        tallies = {
            boundary.TALLY: Counter({boundary.TAKEN: 2}),
            hop_ffn.TALLY: Counter({hop_ffn.TAKEN: 3}),
        }
        model = SimpleNamespace(
            norm="norm", finalize_norm="finalize", end_layer=3, layers_to_capture=[]
        )
        st = SimpleNamespace(
            scratch=dict(tallies),
            bound={"boundary": hop_mxfp8, "hop_ffn": ffn_hop},
            rows=8,
            model=model,
            fb="fb",
        )
        with patch.object(residual_batch, "final_norm", return_value="normed"):
            self.assertEqual(boundary.final(st, "x", SimpleNamespace()), "normed")
        hop_mxfp8.report.assert_called_once_with(8, tallies[boundary.TALLY])
        ffn_hop.report.assert_called_once_with(8, tallies[hop_ffn.TALLY])


# ------------------------------------------------------------------ consumers


def _fp8_method(backend="cute-dsl", use_mxfp8=True):
    method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    method.use_marlin = False
    method.use_mxfp8 = use_mxfp8
    method.block_fp8_as_mxfp8 = False
    method.mxfp8_prefill_autotune_min_tokens = None
    method.mxfp8_dense_backend = SimpleNamespace(
        is_flashinfer_cutlass=lambda: backend == "cutlass",
        is_flashinfer_cutedsl=lambda: backend == "cute-dsl",
        is_deep_gemm=lambda: backend == "deep_gemm",
        is_flashinfer_trtllm=lambda: backend == "trtllm",
        is_gfx95=lambda: False,
    )
    method.w8a8_mxfp8_linear = partial(
        fp8_utils.flashinfer_mxfp8_blockscaled_linear, backend="cute-dsl"
    )
    return method


def _gate_up(k=H, n=40, **method_kw):
    """A gate_up as SGLang builds it: a MergedColumnParallelLinear on the MXFP8 path."""
    linear = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    torch.nn.Module.__init__(linear)
    linear.quant_method = _fp8_method(**method_kw)
    linear.weight = torch.nn.Parameter(
        torch.zeros(n, k, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    linear.weight_scale_inv_swizzled = torch.zeros(n * k // 32, dtype=torch.uint8)
    linear.bias = None
    linear.skip_bias_add = False
    linear.gather_output = False
    linear.tp_group = None
    return linear


def _mlp(**gate_up_kw):
    """A MiniMaxM3MLP (no __init__): the dense MLP or a shared expert."""
    mlp = MiniMaxM3MLP.__new__(MiniMaxM3MLP)
    torch.nn.Module.__init__(mlp)
    mlp.gate_up_proj = _gate_up(**gate_up_kw)
    return mlp


def _pair(rows=8, k=H):
    return Mxfp8SwizzledInput(
        torch.zeros(rows, k, dtype=torch.float8_e4m3fn),
        torch.zeros(ops.mxfp8_scale_bytes(k), dtype=torch.uint8),
    )


def _prepared(rows=8, *, nvfp4=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, H, generator=g).bfloat16()
    triple = (
        torch.randint(0, 256, (rows, H // 2), generator=g, dtype=torch.uint8),
        torch.randint(0, 256, (rows, SF), generator=g, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        ),
        torch.rand(rows, generator=g),
    )
    return hop_ffn.FfnInput(x, _pair(rows), triple if nvfp4 else None)


@contextlib.contextmanager
def forward_state(sp=False):
    with patch.object(
        runtime_context, "get_forward", lambda: SimpleNamespace(sp_active=sp)
    ):
        yield


class TestMlpTakesThePair(CustomTestCase):
    """The dense MLP or shared expert the build accepts, and what it gets per step."""

    def test_the_mxfp8_mlp_takes_the_pair(self):
        self.assertEqual(hop_ffn.mlp_reasons(_mlp(), hidden=H), [])
        cases = {
            "deep_gemm backend": _mlp(backend="deep_gemm"),
            "not MXFP8": _mlp(use_mxfp8=False),
            "other K": _mlp(k=2 * H),
            "not an MLP": SimpleNamespace(),
        }
        hooked = _mlp()
        hooked.register_forward_pre_hook(lambda *a: None)
        cases["hooked MLP"] = hooked
        hooked_gate_up = _mlp()
        hooked_gate_up.gate_up_proj.register_forward_hook(lambda *a: None)
        cases["hooked gate_up"] = hooked_gate_up
        replaced = _mlp()
        replaced.forward = lambda x, forward_batch=None: x
        cases["replaced forward"] = replaced
        other_linear = _mlp()
        other_linear.gate_up_proj.quant_method.w8a8_mxfp8_linear = Mock()
        cases["another MXFP8 linear"] = other_linear
        for name, mlp in cases.items():
            with self.subTest(name):
                self.assertTrue(hop_ffn.mlp_reasons(mlp, hidden=H))

    def test_mlp_input_per_step(self):
        mlp, prepared = _mlp(), _prepared()
        with forward_state():
            self.assertIs(hop_ffn.mlp_input(mlp, prepared), prepared.mxfp8)
        with forward_state(sp=True):
            self.assertIs(hop_ffn.mlp_input(mlp, prepared), prepared.x)
        mlp.gate_up_proj.register_forward_pre_hook(lambda *a: None)
        with forward_state():
            self.assertIs(hop_ffn.mlp_input(mlp, prepared), prepared.x)

    def test_consumers_of_a_layer(self):
        dense = SimpleNamespace(is_layer_sparse=False, mlp=_mlp())
        self.assertEqual(hop_ffn.consumer_reasons(dense, hidden=H), [])
        dense.mlp = _mlp(backend="deep_gemm")
        self.assertTrue(hop_ffn.consumer_reasons(dense, hidden=H))
        trace = nofill_tests.Trace()
        moe = nofill_tests.Moe(trace, nofill_tests.Streams(trace))
        moe.shared_experts = _mlp()
        layer = SimpleNamespace(is_layer_sparse=True, mlp=moe, layer_id=3)
        with patch.object(mn, "layer_reasons", return_value=[]) as chain:
            self.assertEqual(hop_ffn.consumer_reasons(layer, hidden=H), [])
            chain.assert_called_once_with(layer)
            moe.shared_experts = _mlp(use_mxfp8=False)
            self.assertTrue(hop_ffn.consumer_reasons(layer, hidden=H))
            moe.shared_experts = None
            self.assertTrue(hop_ffn.consumer_reasons(layer, hidden=H))
        with patch.object(mn, "layer_reasons", return_value=["the chain"]):
            moe.shared_experts = _mlp()
            self.assertEqual(hop_ffn.consumer_reasons(layer, hidden=H), ["the chain"])


class TestStockModulesTakeThePair(CustomTestCase):
    """MiniMaxM3MLP hands its input only to the gate_up, which hands it to its quant
    method as it is, which hands the pair to the GEMM as its quantized input."""

    def test_the_mlp_reads_its_input_only_through_the_gate_up(self):
        gate_up = torch.randn(8, 80).bfloat16()
        mlp = SimpleNamespace(
            gate_up_proj=Mock(return_value=(gate_up, None)),
            _swiglu_oai_mxfp8=Mock(return_value="mxfp8 act"),
            down_proj=Mock(return_value=("down", None)),
        )
        pair = _pair()
        self.assertEqual(MiniMaxM3MLP.forward(mlp, pair), "down")
        mlp.gate_up_proj.assert_called_once_with(pair)
        mlp._swiglu_oai_mxfp8.assert_called_once_with(gate_up)
        mlp.down_proj.assert_called_once_with("mxfp8 act")

    def test_the_gate_up_and_its_method_pass_the_pair_to_the_gemm(self):
        linear = _gate_up()
        pair = _pair()
        linear.quant_method.w8a8_mxfp8_linear = Mock(return_value="gemm out")
        with (
            patch.object(linear_mod, "get_group_rank_size", lambda group: (0, 4)),
            patch.object(
                linear_mod, "get_forward", lambda: SimpleNamespace(sp_active=False)
            ),
        ):
            self.assertEqual(
                ColumnParallelLinear.forward(linear, pair), ("gemm out", None)
            )
        linear.quant_method.w8a8_mxfp8_linear.assert_called_once_with(
            input=pair.data,
            weight=linear.weight,
            weight_scale=linear.weight_scale_inv_swizzled,
            input_scale=pair.scales,
            bias=None,
        )


# ------------------------------------------------------------------ the FFN step


class _PairShared(torch.nn.Module):
    """A shared expert that records what it is called on."""

    def __init__(self, trace, streams):
        super().__init__()
        self.trace, self.streams = trace, streams
        self.gate_up_proj = torch.nn.Identity()
        self.seen = []

    def forward(self, x):
        self.seen.append(x)
        self.trace.add("shared", self.streams.current.name)
        data = x.data if isinstance(x, Mxfp8SwizzledInput) else x
        return data.float().bfloat16() * 0.5


class TestFfnStepTakesTheInputs(CustomTestCase):
    """moe.step on an FfnInput: the gate_up of the dense MLP and of the shared expert
    get the MXFP8 pair, moe_nofill's routed experts the NVFP4 (no quantize launch);
    everything else is moe_nofill's step on the bf16 rows."""

    def _moe(self, rows=8):
        trace = nofill_tests.Trace()
        streams = nofill_tests.Streams(trace)
        fi = nofill_tests.FakeFlashInfer(trace, streams)
        model = nofill_tests.fake_model(trace, streams)
        layer = model.model.layers[1]
        layer.mlp.shared_experts = _PairShared(trace, streams)
        quant = nofill_tests._quant()
        st = SimpleNamespace(
            slots=m31_plan.Slots(moe_nofill=True, hop_ffn=True),
            bound={"moe": nofill_tests._bound(model, quant)},
            fb="fb",
        )
        return SimpleNamespace(
            trace=trace, streams=streams, fi=fi, layer=layer, quant=quant, st=st
        )

    def test_a_moe_layer_takes_the_pair_and_the_nvfp4(self):
        prepared = _prepared()
        runs = {}
        for path in ("moe_nofill", "hop_ffn"):
            run = self._moe()
            with (
                run.fi.installed(),
                nofill_tests.runtime(run.streams),
                forward_state(),
            ):
                if path == "moe_nofill":
                    run.out = mn.forward(run.layer.mlp, prepared.x, run.quant)
                else:
                    run.out = m31_moe.step(run.layer, prepared, run.st)
            runs[path] = run
        want, got = runs["moe_nofill"], runs["hop_ffn"]
        # moe_nofill's launches on its streams, minus the quantize
        self.assertEqual(
            got.trace.calls, [c for c in want.trace.calls if c[0] != "quant_kernel"]
        )
        self.assertEqual(len(want.fi.launches), 1)
        self.assertEqual(got.fi.launches, [])
        # the shared expert on the pair, the routed MoE on the NVFP4
        (shared_in,) = got.layer.mlp.shared_experts.seen
        self.assertIs(shared_in, prepared.mxfp8)
        (moe_call,) = got.fi.moe_calls
        fp4, hs_scale, token_scales = prepared.nvfp4
        self.assertIs(moe_call["hidden_states"], fp4)
        self.assertIs(moe_call["hidden_states_scale"], hs_scale)
        self.assertIs(moe_call["per_token_scale"], token_scales)
        # every other argument as moe_nofill passes it
        (want_call,) = want.fi.moe_calls
        for name, value in want_call.items():
            if name not in ("hidden_states", "hidden_states_scale", "per_token_scale"):
                self.assertTrue(nofill_tests._same(moe_call[name], value), name)
        self.assertIs(got.out.gated_shared_output.dtype, torch.bfloat16)

    def test_without_the_nvfp4_the_routed_experts_quantize(self):
        prepared = _prepared(nvfp4=False)
        run = self._moe()
        with run.fi.installed(), nofill_tests.runtime(run.streams), forward_state():
            m31_moe.step(run.layer, prepared, run.st)
        self.assertEqual(len(run.fi.launches), 1)
        self.assertIs(run.fi.launches[0]["x"], prepared.x)
        (shared_in,) = run.layer.mlp.shared_experts.seen
        self.assertIs(shared_in, prepared.mxfp8)

    def test_a_moe_layer_that_falls_back_gets_the_bf16_rows(self):
        prepared = _prepared()
        run = self._moe()
        seen = []

        def stock(self, hidden_states, forward_batch):
            seen.append(hidden_states)
            return "stock"

        with (
            run.fi.installed(),
            nofill_tests.runtime(run.streams, defer=False),
            forward_state(),
            patch.object(MiniMaxM3MoE, "forward", stock),
        ):
            self.assertEqual(m31_moe.step(run.layer, prepared, run.st), "stock")
        self.assertEqual(seen, [prepared.x])

    def test_a_dense_layer_takes_the_pair(self):
        prepared = _prepared(nvfp4=False)
        seen = []
        layer = SimpleNamespace(
            is_layer_sparse=False,
            mlp=lambda x, forward_batch: seen.append((x, forward_batch)) or "partial",
        )
        st = SimpleNamespace(slots=m31_plan.Slots(hop_ffn=True), bound={}, fb="fb")
        with patch.object(hop_ffn, "mlp_input", lambda mlp, p: p.mxfp8):
            self.assertEqual(m31_moe.step(layer, prepared, st), "partial")
        self.assertEqual(seen, [(prepared.mxfp8, "fb")])
        seen.clear()
        with patch.object(hop_ffn, "mlp_input", lambda mlp, p: p.x):
            m31_moe.step(layer, prepared, st)
        self.assertIs(seen[0][0], prepared.x)

    def test_prequantized_hands_out_the_nvfp4_for_its_rows_only(self):
        prepared = _prepared()
        quant = Mock(return_value="quantized")
        pre = mn.Prequantized(prepared.x, prepared.nvfp4, quant)
        self.assertIs(pre(prepared.x), prepared.nvfp4)
        other = prepared.x.clone()
        self.assertEqual(pre(other), "quantized")
        quant.assert_called_once_with(other)


# ------------------------------------------------------------------ layers and build


class TestLayersAndBuild(CustomTestCase):
    def test_layers_whose_ffn_input_the_slot_takes(self):
        for moe in (True, False):
            self.assertIsNone(hop_ffn.ffn_hop(_layer(moe=moe), hidden=H)[1])
        cases = {
            "no fusion": _layer(fusions=False),
            "fp16 norm": _layer(norm=hop_tests._norm(torch.float16)),
            "no boundary": SimpleNamespace(),
        }
        for name, case in cases.items():
            with self.subTest(name):
                hop, why = hop_ffn.ffn_hop(case, hidden=H)
                self.assertIsNone(hop)
                self.assertTrue(why)
        self.assertIsNone(hop_ffn.ffn_hop(_layer(), hidden=2 * H)[0])

        def step_with(entry, **keywords):
            # the layer's own entry with its reduce step's keywords changed
            step = entry.prepare.keywords["step"]
            return dict(
                prepare=partial(
                    _run_entry,
                    **{
                        **entry.prepare.keywords,
                        "step": partial(step.func, **{**step.keywords, **keywords}),
                    },
                )
            )

        for name, change in (
            ("input move", lambda e: dict(input_move=lambda **kw: kw["hidden_states"])),
            ("no declared sum", lambda e: dict(declared_sum=None)),
            (
                "fusion order",
                lambda e: step_with(
                    e, fusions=e.prepare.keywords["step"].keywords["fusions"][::-1]
                ),
            ),
            ("read fusions first", lambda e: step_with(e, read_fusions=(object(),))),
            ("read fusion indices", lambda e: step_with(e, read_fusion_indices=(0,))),
            (
                "read bound per batch",
                lambda e: step_with(e, read=SimpleNamespace(bind=print)),
            ),
            (
                "update capabilities",
                lambda e: dict(prepare=partial(lambda *a, **k: None, paths={})),
            ),
        ):
            with self.subTest(name):
                layer = _layer()
                path = layer.ffn_boundary.plan.paths[BatchVariant.ORDINARY]
                layer.ffn_boundary.plan.paths[BatchVariant.ORDINARY] = (
                    msgspec.structs.replace(
                        path,
                        entry=msgspec.structs.replace(path.entry, **change(path.entry)),
                    )
                )
                hop, why = hop_ffn.ffn_hop(layer, hidden=H)
                self.assertIsNone(hop)
                self.assertTrue(why)
        layer = _layer()
        layer.ffn_boundary.plan._publish_lora_layout = True
        self.assertIsNone(hop_ffn.ffn_hop(layer, hidden=H)[0])

    def _model(self, layers):
        return SimpleNamespace(
            model=SimpleNamespace(layers=layers, start_layer=0, end_layer=len(layers))
        )

    @contextlib.contextmanager
    def _building(self, *, probe=(False, ""), ranks=1):
        """hop_ffn.build's world: TP4, FFNs that take the slot's inputs, the probe's
        result ``probe`` on each of ``ranks`` ranks; the geometries the kernel launches
        at its hidden width (6144; the fakes carry 64)."""
        probes = []

        def probing(geometries, hidden=H, device="cuda"):
            probes.append(list(geometries))
            return probe

        geometries = hop_ffn._geometries
        with contextlib.ExitStack() as stack:
            stack.enter_context(tp4())
            stack.enter_context(patch.object(hop_ffn, "HIDDEN", H))
            stack.enter_context(
                patch.object(
                    hop_ffn, "_geometries", lambda w, ws, h: geometries(w, ws, 6144)
                )
            )
            stack.enter_context(
                patch.object(hop_ffn, "consumer_reasons", lambda layer, hidden=H: [])
            )
            stack.enter_context(patch.object(hop_ffn, "probe_nvfp4", probing))
            stack.enter_context(patch.object(mn, "_gather", lambda obj: [obj] * ranks))
            stack.enter_context(
                patch.object(torch.cuda, "is_available", return_value=True)
            )
            plain = stack.enter_context(patch.object(ops, "load_residual_norm_quant"))
            nvfp4 = stack.enter_context(
                patch.object(ops, "load_residual_norm_quant_nvfp4")
            )
            yield SimpleNamespace(probes=probes, plain=plain, nvfp4=nvfp4)

    def test_build_takes_every_layer_at_every_width(self):
        self.assertIsNone(hop_ffn.build(None, (8, 16), frozenset()))
        self.assertFalse(hop_ffn.serves(None, 8))
        layers = [_layer(moe=False), _layer(moe=False), _layer(moe=True)]
        widths = tuple(range(8, 129, 8))
        with self._building() as world:
            built = hop_ffn.build(self._model(layers), widths, frozenset({"hop_ffn"}))
        self.assertEqual(sorted(built.hops), [0, 1, 2])
        self.assertEqual([built.hops[i].moe for i in range(3)], [False, False, True])
        self.assertEqual(built.widths, widths)
        self.assertEqual((built.max_rows, built.ftz, built.nvfp4), (128, False, True))
        self.assertEqual(
            (built.data.shape, built.data.dtype), ((128, H), torch.float8_e4m3fn)
        )
        self.assertEqual(built.scales.shape, (ops.mxfp8_scale_bytes(H),))
        self.assertEqual(
            (built.fp4.shape, built.fp4_scales.shape, built.token_scales.shape),
            ((128, H // 2), (128 * SF,), (128,)),
        )
        self.assertTrue(all(hop_ffn.serves(built, w) for w in widths))
        self.assertFalse(hop_ffn.serves(built, 12))
        # probed once, at the geometries the widths launch: TP4 one-shot (cluster 6)
        # below 48 rows, two-shot (cluster 4) from 48 on
        self.assertEqual(world.probes, [[(6, 1), (4, 1)]])
        # each width's plain (dense layers) and NVFP4 (MoE layers) kernels compiled
        self.assertEqual(world.plain.call_count, len(widths))
        self.assertEqual(world.nvfp4.call_count, len(widths))
        world.nvfp4.assert_any_call(4, 128, H, ftz=False)

    def test_the_nvfp4_is_off_unless_every_rank_found_one_mode(self):
        layers = [_layer(moe=False), _layer(moe=True)]
        model = self._model(layers)
        why = "the kernel's NVFP4 differs from FlashInfer's per-token quantize"
        for probe, ranks, ftz in (
            ((True, ""), 4, True),
            ((None, why), 4, None),
        ):
            with self._building(probe=probe, ranks=ranks) as world:
                built = hop_ffn.build(model, (8, 16), frozenset({"hop_ffn"}))
            self.assertEqual(built.ftz, ftz)
            self.assertEqual(built.nvfp4, ftz is not None)
            self.assertEqual(built.nvfp4_off, "" if ftz is not None else why)
            self.assertEqual(world.nvfp4.call_count, 2 if ftz is not None else 0)
            self.assertEqual(built.fp4.shape[0], 16 if ftz is not None else 0)
        # ranks that differ, a probe that raises
        results = iter([[(False, ""), (True, ""), (False, ""), (False, "")]])
        with (
            self._building(),
            patch.object(mn, "_gather", lambda obj: next(results)),
        ):
            built = hop_ffn.build(model, (8,), frozenset({"hop_ffn"}))
        self.assertIsNone(built.ftz)
        self.assertIn("rank 1: subnormals flushed", built.nvfp4_off)

        def raising(geometries, hidden=H, device="cuda"):
            raise RuntimeError("no GPU")

        with self._building(), patch.object(hop_ffn, "probe_nvfp4", raising):
            built = hop_ffn.build(model, (8,), frozenset({"hop_ffn"}))
        self.assertIsNone(built.ftz)
        self.assertIn("RuntimeError: no GPU", built.nvfp4_off)

    def test_only_dense_layers_or_no_layer(self):
        gathered = []
        with (
            self._building() as world,
            patch.object(mn, "_gather", lambda obj: gathered.append(obj) or [obj]),
        ):
            built = hop_ffn.build(
                self._model([_layer(moe=False)]), (8,), frozenset({"hop_ffn"})
            )
            # the gather runs on every rank even without a probe
            self.assertEqual(world.probes, [])
            self.assertEqual(len(gathered), 1)
            self.assertFalse(built.nvfp4)
            nothing = hop_ffn.build(
                self._model([_layer(fusions=False)]), (8,), frozenset({"hop_ffn"})
            )
        self.assertEqual(nothing.hops, {})
        self.assertFalse(hop_ffn.serves(nothing, 8))
        self.assertEqual(list(nothing.refused), [0])
        self.assertEqual(len(gathered), 2)

    def test_the_plan_needs_moe_nofill(self):
        served = SimpleNamespace(serves=lambda rows: rows <= 64)
        ctx = SimpleNamespace(
            widths=tuple(range(8, 129, 8)), tokens_per_request=8, binding=None
        )
        nofill = SimpleNamespace(serves=lambda rows: rows >= 32)
        with (
            patch.object(m31_runtime, "SLOTS", frozenset({"hop_ffn", "moe_nofill"})),
            patch.object(hop_ffn, "build", return_value=served),
            patch.object(m31_moe, "build", return_value=nofill),
            patch.object(m31_agree, "gather_tp", lambda obj: [obj]),
        ):
            built = m31_runtime.build(SimpleNamespace(), ctx)
        self.assertEqual(
            [w for w, s in built.slot_plan.items() if s.hop_ffn],
            list(range(32, 65, 8)),
        )
        self.assertIn("hop_ffn at", built.summary())

    def test_slot_constants(self):
        self.assertIn("hop_ffn", m31_runtime.OPT_IN_SLOTS)
        self.assertNotIn("hop_ffn", m31_runtime.SHIPPED_SLOTS)
        self.assertIn("hop_ffn", m31_runtime.BITWISE_SLOTS)
        self.assertEqual(m31_runtime.NEEDS["hop_ffn"], "moe_nofill")
        self.assertTrue(m31_runtime.slot_reasons(frozenset({"hop_ffn"})))
        self.assertEqual(
            m31_runtime.slot_reasons(frozenset({"hop_ffn", "moe_nofill"})), []
        )
        self.assertIn("hop_ffn", m31_plan.Slots.__struct_fields__)


class TestProbeRows(CustomTestCase):
    """The probe rows and global scales, on a model of FlashInfer's per-token scale and
    block scale bytes (``_model``): the two subnormal modes of the f32 operations
    CuTe-DSL compiles (the compare, the multiplies) give different bytes on the
    subnormal-window rows and the same bytes on every other; another association of
    the block scale's three multiplies, or a one-ulp change in rcp(6), changes some
    byte whatever rcp.approx returns within an ulp."""

    def test_the_subnormal_rows_tell_the_modes_apart(self):
        x = hop_ffn.probe_rows(6144, "cpu").float().numpy()
        for ulps, scale in hop_ffn.probe_scales(448.0, "cpu"):
            gsi = np.float32(scale.item())
            differ = [
                i
                for i, row in enumerate(x)
                if _model(row, gsi, flush=False) != _model(row, gsi, flush=True)
            ]
            with self.subTest(ulps=ulps):
                self.assertEqual(differ, [len(x) - 3, len(x) - 2, len(x) - 1])
        self.assertEqual(hop_ffn.probe_rows(6144, "cpu").dtype, torch.bfloat16)

    def test_the_rows_tell_the_orders_and_rcp6_apart(self):
        """Over the probe's global scales, under every model of rcp.approx (the encode
        scale's and rcp(6) off by -1, 0 or +1 ulp, the encode scale's throughout or at
        random per per-token scale): each other association of the block scale's
        multiplies, and rcp(6) one ulp either side, changes some block scale byte of
        the probe rows. (At the served scale alone, the encode scale and rcp(6) off
        the same way leave the associations alike on every row: the rows' ties need
        the neighbouring scales.)"""
        x = hop_ffn.probe_rows(6144, "cpu").float().numpy()
        scales = [np.float32(s.item()) for _, s in hop_ffn.probe_scales(448.0, "cpu")]
        models = [(dg, dr) for dg in (-1, 0, 1) for dr in (-1, 0, 1)]
        models += [(_random_ulps(seed), dr) for seed in range(3) for dr in (-1, 0, 1)]
        for encode_ulps, rcp6_ulps in models:
            kw = dict(encode_ulps=encode_ulps, rcp6_ulps=rcp6_ulps)
            moved = Counter()
            for gsi in scales:
                for row in x:
                    want = np.array(_model(row, gsi, **kw)[1:])
                    for order in (1, 2):
                        got = np.array(_model(row, gsi, order=order, **kw)[1:])
                        moved[_ORDERS[order]] += int((got != want).sum())
                    for d in (-1, 1):
                        kw_r = dict(kw, rcp6_ulps=rcp6_ulps + d)
                        got = np.array(_model(row, gsi, **kw_r)[1:])
                        moved[f"rcp(6) {d:+d} ulp"] += int((got != want).sum())
            with self.subTest(encode_ulps=encode_ulps, rcp6_ulps=rcp6_ulps):
                self.assertEqual(len(moved), 4)
                self.assertTrue(all(n > 0 for n in moved.values()), moved)

    def test_the_midpoint_rows_sit_on_midpoints(self):
        """Every block of a midpoint row but the amax's has the max of an e4m3 rounding
        midpoint, exactly: ``e4m3_max * block_max / amax`` is one; no row's per-token
        scale is a power of two (as at the tie rows, where every association gives
        the same product); the rows are bf16."""
        for e4m3_max in mn.E4M3_MAXES:
            _, mids = hop_ffn._e4m3_values_and_midpoints(e4m3_max)
            mids = {Fraction(m) for m in mids.tolist()}
            rows = hop_ffn._midpoint_rows(6144, e4m3_max)
            self.assertTrue(torch.equal(rows.to(torch.bfloat16).float(), rows))
            gsi = np.float32(1.0 / (e4m3_max * 6.0))
            for c, row in zip(hop_ffn.MIDPOINT_ROW_C, rows):
                with self.subTest(e4m3_max=e4m3_max, c=c):
                    maxima = row.reshape(-1, 16).abs().amax(dim=1).tolist()
                    amax = max(maxima)
                    self.assertEqual((amax, maxima[0]), (e4m3_max * c, amax))
                    ratios = [
                        Fraction(e4m3_max) * Fraction(b) / Fraction(amax)
                        for b in maxima[1:]
                        if b
                    ]
                    self.assertGreaterEqual(len(ratios), 8)
                    self.assertLessEqual(set(ratios), mids)
                    token = np.float32(np.float32(amax) * gsi)
                    self.assertNotEqual(int(token.view(np.uint32)) & 0x7FFFFF, 0)
        tie = hop_ffn._tie_rows(6144).abs().amax(dim=1)
        token = (tie.numpy().astype(np.float32) * np.float32(1 / 2688)).astype(
            np.float32
        )
        self.assertEqual((token.view(np.uint32) & 0x7FFFFF).tolist(), [0] * len(token))

    def test_the_probe_scales(self):
        """The served inverse global scale first, as the stock chain's torch.full
        rounds it, then its f32 neighbours, each its ulps away."""
        for e4m3_max in mn.E4M3_MAXES:
            scales = hop_ffn.probe_scales(e4m3_max, "cpu")
            self.assertEqual([u for u, _ in scales], list(hop_ffn.PROBE_SCALE_ULPS))
            self.assertEqual(scales[0][0], 0)
            served = torch.full((1,), 1.0 / (e4m3_max * 6.0), dtype=torch.float32)
            for ulps, scale in scales:
                self.assertEqual(
                    (scale.dtype, tuple(scale.shape)), (torch.float32, (1,))
                )
                bits = scale.view(torch.int32).item() - served.view(torch.int32).item()
                self.assertEqual(bits, ulps)


_TINY = np.float32(2.0**-126)
# The block scale's association: FlashInfer's, and the two others
_ORDERS = (
    "encode * (block_max * rcp6)",
    "(encode * block_max) * rcp6",
    "(encode * rcp6) * block_max",
)


def _flush(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where((x != 0) & (np.abs(x) < _TINY), np.float32(0.0), x).astype(
        np.float32
    )


def _moved(x, ulps):
    """f32 ``x`` moved ``ulps`` ulps."""
    x = np.float32(x)
    for _ in range(abs(ulps)):
        x = np.nextafter(x, np.float32(np.copysign(np.inf, ulps)))
    return np.float32(x)


def _random_ulps(seed):
    """An rcp.approx model off by -1, 0 or +1 ulp at random per input (the same at
    the same input)."""
    rng, table = np.random.default_rng(seed), {}

    def ulps(a):
        key = int(np.float32(a).view(np.uint32))
        if key not in table:
            table[key] = int(rng.integers(-1, 2))
        return table[key]

    return ulps


def _model(row, gsi, *, flush=False, order=0, encode_ulps=0, rcp6_ulps=0):
    """A model of FlashInfer's per-token scale and e4m3 block scale bytes of ``row``
    (fast math): the compare and the multiplies IEEE f32, ``flush``: flushing
    subnormals; rcp.approx.ftz IEEE division of the flushed input, moved by
    ``encode_ulps`` (ulps, or a function of the per-token scale giving them) for the
    encode scale and ``rcp6_ulps`` for rcp(6); the block scale's multiplies in
    association ``order`` (``_ORDERS``, 0: FlashInfer's)."""

    def mul(a, b):
        a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        if flush:
            a, b = _flush(a), _flush(b)
        with np.errstate(over="ignore", invalid="ignore"):
            r = (a * b).astype(np.float32)
        return _flush(r) if flush else r

    def rcp(a, ulps):
        with np.errstate(divide="ignore"):
            r = np.float32(1.0) / _flush(a)  # rcp.approx.ftz
        return _flush(_moved(r, ulps)) if np.isfinite(r) else r

    a = np.abs(row.astype(np.float32))
    amax = np.fmax.reduce(np.concatenate(([np.float32(0.0)], a))).astype(np.float32)
    if (_flush(amax) if flush else amax) == 0:
        token, encode = np.float32(0.0), np.float32(3.4028234663852886e38)
    else:
        token = np.float32(mul(amax, gsi))
        ulps = encode_ulps(token) if callable(encode_ulps) else encode_ulps
        encode = rcp(token, ulps)
    rcp6 = rcp(np.float32(6.0), rcp6_ulps)
    bmax = np.fmax.reduce(a.reshape(-1, 16), axis=1).astype(np.float32)
    if order == 0:
        scale = mul(encode, mul(bmax, rcp6))
    elif order == 1:
        scale = mul(mul(encode, bmax), rcp6)
    else:
        scale = mul(mul(encode, rcp6), bmax)
    scale = torch.from_numpy(np.clip(scale, -448, 448).astype(np.float32))
    return [float(token)] + scale.to(torch.float8_e4m3fn).view(torch.uint8).tolist()


class TestProbe(CustomTestCase):
    """probe_nvfp4 with FlashInfer's quantize and the epilogue faked: both at every
    global scale of probe_scales, at every geometry and in both modes; the mode that
    matches at all of them, or why neither does."""

    GEOMETRIES = ((6, 1), (4, 1))

    def _probe(self, wrong=lambda ftz, ulps, geometry: False, e4m3_max=448.0):
        """probe_nvfp4's result and FlashInfer's calls' scales, the epilogue writing
        FlashInfer's bytes but one block scale where ``wrong``."""
        scales = {s.item(): u for u, s in hop_ffn.probe_scales(e4m3_max, "cpu")}
        flashinfer_calls, epilogue_calls = [], []

        def outputs(x, scale):
            m, k = x.shape
            fp4 = torch.full((m, k // 2), 0x2C, dtype=torch.uint8)
            sf = torch.full((m, k // 16), 0x38, dtype=torch.uint8).view(
                torch.float8_e4m3fn
            )
            return fp4, sf, x.float().abs().amax(dim=1) * scale

        def quantize(x, scale, *, sfLayout, per_token_activation, backend):
            self.assertEqual(
                (sfLayout, per_token_activation, backend), ("linear", True, "cute-dsl")
            )
            flashinfer_calls.append(scale.item())
            return outputs(x, scale)

        def row_quant(x, scale, *, ftz, cluster_size, vecs_per_thread):
            ulps = scales[scale.item()]
            epilogue_calls.append((ftz, ulps, (cluster_size, vecs_per_thread)))
            fp4, sf, pts = outputs(x, scale)
            sf = sf.view(torch.uint8).reshape(-1).clone()
            if wrong(ftz, ulps, (cluster_size, vecs_per_thread)):
                sf[5] ^= 1
            return fp4, sf, pts

        fake = SimpleNamespace(
            SfLayout=SimpleNamespace(layout_linear="linear"), nvfp4_quantize=quantize
        )
        with (
            patch.dict(sys.modules, {"flashinfer": fake}),
            patch.object(hop_ffn, "config_reason", return_value=""),
            patch.object(mn, "e4m3_max", return_value=e4m3_max),
            patch.object(ops, "nvfp4_row_quant", row_quant),
        ):
            got = hop_ffn.probe_nvfp4(self.GEOMETRIES, hidden=6144, device="cpu")
        self.assertEqual(flashinfer_calls, list(scales))
        self.assertEqual(
            epilogue_calls,
            [
                (ftz, u, g)
                for u in hop_ffn.PROBE_SCALE_ULPS
                for ftz in (False, True)
                for g in self.GEOMETRIES
            ],
        )
        return got

    def test_every_global_scale_runs(self):
        for e4m3_max in mn.E4M3_MAXES:
            with self.subTest(e4m3_max=e4m3_max):
                self.assertEqual(self._probe(e4m3_max=e4m3_max), (False, ""))

    def test_a_mode_must_match_at_every_scale(self):
        # subnormals kept off by a byte at +2 ulps only: the flushed mode
        self.assertEqual(
            self._probe(lambda ftz, ulps, g: not ftz and ulps == 2), (True, "")
        )
        ftz, why = self._probe(
            lambda ftz, ulps, g: (
                (ulps == 0) if not ftz else (ulps == -3 and g == (4, 1))
            )
        )
        self.assertIsNone(ftz)
        self.assertIn(
            "subnormals kept: block scales at cluster 6 x 1 vectors, block scales at "
            "cluster 4 x 1 vectors; flushed: block scales at cluster 4 x 1 vectors, "
            "global scale -3 ulps",
            why,
        )


if __name__ == "__main__":
    unittest.main()
