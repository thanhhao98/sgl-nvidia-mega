# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of the MiniMax M3.1 megakernel's runtime: the covered-step plan, the
executor's stock step against SGLang's own forward, and the build's per-width slot
plan, its row bounds, the ranks' agreement on it and the refusal of a requested slot
that serves no width."""

import contextlib
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import Step  # noqa: E402
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import Step  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

from sgl_nvidia_mega.minimax_m31 import attention, boundary  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import moe as m31_moe  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import plan as m31_plan  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import runtime as m31_runtime  # noqa: E402
from sgl_nvidia_mega.minimax_m31.attention import qk_store  # noqa: E402
from sgl_nvidia_mega.minimax_m31.boundary import hop_ffn  # noqa: E402
from sgl_nvidia_mega.minimax_m31.moe import moe_nofill as mn  # noqa: E402
from sgl_nvidia_mega.minimax_m31.runtime import agree as m31_agree  # noqa: E402
from sgl_nvidia_mega.minimax_m31.runtime import executor as m31_executor  # noqa: E402
from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll  # noqa: E402


class _Mode:
    def __init__(self, name):
        self.name = name

    def is_target_verify(self):
        return self.name == "target_verify"

    def is_decode(self):
        return self.name == "decode"

    def is_extend(self):
        return self.name in ("extend", "target_verify")


def _batch(mode):
    return SimpleNamespace(forward_mode=_Mode(mode))


class TestPlan(CustomTestCase):
    def _built(self, tokens):
        widths = m31_plan.served_widths(tuple(range(tokens, 129, tokens)), tokens)
        return m31_executor.MegakernelDecode(
            None, widths=widths, tokens_per_request=tokens, plan={}, bound={}
        )

    def test_widths(self):
        self.assertEqual(self._built(8).widths, tuple(range(8, 129, 8)))
        self.assertEqual(self._built(1).widths, tuple(range(1, 17)))

    def test_verify_steps_of_built_widths_are_covered_only_while_capturing(self):
        built = self._built(8)
        ids = torch.zeros(16, dtype=torch.long)
        with patch.object(m31_executor, "serves_this_step", return_value=True):
            self.assertEqual(
                built.plan(ids, _batch("target_verify")),
                Step(mode="target_verify", rows=16),
            )
            self.assertEqual(
                built.plan(ids, _batch("target_verify"), pp_proxy_tensors=None),
                Step(mode="target_verify", rows=16),
            )
            for mode in ("decode", "extend"):
                self.assertIsNone(built.plan(ids, _batch(mode)))
            self.assertIsNone(built.plan(torch.zeros(12), _batch("target_verify")))
            self.assertIsNone(built.plan(torch.zeros(136), _batch("target_verify")))
            self.assertIsNone(built.plan(ids, _batch("target_verify"), True))
            self.assertIsNone(
                built.plan(ids, _batch("target_verify"), pp_proxy_tensors=object())
            )
        with patch.object(m31_executor, "serves_this_step", return_value=False):
            self.assertIsNone(built.plan(ids, _batch("target_verify")))

    def test_decode_steps_without_speculation(self):
        built = self._built(1)
        with patch.object(m31_executor, "serves_this_step", return_value=True):
            self.assertEqual(
                built.plan(torch.zeros(5), _batch("decode")),
                Step(mode="decode", rows=5),
            )
            self.assertIsNone(built.plan(torch.zeros(17), _batch("decode")))
            self.assertIsNone(built.plan(torch.zeros(5), _batch("target_verify")))


class _Trace:
    """Fake SGLang modules that record each call and transform their input, so that
    two forwards that make the same calls on the same data give equal outputs."""

    def __init__(self):
        self.calls = []

    def add(self, *call):
        self.calls.append(call)


def _fake_lm(trace: _Trace, *, num_layers=5, capture=(), dense=2, hidden=8):
    from sglang.srt.models.minimax_m3 import MiniMaxM3DecoderLayer, MiniMaxM3Model
    from sglang.srt.models.minimax_m3_vl import (
        MiniMaxM3SparseForConditionalGeneration as RealVL,
    )

    class Exit:
        def __init__(self, i):
            self.i = i

        def __enter__(self):
            trace.add("ffn.exit.enter", self.i)
            return self

        def __exit__(self, *exc):
            trace.add("ffn.exit.leave", self.i)
            return False

        def finish(self, x):
            trace.add("ffn.exit.finish", self.i)
            return x - 0.5

    class Boundary:
        def __init__(self, name, i):
            self.name, self.i = name, i

        def prepare(self, x, fb, capture_gathered=None, **kw):
            trace.add(f"{self.name}.prepare", self.i, capture_gathered is not None, kw)
            y = x * 1.5 + self.i
            if capture_gathered is not None:
                capture_gathered.capture(y + 100)
            return y

        def finish(self, x, fb):
            trace.add(f"{self.name}.finish", self.i)
            return x + 0.25

        def exit(self, fb):
            trace.add(f"{self.name}.exit", self.i)
            return Exit(self.i)

    class Attn(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            self.i = i

        def forward(self, positions, hidden_states, forward_batch):
            trace.add("self_attn", self.i, positions.tolist())
            return hidden_states * 2 + self.i

    class Mlp(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            self.i = i

        def forward(self, x, forward_batch=None):
            trace.add("mlp", self.i, forward_batch is not None)
            return x + 3 + self.i

    class Layer(torch.nn.Module):
        forward = MiniMaxM3DecoderLayer.forward

        def __init__(self, i):
            super().__init__()
            self.attn_boundary = Boundary("attn", i)
            self.ffn_boundary = Boundary("ffn", i)
            self.self_attn, self.mlp = Attn(i), Mlp(i)
            self.is_layer_sparse = i >= dense
            if i in capture:
                self._is_layer_to_capture = True

    class Embed(torch.nn.Module):
        def forward(self, ids):
            trace.add("embed", ids.tolist())
            return ids.float()[:, None].expand(-1, hidden) + 0.125

    class Model(torch.nn.Module):
        forward = MiniMaxM3Model.forward
        get_input_embeddings = MiniMaxM3Model.get_input_embeddings

        def __init__(self):
            super().__init__()
            self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
            self.embed_tokens = Embed()
            self.layers = torch.nn.ModuleList(Layer(i) for i in range(num_layers))
            self.start_layer, self.end_layer = 0, num_layers
            self.norm = "norm"
            self.finalize_norm = "finalize"
            self.layers_to_capture = [i for i in capture] + (
                [num_layers] if capture else []
            )

    class Lm(torch.nn.Module):
        forward = RealVL.forward

        def __init__(self):
            super().__init__()
            self.model = Model()
            self.pp_group = SimpleNamespace(is_last_rank=True)
            self.is_mrope_enabled = False
            self.capture_aux_hidden_states = bool(capture)
            self.lm_head = "lm_head"

        def logits_processor(self, input_ids, hidden, lm_head, fb, aux):
            trace.add(
                "logits", input_ids.tolist(), lm_head, None if aux is None else len(aux)
            )
            return SimpleNamespace(
                next_token_logits=hidden * 7,
                hidden_states=None if aux is None else torch.cat(list(aux), dim=-1),
            )

    return Lm()


def _patched_runtime(trace: _Trace):
    """SGLang's runtime queries both forwards make, faked alike."""
    import contextlib

    from sglang.srt.eplb import expert_distribution
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.model_executor import cuda_graph_config
    from sglang.srt.model_executor.runner_backend_utils import breakable_cuda_graph
    from sglang.srt.models import minimax_m3

    @contextlib.contextmanager
    def with_current_layer(i):
        trace.add("recorder", i)
        yield

    recorder = SimpleNamespace(with_current_layer=with_current_layer)

    def start(fb):
        trace.add("residual.start")

    def final_norm(x, fb, norm, *, finalize_norm, capture, skip_empty):
        trace.add(
            "residual.final_norm", norm, finalize_norm, capture is not None, skip_empty
        )
        if capture is not None:
            capture(x + 1000)
        return x / 4

    stack = contextlib.ExitStack()
    for target, name, value in (
        (minimax_m3, "check_cuda_graph_backend", lambda *a: False),
        (cuda_graph_config, "check_cuda_graph_backend", lambda *a: False),
        (minimax_m3, "is_in_breakable_cuda_graph", lambda: False),
        (breakable_cuda_graph, "is_in_breakable_cuda_graph", lambda: False),
        (minimax_m3, "get_global_expert_distribution_recorder", lambda: recorder),
        (
            expert_distribution,
            "get_global_expert_distribution_recorder",
            lambda: recorder,
        ),
        (residual_batch, "start", start),
        (residual_batch, "final_norm", final_norm),
    ):
        stack.enter_context(patch.object(target, name, value))
    return stack


class TestStockStep(CustomTestCase):
    """With every slot off the executor makes SGLang's calls in SGLang's order."""

    def _compare(self, mode, rows, **lm_kwargs):
        ids = torch.arange(rows, dtype=torch.long) + 3
        positions = torch.arange(rows, dtype=torch.long) + 40
        runs = []
        for use_executor in (False, True):
            trace = _Trace()
            lm = _fake_lm(trace, **lm_kwargs)
            fb = SimpleNamespace(
                forward_mode=_Mode(mode), input_embeds=None, can_run_tbo=False
            )
            with _patched_runtime(trace):
                if use_executor:
                    built = m31_executor.MegakernelDecode(
                        lm,
                        widths=(rows,),
                        tokens_per_request=8 if mode == "target_verify" else 1,
                        plan={rows: m31_plan.Slots()},
                        bound={},
                    )
                    step = Step(mode=mode, rows=rows)
                    out = built.forward(step, lm.forward, ids, positions, fb)
                else:
                    out = lm(ids, positions, fb)
            runs.append((trace.calls, out))
        (want_calls, want), (got_calls, got) = runs
        self.assertEqual(got_calls, want_calls)
        self.assertTrue(torch.equal(got.next_token_logits, want.next_token_logits))
        if want.hidden_states is None:
            self.assertIsNone(got.hidden_states)
        else:
            self.assertTrue(torch.equal(got.hidden_states, want.hidden_states))
        return want_calls

    def test_dspark_verify_step_with_captures(self):
        calls = self._compare("target_verify", 16, capture=(1, 3))
        self.assertIn(("logits", list(range(3, 19)), "lm_head", 3), calls)

    def test_decode_step_without_captures(self):
        calls = self._compare("decode", 4)
        self.assertEqual(calls[-1], ("logits", [3, 4, 5, 6], "lm_head", None))

    def test_parity_mode_runs_both_forwards_and_returns_the_executor_output(self):
        rows = 8
        ids = torch.arange(rows, dtype=torch.long) + 3
        positions = torch.arange(rows, dtype=torch.long) + 40
        outs = {}
        for parity in ("0", "1", "control"):
            trace = _Trace()
            lm = _fake_lm(trace, capture=(2,))
            fb = SimpleNamespace(
                forward_mode=_Mode("target_verify"),
                input_embeds=None,
                can_run_tbo=False,
            )
            built = m31_executor.MegakernelDecode(
                lm,
                widths=(rows,),
                tokens_per_request=8,
                plan={rows: m31_plan.Slots()},
                bound={},
            )
            with (
                _patched_runtime(trace),
                patch.object(m31_executor, "PARITY", parity != "0"),
                patch.object(m31_executor, "PARITY_MODE", parity),
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                ),
                self.assertLogs(m31_executor.logger, level="INFO") as logs,
            ):
                out = built.forward(
                    Step(mode="target_verify", rows=rows),
                    lm.forward,
                    ids,
                    positions,
                    fb,
                )
                m31_executor.logger.info("end")
            outs[parity] = (trace.calls, out, logs.output)
        calls0, out0, _ = outs["0"]
        for parity in ("1", "control"):
            calls, out, log = outs[parity]
            # SGLang's forward, then the executor's (or SGLang's again): the same calls
            self.assertEqual(calls, calls0 + calls0)
            self.assertTrue(torch.equal(out.next_token_logits, out0.next_token_logits))
            self.assertTrue(torch.equal(out.hidden_states, out0.hidden_states))
            line = [x for x in log if "megakernel parity" in x]
            self.assertEqual(len(line), 1, log)
            self.assertIn(
                "rows=8 logits_max_abs=0 (rel 0.00e+00, 0.00% differ)", line[0]
            )
            self.assertIn("hidden_max_abs=0 (rel", line[0])
            self.assertEqual("parity control" in line[0], parity == "control")


class TestSlotPlan(CustomTestCase):
    """The build's per-width slot plan (``plan.plan`` over the widths each package's
    build serves) against the per-slot ``serves`` comprehension it replaced."""

    def test_plan_equals_the_per_slot_serves(self):
        for tokens, widths in ((8, tuple(range(8, 129, 8))), (1, tuple(range(1, 17)))):
            # each package's build result, within its kernels' limits: slot off,
            # built for no width, built for some or every width
            moe_bounds = (None, mn.Bound(), mn.Bound(widths=widths))
            attention_bounds = (None,) + tuple(
                attention.AttnLowLatency(
                    layers={},
                    widths=tuple(w for w in widths if w <= limit),
                    tactics={},
                    reasons=[],
                    skipped={},
                )
                for limit in (0, mxfp8_ll.MAX_ROWS)
            )
            boundary_bounds = (None,) + tuple(
                SimpleNamespace(serves=lambda rows, limit=limit: rows <= limit)
                for limit in (0, 64, 128)
            )
            qk_store_bounds = (None, qk_store.Bound(), qk_store.Bound(widths=widths))
            ctx = SimpleNamespace(
                widths=widths, tokens_per_request=tokens, binding=None
            )
            for (
                moe_bound,
                attention_bound,
                boundary_bound,
                qk_store_bound,
                hop_ffn_bound,
            ) in itertools.product(
                moe_bounds,
                attention_bounds,
                boundary_bounds,
                qk_store_bounds,
                boundary_bounds,
            ):
                want = {
                    w: m31_plan.Slots(
                        moe_nofill=m31_moe.serves(moe_bound, w),
                        attn_ll=attention.serves(attention_bound, w),
                        hop_mxfp8=boundary.serves(boundary_bound, w),
                        qk_store=qk_store.serves(qk_store_bound, w),
                        # its MoE inputs are moe_nofill's
                        hop_ffn=hop_ffn.serves(hop_ffn_bound, w)
                        and m31_moe.serves(moe_bound, w),
                    )
                    for w in widths
                }
                with (
                    patch.object(m31_runtime, "SLOTS", frozenset()),
                    patch.object(m31_moe, "build", return_value=moe_bound),
                    patch.object(attention, "build", return_value=attention_bound),
                    patch.object(boundary, "build", return_value=boundary_bound),
                    patch.object(qk_store, "build", return_value=qk_store_bound),
                    patch.object(hop_ffn, "build", return_value=hop_ffn_bound),
                    # one rank: a gather returns this rank's value
                    patch.object(m31_agree, "gather_tp", lambda obj: [obj]),
                ):
                    built = m31_runtime.build(SimpleNamespace(), ctx)
                self.assertEqual(built.slot_plan, want)
                self.assertEqual(built.widths, widths)

    def test_plan_bounds_are_the_kernels_limits(self):
        from sglang.kernels.ops.communication.all_reduce_fusion import (
            MXFP8_QUANT_MAX_TOKENS,
        )

        self.assertEqual(m31_plan.ATTN_LL_MAX_ROWS, mxfp8_ll.MAX_ROWS)
        self.assertEqual(m31_plan.HOP_MXFP8_MAX_ROWS, MXFP8_QUANT_MAX_TOKENS)
        self.assertEqual(m31_plan.HOP_FFN_MAX_ROWS, MXFP8_QUANT_MAX_TOKENS)
        # a build whose widths reach past a kernel's bound serves only those within it
        every = frozenset(range(1, 2 * m31_plan.MAX_ROWS + 1))
        facts = m31_plan.Facts(
            moe_nofill=every, attn_ll=every, hop_mxfp8=every, hop_ffn=every
        )
        for rows in range(1, m31_plan.MAX_ROWS + 1):
            self.assertEqual(
                m31_plan.slots(rows, facts),
                m31_plan.Slots(
                    moe_nofill=True,
                    attn_ll=rows <= mxfp8_ll.MAX_ROWS,
                    hop_mxfp8=rows <= MXFP8_QUANT_MAX_TOKENS,
                    hop_ffn=rows <= MXFP8_QUANT_MAX_TOKENS,
                ),
            )
        with self.assertRaises(ValueError):
            m31_plan.slots(m31_plan.MAX_ROWS + 1, facts)
        # hop_mxfp8's bound is the covered steps' own; a lower one would cap it too
        with patch.object(m31_plan, "HOP_MXFP8_MAX_ROWS", 64):
            self.assertTrue(m31_plan.slots(64, facts).hop_mxfp8)
            self.assertFalse(m31_plan.slots(72, facts).hop_mxfp8)
        # hop_ffn serves only where moe_nofill does (its MoE inputs are moe_nofill's)
        nofill = m31_plan.Facts(moe_nofill=frozenset(range(1, 65)), hop_ffn=every)
        self.assertEqual(
            [r for r in range(1, 129) if m31_plan.slots(r, nofill).hop_ffn],
            list(range(1, 65)),
        )


@contextlib.contextmanager
def _builds(slots, *, moe=None, attn=None, hop=None, gather=lambda obj: [obj]):
    """``runtime.build`` with ``slots`` requested, the packages' builds returning these
    results, and the TP group's gather (one rank by default)."""
    with (
        patch.object(m31_runtime, "SLOTS", frozenset(slots)),
        patch.object(m31_moe, "build", return_value=moe),
        patch.object(attention, "build", return_value=attn),
        patch.object(boundary, "build", return_value=hop),
        patch.object(m31_agree, "gather_tp", gather),
    ):
        yield


VERIFY = SimpleNamespace(
    widths=tuple(range(8, 129, 8)), tokens_per_request=8, binding=None
)


class TestBuildAgreesAndRefuses(CustomTestCase):
    """The build's agreement over the TP group on the slot plan (each package's build
    decides from its own rank), and its refusal of a requested slot that serves no
    width, naming each package's reason."""

    def test_ranks_agree_on_the_plan(self):
        hop = SimpleNamespace(serves=lambda rows: rows <= 64)
        other = m31_plan.Facts(hop_mxfp8=frozenset(range(8, 33, 8)))
        gathered = []

        def gather(facts):
            # rank 2's hop build served fewer widths
            gathered.append(facts)
            return [facts, facts, other, facts]

        with (
            _builds({"hop_mxfp8"}, hop=hop, gather=gather),
            self.assertRaises(RuntimeError) as caught,
        ):
            m31_runtime.build(SimpleNamespace(), VERIFY)
        (facts,) = gathered
        self.assertEqual(facts, m31_plan.Facts(hop_mxfp8=frozenset(range(8, 65, 8))))
        self.assertEqual(
            str(caught.exception),
            "--enable-nvidia-megakernel: the ranks' slot plans (widths per slot) "
            f"differ: rank 0: {facts}; rank 1: {facts}; rank 2: {other}; "
            f"rank 3: {facts}",
        )
        with _builds({"hop_mxfp8"}, hop=hop, gather=lambda facts: [facts] * 4):
            built = m31_runtime.build(SimpleNamespace(), VERIFY)
        self.assertEqual(
            [w for w, s in built.slot_plan.items() if s.hop_mxfp8],
            list(range(8, 65, 8)),
        )

    def test_build_gathers_the_facts_once(self):
        gather = Mock(side_effect=lambda obj: [obj])
        hop = SimpleNamespace(serves=lambda rows: rows <= 64)
        with _builds({"hop_mxfp8"}, hop=hop, gather=gather):
            built = m31_runtime.build(SimpleNamespace(), VERIFY)
        gather.assert_called_once()
        (facts,) = gather.call_args.args
        self.assertIsInstance(facts, m31_plan.Facts)
        self.assertEqual(facts, m31_plan.Facts(hop_mxfp8=frozenset(range(8, 65, 8))))
        self.assertEqual(built.slot_plan, m31_plan.plan(VERIFY.widths, facts))

    def test_the_missing_slot_refusal_names_each_package(self):
        ctx = SimpleNamespace(widths=(8, 16), tokens_per_request=8, binding=None)
        moe = mn.Bound(
            reason="rank 1: layer 3: the module has forward or backward hooks"
        )
        attn = attention.AttnLowLatency(
            layers={},
            widths=(),
            tactics={},
            reasons=["no layer's attention is the branch the slot reproduces"],
            skipped={},
        )
        first = "the stack's first attention input (the embeddings' norm)"
        no_fusion = "no push-plane fusion of hidden 6144"
        hop = boundary.HopMxfp8(
            widths=(),
            hops={},
            refused={0: first, 1: no_fusion, 2: no_fusion},
            hidden=64,
            device="cpu",
        )
        prefix = "NVIDIA megakernel: requested slots {} serve no width of [8, 16]"
        with (
            _builds(
                {"moe_nofill", "attn_ll", "hop_mxfp8"}, moe=moe, attn=attn, hop=hop
            ),
            self.assertRaises(RuntimeError) as caught,
        ):
            m31_runtime.build(SimpleNamespace(), ctx)
        self.assertEqual(
            str(caught.exception),
            prefix.format(["attn_ll", "hop_mxfp8", "moe_nofill"])
            + f" (moe_nofill: {moe.reason}; attn_ll: {attn.reasons[0]}; "
            f"hop_mxfp8: {no_fusion}; {first}; qk_store: not requested; hop_ffn: not "
            "requested)",
        )
        # a slot that is off, and one that serves
        hop = boundary.HopMxfp8(
            widths=(8, 16), hops={1: None}, refused={0: first}, hidden=64, device="cpu"
        )
        with (
            _builds({"moe_nofill", "hop_mxfp8"}, moe=moe, hop=hop),
            self.assertRaises(RuntimeError) as caught,
        ):
            m31_runtime.build(SimpleNamespace(), ctx)
        self.assertEqual(
            str(caught.exception),
            prefix.format(["moe_nofill"])
            + f" (moe_nofill: {moe.reason}; attn_ll: not requested; hop_mxfp8: built; "
            "qk_store: not requested; hop_ffn: not requested)",
        )
        # the slot qk_store's build names why it serves no width
        qk = qk_store.Bound(reason="rank 0: no layer's attention is the branch")
        with (
            _builds({"qk_store"}),
            patch.object(qk_store, "build", return_value=qk),
            self.assertRaises(RuntimeError) as caught,
        ):
            m31_runtime.build(SimpleNamespace(), ctx)
        self.assertEqual(
            str(caught.exception),
            prefix.format(["qk_store"])
            + " (moe_nofill: not requested; attn_ll: not requested; hop_mxfp8: not "
            f"requested; qk_store: {qk.reason}; hop_ffn: not requested)",
        )


if __name__ == "__main__":
    unittest.main()
