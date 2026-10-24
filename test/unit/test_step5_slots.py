# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Step-5 slots on CPU: where each serves and declines, what it issues, and on
which stream (the stubs of test_step5_steps.py).

- moe_quant_alt: on an FP8 MoE layer the stock op log plus exactly the quantization on
  the alternate stream after the fork, its event, the main stream's wait for it and the
  pair handed to the experts; nothing on a BF16 layer; each refusal names its reason.
- swa_attn: an SWA layer's FA4 call is replaced by the slot (no RadixAttention call, no
  separate head gate, o_proj on the slot's output); any other path stops the step.
- build: the facts per width, the kernels bound, refusals before any kernel is built.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from test_step5_steps import (
    Log,
    Step5View,
    Stream,
    T,
    _attention,
    _layer_ffn,
    streams,
    tag,
)

from sgl_nvidia_mega.step5 import attention, ffn, plan, runtime, spec
from sgl_nvidia_mega.step5.attention import swa_attn
from sgl_nvidia_mega.step5.ffn import moe_quant_alt

# --------------------------------------------------------------- moe_quant_alt


class _Event:
    def __init__(self, log):
        self.log, self.name = log, "ready"

    def record(self, stream):
        self.log.append(("event.record", stream.name))


def _ffn_log(layer_id, slots, kind="fp8"):
    log = Log()
    layer = _layer_ffn(log, kind=kind, rows=8, can_fuse=True)
    layer.__dict__["layer_id"] = layer_id
    x = T(log, "x", (8, 4096))
    fb = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY, num_token_non_padded="ntnp"
    )
    pair = (T(log, "a_q", (8, 4096), torch.float8_e4m3fn), T(log, "a_sf", (8, 32)))
    quant = log.op("quant", pair)
    st = SimpleNamespace(fb=fb, bound={"ffn.runner_defers": True}, slots=slots)
    with (
        streams(log, capture=True),
        patch.object(
            Stream,
            "wait_event",
            create=True,
            new=lambda self, ev: log.append(("wait_event", self.name, ev.name)),
        ),
        patch.object(torch.cuda, "Event", side_effect=lambda: _Event(log)),
        patch(
            "sglang.kernels.ops.quantization.fp8_kernel.per_token_group_quant_fp8",
            side_effect=quant,
        ),
    ):
        out = ffn.step(layer, x, st, defer=True)
    log.append(("return", type(out).__name__))
    return list(log)


def test_moe_quant_alt_moves_the_quantization_to_the_alternate_stream():
    stock = _ffn_log(3, plan.Slots())
    slot = _ffn_log(3, plan.Slots(moe_quant_alt=True))
    router = next(i for i, e in enumerate(stock) if e[0] == "router")
    inserted = [
        ("quant", "alt", ("x", 128), (("column_major_scales", True),)),
        ("event.record", "alt"),
        ("record", "a_q", "main"),
        ("record", "a_sf", "main"),
        ("wait_event", "main", "ready"),
    ]
    want = stock[: router + 1] + inserted + stock[router + 1 :]
    experts = next(i for i, e in enumerate(want) if e[0] == "experts_deferred")
    name, stream, args, kw = want[experts]
    want[experts] = (name, stream, args, kw + (("pre_quant_input", ("a_q", "a_sf")),))
    assert slot == want, (
        "\n".join(map(str, slot)) + "\n----\n" + "\n".join(map(str, want))
    )
    # the shared expert still follows on the alternate stream, then the join
    shared = next(i for i, e in enumerate(slot) if e[0] == "shared")
    assert slot[shared][1] == "alt" and shared > experts


def test_moe_quant_alt_leaves_the_bf16_tails_alone():
    assert _ffn_log(88, plan.Slots(moe_quant_alt=True), kind="bf16") == _ffn_log(
        88, plan.Slots(), kind="bf16"
    )


def _moe_model(**overrides):
    def method(i):
        mxfp8 = overrides.get("mxfp8_layer") == i
        block = [128, 128] if overrides.get("block_layer") != i else [1, 128]
        return SimpleNamespace(
            block_quant=True,
            quant_config=SimpleNamespace(weight_block_size=block, use_mxfp8=mxfp8),
        )

    layers = [
        SimpleNamespace(
            moe=SimpleNamespace(experts=SimpleNamespace(quant_method=method(i)))
        )
        for i in range(92)
    ]
    return SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
    )


def test_moe_quant_alt_runs_on_deepseek_fp8_experts_only(monkeypatch):
    bound = {"ffn.runner_defers": True}
    assert moe_quant_alt.unsupported(_moe_model(), bound) == []
    assert moe_quant_alt.unsupported(_moe_model(mxfp8_layer=40), bound) == [
        "slot moe_quant_alt: the experts of layers [40] are not DeepSeek FP8 with "
        "128 x 128 blocks"
    ]
    assert "[7]" in moe_quant_alt.unsupported(_moe_model(block_layer=7), bound)[0]
    assert moe_quant_alt.unsupported(_moe_model(), {"ffn.runner_defers": False}) == [
        "slot moe_quant_alt: the MoE runner backend is not FlashInfer TRT-LLM"
    ]
    from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm

    monkeypatch.delattr(flashinfer_trtllm, "deepseek_fp8_moe_input")
    assert (
        "takes no pre-quantized input"
        in moe_quant_alt.unsupported(_moe_model(), bound)[0]
    )


def test_moe_quant_alt_needs_the_deferred_finalize_experts():
    log = Log()
    layer = _layer_ffn(log, kind="fp8", rows=8, can_fuse=True)
    layer.__dict__["layer_id"] = 3
    st = SimpleNamespace(bound={"ffn.runner_defers": False})
    with pytest.raises(RuntimeError, match="do not run FlashInfer TRT-LLM's deferred"):
        ffn.experts(
            layer, T(log, "x", (8, 4096)), "topk", st, pre_quant_input=("q", "s")
        )


# --------------------------------------------------------------------- swa_attn


@pytest.mark.skipif(Step5View is None, reason="no megakernel Step-5 view")
@pytest.mark.parametrize("capture", [False, True])
def test_swa_attn_replaces_the_fa4_call(capture):
    log = Log()
    a, backend, qkvg_out = _attention(
        log, dsa=False, rows=8, kv_target=True, dsa_gate=False
    )
    md = SimpleNamespace(
        swa_spec_metadata=None,
        swa_page_table=T(log, "swa_pt", (2, 16), torch.int32),
        cache_seqlens_int32=T(log, "seqlens", (2,), torch.int32),
    )
    backend.swa_backend = SimpleNamespace(forward_metadata=md)
    x = T(log, "x", (8, 4096))
    pos = T(log, "pos", (8,), torch.int64)
    fb = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
    exec_ns = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=False)
    )
    calls = []

    def attend(layer, q, gate, kv_write, st, *, gate_epilogue):
        calls.append((tag(q), tag(gate), tag(kv_write), gate_epilogue))
        return T(log, "swa_out", (8, 3072))

    layer = SimpleNamespace(self_attn=a, layer_id=4)
    st = SimpleNamespace(
        fb=fb,
        positions=pos,
        step5=Step5View(backend),
        slots=plan.Slots(swa_attn=True),
        rows=8,
    )
    with (
        streams(log, capture),
        patch("sglang.srt.runtime_context.get_exec", return_value=exec_ns),
        patch(
            "sglang.kernels.ops.attention.step5.qknorm_rope.step5_qknorm_rope_gated",
            side_effect=log.op("qknorm", qkvg_out),
        ),
        patch.object(swa_attn, "attend", side_effect=attend),
    ):
        out = attention.step(layer, x, st)
    names = [e[0] for e in log]
    assert "attn" not in names and "core" not in names  # no RadixAttention call
    assert calls == [("q", "gate", ("k_cache", "v_cache", "swa_loc"), True)]
    o_proj = next(e for e in log if e[0] == "o_proj")
    assert o_proj[2] == ("swa_out.flat",)  # the slot's output, gated already
    assert tag(out) == "o_out"


def _attend_st(md, rows=8):
    kernels = []
    st = SimpleNamespace(
        step5=SimpleNamespace(swa_metadata=md),
        rows=rows,
        bound={"swa_attn.kernels": lambda *args: kernels.append(args) or "out"},
    )
    return st, kernels


def test_swa_attn_reads_fa4s_inputs_and_stops_on_any_other_path():
    layer = SimpleNamespace(layer_id=4)
    table = torch.zeros(8, 16, dtype=torch.int32)
    seqlens = torch.arange(8, dtype=torch.int32)
    md = SimpleNamespace(
        swa_spec_metadata=None, swa_page_table=table, cache_seqlens_int32=seqlens
    )
    kv = ("k_rows", "v_rows", "loc")
    st, kernels = _attend_st(md, rows=12)
    assert swa_attn.attend(layer, "q", "gate", kv, st, gate_epilogue=True) == "out"
    (args,) = kernels
    assert args[:4] == ("q", "gate", "k_rows", "v_rows")
    assert args[4].shape == (3, 16) and torch.equal(args[5], seqlens[:3])
    with pytest.raises(RuntimeError, match="gate epilogue False"):
        swa_attn.attend(layer, "q", "gate", kv, st, gate_epilogue=False)
    with pytest.raises(RuntimeError, match="fused store False"):
        swa_attn.attend(layer, "q", "gate", None, st, gate_epilogue=True)
    spec_md = SimpleNamespace(
        swa_spec_metadata=object(), swa_page_table=table, cache_seqlens_int32=seqlens
    )
    with pytest.raises(RuntimeError, match="SWA page table"):
        swa_attn.attend(
            layer, "q", "gate", kv, _attend_st(spec_md)[0], gate_epilogue=True
        )


def _swa_model(**attn_overrides):
    def layer(i):
        dsa = i in (3, 7)
        attn = SimpleNamespace(
            tp_q_head_num=16,
            tp_k_head_num=1,
            head_dim=192,
            sliding_window_size=-1 if dsa else 511,
            scaling=192**-0.5,
        )
        if not dsa and i == 5:
            attn.__dict__.update(attn_overrides)
        return SimpleNamespace(
            layer_id=i, self_attn=SimpleNamespace(is_dsa=dsa, attn=attn)
        )

    layers = [layer(i) for i in range(9)]
    return SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
    )


def test_swa_attn_runs_on_the_shapes_its_kernels_take():
    exec_ns = lambda det: SimpleNamespace(  # noqa: E731
        deterministic=SimpleNamespace(enable_deterministic_inference=det)
    )
    with patch("sglang.srt.runtime_context.get_exec", return_value=exec_ns(False)):
        assert swa_attn.unsupported(_swa_model()) == []
        (reason,) = swa_attn.unsupported(_swa_model(tp_k_head_num=2))
        assert "SWA layer 5" in reason and "'tp_k_head_num': 2" in reason
        (reason,) = swa_attn.unsupported(_swa_model(scaling=0.1))
        assert "'scaling': 0.1" in reason
    with patch("sglang.srt.runtime_context.get_exec", return_value=exec_ns(True)):
        assert swa_attn.unsupported(_swa_model()) == [
            "slot swa_attn: deterministic inference keeps SGLang's FA4 reduction order"
        ]


# ------------------------------------------------------------- registry and build


def test_the_registry_holds_both_slots_opt_in():
    assert runtime.SLOT_TABLE["moe_quant_alt"].numerics == "bitwise"
    assert runtime.SLOT_TABLE["swa_attn"].numerics == "tolerance"
    assert runtime.SHIPPED_SLOTS == ()
    assert spec.numerics(frozenset({"moe_quant_alt"})).kind.value == "bitwise"
    numerics = spec.numerics(frozenset({"moe_quant_alt", "swa_attn"}))
    assert numerics.kind.value == "tolerance"
    assert numerics.tolerances == runtime.TOLERANCES["swa_attn"]


@pytest.mark.parametrize("rows", [4, 8, 16, 32])
def test_a_slot_serves_every_covered_width_it_was_built_for(rows):
    facts = plan.Facts(moe_quant_alt=True, swa_attn=False)
    assert plan.slots(rows, facts) == plan.Slots(moe_quant_alt=True, swa_attn=False)
    assert plan.slots(rows, plan.Facts()) == plan.Slots()


def test_the_build_refuses_before_it_builds_a_kernel(monkeypatch):
    from test_step5_runtime import _build

    monkeypatch.setattr(runtime, "SLOTS", frozenset({"moe_quant_alt", "swa_attn"}))
    monkeypatch.setattr(moe_quant_alt, "unsupported", lambda model, bound: ["no pair"])
    monkeypatch.setattr(swa_attn, "unsupported", lambda model: ["no shape"])
    built = []
    monkeypatch.setattr(swa_attn, "build", lambda widths: built.append(widths))
    with pytest.raises(RuntimeError, match="NVIDIA megakernel: no pair; no shape"):
        _build()
    assert built == []


def test_the_build_binds_the_kernels_and_plans_both_slots(monkeypatch):
    from test_step5_runtime import _build

    monkeypatch.setattr(runtime, "SLOTS", frozenset({"moe_quant_alt", "swa_attn"}))
    monkeypatch.setattr(moe_quant_alt, "unsupported", lambda model, bound: [])
    monkeypatch.setattr(swa_attn, "unsupported", lambda model: [])
    monkeypatch.setattr(swa_attn, "build", lambda widths: ("kernels", widths))
    built = _build()
    assert built.bound["swa_attn.kernels"] == ("kernels", built.widths)
    assert set(built.slots.values()) == {plan.Slots(moe_quant_alt=True, swa_attn=True)}
