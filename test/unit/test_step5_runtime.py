# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Step-5 runtime on CPU: which steps plan() covers, what build() builds and says,
what the loaded model must be, and the slot registry later slots plug into."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import step3p7
from sglang.srt.models import step5 as sgl_step5

try:
    from sglang.kernels.plugin_kernels.megakernel.spec import BuildContext, Step
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import BuildContext, Step

from sgl_nvidia_mega.step5 import runtime, spec
from sgl_nvidia_mega.step5.runtime import SlotInfo
from sgl_nvidia_mega.step5.runtime.executor import Step5Verify

WIDTHS = tuple(range(4, 33, 4))
CTX = BuildContext(widths=WIDTHS, tokens_per_request=4, deadline_s=60.0, binding="b")


def _verify():
    return Step5Verify(
        None, slots={}, bound={}, widths=WIDTHS, binding=None, summary="", parity=None
    )


def _plan(rows, mode=ForwardMode.TARGET_VERIFY, capture=True, breakable=False, **kw):
    fb = SimpleNamespace(forward_mode=mode)
    parity = kw.pop("parity", "0")
    with (
        patch(
            "sglang.srt.model_executor.runner.get_is_capture_mode",
            return_value=capture,
        ),
        patch(
            "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph."
            "is_in_breakable_cuda_graph",
            return_value=breakable,
        ),
        patch.dict("os.environ", {"SGLANG_NVIDIA_MEGAKERNEL_PARITY": parity}),
    ):
        return _verify().plan(torch.zeros(rows, dtype=torch.int64), fb, **kw)


@pytest.mark.parametrize("rows", WIDTHS)
def test_a_captured_verify_step_of_a_built_width_is_covered(rows):
    assert _plan(rows) == Step(mode="target_verify", rows=rows)


@pytest.mark.parametrize(
    "kw",
    [
        dict(rows=36),  # batch 9: past the multicast tier
        dict(rows=2),  # not a whole number of requests
        dict(rows=4, mode=ForwardMode.DECODE),
        dict(rows=4, mode=ForwardMode.EXTEND),
        dict(rows=4, mode=ForwardMode.IDLE),
        dict(rows=4, capture=False),  # an eager step under FULL graphs
        dict(rows=4, breakable=True),
        dict(rows=4, get_embedding=True),
        dict(rows=4, pp_proxy_tensors=object()),
    ],
)
def test_every_other_step_runs_sglangs_forward(kw):
    assert _plan(**kw) is None


def test_the_parity_switch_covers_eager_steps():
    assert _plan(8, capture=False, parity="1") == Step(mode="target_verify", rows=8)


def _model(layers):
    """A Step5ForConditionalGeneration over a Step5ForCausalLM holding these layers."""
    vl = step3p7.Step3p7ForConditionalGeneration.__new__(
        step3p7.Step3p7ForConditionalGeneration
    )
    lm = sgl_step5.Step5ForCausalLM.__new__(sgl_step5.Step5ForCausalLM)
    lm.__dict__["model"] = SimpleNamespace(
        start_layer=0, end_layer=len(layers), layers=layers
    )
    vl.__dict__["language_model"] = lm
    return vl


def _layer(i, comm=True, **kw):
    attn = SimpleNamespace(
        is_dsa=i % 4 == 3,
        alt_stream="alt",
        qkv_gate_weight=object(),
        _qkv_gate_layer=object(),
    )
    moe = SimpleNamespace(
        use_mega_moe=False,
        use_nvfp4_experts=False,
        attn_dp_size=1,
        alt_stream="alt",
        kernel_routing=3 <= i <= 87,
        fuse_shared_finalize=3 <= i <= 87,
        keep_bf16_experts=88 <= i <= 90,
        bf16_trtllm=88 <= i <= 90,
    )
    # Step5BoundaryFusions: 32 fused rows with the multicast communicator, else 4
    fusions = SimpleNamespace(
        max_fused_rows=32 if comm else 4, allreduce_comm=object() if comm else None
    )
    layer = SimpleNamespace(
        layer_id=i,
        self_attn=attn,
        use_dp_attention_reduce=False,
        use_moe=3 <= i <= 90,
        moe=moe,
        boundary_fusions=fusions,
    )
    for path, value in kw.items():
        owner, _, name = path.rpartition(".")
        setattr(getattr(layer, owner) if owner else layer, name, value)
    return layer


def test_the_deployed_model_has_no_load_time_refusal():
    assert runtime.model_disable_reasons(_model([_layer(i) for i in range(92)])) == []


@pytest.mark.parametrize(
    "layer, kw, want",
    [
        (
            5,
            {"self_attn.qkv_gate_weight": None},
            "layer 5: q/k/v and the gate are not one fused GEMM",
        ),
        (
            7,
            {"self_attn.alt_stream": None},
            "layer 7: no alternate stream for the indexer",
        ),
        (
            40,
            {"moe.use_nvfp4_experts": True},
            "layer 40: MegaMoE, NVFP4 experts or attention DP",
        ),
        (
            40,
            {"moe.fuse_shared_finalize": False},
            "layer 40: FP8 experts without in-kernel routing",
        ),
        (
            40,
            {"moe.alt_stream": None},
            "layer 40: no alternate stream for the shared expert",
        ),
        (
            89,
            {"moe.bf16_trtllm": False},
            "layer 89: BF16 experts not on FlashInfer TRT-LLM",
        ),
        (0, {"use_dp_attention_reduce": True}, "layer 0: attention DP"),
    ],
)
def test_a_layer_off_the_mirrored_branches_is_refused(layer, kw, want):
    layers = [_layer(i, **(kw if i == layer else {})) for i in range(92)]
    assert want in "\n".join(runtime.model_disable_reasons(_model(layers)))


def test_a_partial_layer_stack_is_refused():
    layers = [_layer(i) for i in range(46)]
    assert "not all 92" in "\n".join(runtime.model_disable_reasons(_model(layers)))


def test_another_model_is_refused():
    assert "is not the Step-5 VL model" in runtime.model_disable_reasons(object())[0]


def _build(comm=True):
    with patch(
        "sglang.srt.layers.moe.utils.get_moe_runner_backend",
        return_value=SimpleNamespace(value="flashinfer_trtllm"),
    ):
        return runtime.build(_model([_layer(i, comm=comm) for i in range(92)]), CTX)


def test_the_build_covers_batch_1_to_8_and_says_so():
    built = _build()
    assert built.widths == WIDTHS and built.draft_sampler is None
    assert built.binding == "b" and built.parity is None
    assert built.bound["ffn.runner_defers"] is True
    assert built.summary() == (
        "slots none (stock); verify rows 4-32 (batch 1-8); boundaries Step-5 multicast "
        "at <= 32 rows; experts FP8 3-87 deferred finalize, BF16 88-90"
    )


def test_the_build_names_flashinfer_boundaries_without_the_multicast_communicator():
    """Without SGLang's Step-5 multicast communicator its whole fused tier is off:
    every boundary is FlashInfer's and the FP8 MoE finalizes inside the FFN."""
    built = _build(comm=False)
    assert built.bound["boundary.multicast"] is False
    assert built.summary() == (
        "slots none (stock); verify rows 4-32 (batch 1-8); FlashInfer boundaries, no "
        "multicast communicator (finalize in the FFN); experts FP8 3-87, BF16 88-90"
    )


def test_the_attention_switch_is_named_in_the_summary(monkeypatch):
    monkeypatch.setattr(runtime, "ATTN_MODE", "module")
    assert (
        _build()
        .summary()
        .endswith(
            "; attention Step5Attention.forward whole "
            "(SGLANG_NVIDIA_MEGAKERNEL_STEP5_ATTN=module)"
        )
    )


def test_the_build_refuses_another_verify_width():
    ctx = BuildContext(
        widths=tuple(range(6, 49, 6)), tokens_per_request=6, deadline_s=60
    )
    with pytest.raises(RuntimeError, match="built for 4 verify rows a request, not 6"):
        runtime.build(_model([_layer(i) for i in range(92)]), ctx)


def test_a_requested_slot_that_builds_no_width_stops_the_build(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"attn_slot"}))
    with pytest.raises(RuntimeError, match=r"requested slots \['attn_slot'\] built no"):
        _build()


def test_spec_and_runtime_agree_on_the_stock_numerics():
    assert runtime.SLOTS == frozenset() and runtime.slot_reasons() == []
    assert runtime.SHIPPED_SLOTS == () and runtime.BITWISE_SLOTS == frozenset()
    assert spec.numerics(frozenset()).kind.value == "bitwise"


@pytest.mark.parametrize(
    "raw, want",
    [
        (None, frozenset()),  # unset: the shipped slots (none yet)
        ("", frozenset()),  # the stock step
        (" a, b ,", frozenset({"a", "b"})),
    ],
)
def test_the_slots_variable(monkeypatch, raw, want):
    if raw is None:
        monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    else:
        monkeypatch.setenv(runtime.SLOTS_ENV, raw)
    assert runtime.requested_slots() == want


# A registry with two slots, as later slots fill it: a bitwise FFN slot that runs on a
# tolerance attention slot
TABLE = {
    "ffn_slot": SlotInfo(
        seam="ffn.experts", numerics="bitwise", max_rows=32, needs=("attn_slot",)
    ),
    "attn_slot": SlotInfo(
        seam="attention.core",
        numerics="tolerance",
        max_rows=32,
        tolerances={"attn_slot.rel": 3e-3},
    ),
}


def test_the_registry_names_a_missing_dependency_and_the_numerics(monkeypatch):
    monkeypatch.setattr(runtime, "SLOT_TABLE", TABLE)
    assert runtime.slot_reasons(frozenset({"ffn_slot"})) == [
        "SGLANG_NVIDIA_MEGAKERNEL_SLOTS: slot 'ffn_slot' runs on ['attn_slot']"
    ]
    both = frozenset(TABLE)
    assert runtime.slot_reasons(both) == []
    assert spec.numerics(frozenset({"ffn_slot"})).kind.value == "bitwise"
    numerics = spec.numerics(both)
    assert numerics.kind.value == "tolerance"
    assert numerics.tolerances == {"attn_slot.rel": 3e-3}


def test_the_attention_switch_refuses_an_attention_slot(monkeypatch):
    monkeypatch.setattr(runtime, "SLOT_TABLE", TABLE)
    monkeypatch.setattr(runtime, "ATTN_MODE", "module")
    assert runtime.attention_mode_reasons(frozenset({"ffn_slot"})) == []
    assert runtime.attention_mode_reasons(frozenset(TABLE)) == [
        "SGLANG_NVIDIA_MEGAKERNEL_STEP5_ATTN=module runs SGLang's attention module, "
        "which leaves out slot 'attn_slot' (attention.core)"
    ]


@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(numerics="exact"), "bitwise or tolerance"),
        (dict(numerics="tolerance"), "states its tolerances"),
        (dict(tolerances={"x": 1.0}), "states its tolerances"),
        (dict(max_rows=36), "outside 1-32"),
    ],
)
def test_a_mis_declared_slot_is_rejected(kw, want):
    with pytest.raises(ValueError, match=want):
        SlotInfo(**{"seam": "ffn.experts", "numerics": "bitwise", "max_rows": 32, **kw})
