# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Step-5 spec through SGLang's startup checks on CPU, as the plugin registers it:
the deployed setup passes, and every other setup is refused at argument resolution
with each reason named.

The HF config is SGLang's own Step5VLConfig built from the checkpoint's config.json
(``fixtures/step5_preview_config.json``: stepfun-ai/Step-5-Preview at revision
364b3b03fed2477aa0953fe855a2c4c8c48ee3cc). The ServerArgs carry the values the
resolution pipeline gives the deployed launch before the megakernel hook runs (NEXTN
-> EAGLE, multi-layer EAGLE forced on, boundary reduction "ar", kv_cache_dtype still
"auto"). On an SGLang without the Step-5 view of the megakernel binding
(``ForwardView.step5``) the cases that accept a setup skip, naming it.
"""

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    default_cuda_graph_config,
    with_phase,
)
from sglang.srt.server_args import ServerArgs

try:
    from sglang.kernels.plugin_kernels.megakernel import spec as registry
    from sglang.srt.kernel_plugins.megakernel import binding as guards

    # the startup checks and the binding are one module there
    sgl_binding = guards
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels import binding as sgl_binding
    from sglang.srt.megakernels import guards
    from sglang.srt.megakernels import spec as registry

from sgl_nvidia_mega import plugin
from sgl_nvidia_mega.step5 import ARCHITECTURE, runtime, spec

CONFIG_JSON = Path(__file__).parent / "fixtures" / "step5_preview_config.json"
NO_STEP5_VIEW = "this SGLang has no megakernel Step-5 view (ForwardView.step5)"
requires_view = pytest.mark.skipif(
    not hasattr(getattr(sgl_binding, "ForwardView", None), "step5"),
    reason=NO_STEP5_VIEW,
)

DEPLOYED = dict(
    model_path="dummy",
    enable_nvidia_megakernel=True,
    tp_size=4,
    ep_size=1,
    attention_backend="fa4",
    page_size=64,
    moe_runner_backend="flashinfer_trtllm",
    flashinfer_allreduce_fusion_backend="trtllm",
    speculative_algorithm="EAGLE",
    speculative_num_steps=3,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=4,
    enable_multi_layer_eagle=True,
    boundary_reduction="ar",
    max_running_requests=64,
    chunked_prefill_size=16384,
    swa_full_tokens_ratio=0.1,
    cuda_graph_config=default_cuda_graph_config(),
)


def _vl_config(text=None, top=None):
    from sglang.srt.configs.step5 import Step5VLConfig

    raw = json.loads(CONFIG_JSON.read_text())
    raw["text_config"] = {**raw["text_config"], **(text or {})}
    raw.update(top or {})
    return Step5VLConfig(**raw)


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    plugin.register()


@pytest.fixture
def without_view(monkeypatch):
    """SGLang's megakernel binding as before its Step-5 view."""
    monkeypatch.delattr(sgl_binding.ForwardView, "step5", raising=False)
    monkeypatch.delattr(sgl_binding, "Step5View", raising=False)


def _refusal(text=None, top=None, dtype="bfloat16", isa="sm_103", env=None, **args):
    """SGLang's resolution step on these arguments: "" when it accepts them, else its
    error message."""
    hf_config = _vl_config(text, top)
    with (
        contextlib.ExitStack() as stack,
        patch.object(guards, "plugin_reasons", return_value=[]),
        patch.object(guards, "device_isa", return_value=isa),
        patch.object(guards, "model_config_of") as model_config,
    ):
        model_config.return_value = SimpleNamespace(hf_config=hf_config, dtype=dtype)
        for name, value in (env or {}).items():
            stack.enter_context(getattr(envs, name).override(value))
        try:
            guards.handle_nvidia_megakernel(ServerArgs(**{**DEPLOYED, **args}))
        except ValueError as e:
            return str(e)
    return ""


@requires_view
def test_the_deployed_setup_is_accepted():
    assert _refusal() == ""


def test_an_sglang_without_the_step5_view_is_refused(without_view):
    message = _refusal()
    assert "lacks the Step-5 view members ForwardView.step5" in message, message


def test_the_attention_switch_needs_no_step5_view(without_view, monkeypatch):
    """Test only: SGLang's attention module whole asks its backend itself."""
    monkeypatch.setattr(runtime, "ATTN_MODE", "module")
    assert _refusal() == ""


def test_an_unknown_attention_switch_value_is_refused(monkeypatch):
    monkeypatch.setattr(runtime, "ATTN_MODE", "kernel")
    message = _refusal()
    assert "SGLANG_NVIDIA_MEGAKERNEL_STEP5_ATTN 'kernel' is not 'module'" in message


BREAKABLE = with_phase(
    default_cuda_graph_config(), Phase.DECODE, backend=Backend.BREAKABLE
)
DISABLED = with_phase(
    default_cuda_graph_config(), Phase.DECODE, backend=Backend.DISABLED
)


@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(tp_size=8), "tp_size 8 is not one of ['4']"),
        (dict(isa="sm_100"), "not sm_100"),
        (dict(dtype="float16"), "dtype 'float16'"),
        (dict(attention_backend="fa3"), "attention_backend 'fa3'"),
        (dict(page_size=128), "page_size 128"),
        (dict(kv_cache_dtype="fp8_e4m3"), "kv_cache_dtype 'fp8_e4m3'"),
        (dict(moe_runner_backend="flashinfer_trtllm_routed"), "moe_runner_backend"),
        (
            dict(flashinfer_allreduce_fusion_backend="mnnvl"),
            "flashinfer_allreduce_fusion_backend 'mnnvl'",
        ),
        (
            dict(speculative_num_steps=5, speculative_num_draft_tokens=6),
            "speculative_num_draft_tokens 6",
        ),
        (dict(enable_multi_layer_eagle=False), "enable_multi_layer_eagle False"),
        (
            dict(startup_weight_load_mode="overlap"),
            "startup_weight_load_mode 'overlap'",
        ),
        (dict(ep_size=4), "expert parallelism is in use"),
        (dict(cuda_graph_config=BREAKABLE), "decode CUDA graphs"),
        (
            dict(cuda_graph_config=DISABLED),
            "covered steps run only in captured decode graphs",
        ),
        (
            dict(text=dict(moe_top_k=4, num_experts_per_tok=4)),
            "config.text_config.moe_top_k 4 is not 8",
        ),
        (
            dict(text=dict(sliding_window=1024)),
            "config.text_config.sliding_window 1024",
        ),
        (dict(enable_hierarchical_cache=True), "KV offload is in use"),
        (dict(enable_dp_attention=True, dp_size=2), "DP attention is in use"),
        (dict(enable_torch_compile=True), "torch-compiled decode is in use"),
        (
            dict(top=dict(quantization_config={"quant_method": "modelopt_fp4"})),
            "quant_method 'modelopt_fp4'",
        ),
        (
            dict(env={"SGLANG_STEP5_QUANTIZE_BF16_MOE": True}),
            "SGLANG_STEP5_QUANTIZE_BF16_MOE True is not False",
        ),
    ],
)
def test_each_refusal_names_its_reason(kw, want):
    message = _refusal(**kw)
    assert want in message, message


@requires_view
def test_the_parity_switch_runs_eager_steps(monkeypatch):
    """The stock-parity serving test steps eagerly; only its switch lets a launch
    without decode CUDA graphs through."""
    monkeypatch.setenv("SGLANG_NVIDIA_MEGAKERNEL_PARITY", "1")
    assert spec.decode_graph_backends() == {"full", "disabled"}
    requires = msgspec.structs.replace(
        spec.SPEC.requires, decode_graph_backends=spec.decode_graph_backends()
    )
    monkeypatch.setattr(
        spec, "SPEC", msgspec.structs.replace(spec.SPEC, requires=requires)
    )
    assert _refusal(cuda_graph_config=DISABLED) == ""


def test_an_unknown_slot_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"swa_core"}))
    message = _refusal()
    assert (
        "SGLANG_NVIDIA_MEGAKERNEL_SLOTS: unknown slot 'swa_core' "
        "(known: moe_quant_alt, swa_attn)" in message
    ), message
    # the spec still imports with an unknown name requested: it claims no tolerance
    assert spec.numerics(frozenset({"swa_core"})).kind.value == "bitwise"


def test_the_vl_architecture_resolves_and_the_mtp_draft_keeps_its_forward():
    """plugin main's test_plugin resolves every SPECS row, which fails on an SGLang
    without the DeepSeek-V4.1 and Kimi K3 models; the Step-5 row is checked here."""
    found, why = registry.find_spec(architectures=[ARCHITECTURE], isa="sm_103")
    assert found is spec.SPEC, why
    draft, why = registry.find_spec(architectures=["Step5MTP"], isa="sm_103")
    assert draft is None and "no megakernel spec is registered for ['Step5MTP']" in why


def test_the_stock_build_is_bitwise_and_covers_batch_1_to_8():
    assert spec.SPEC.numerics.kind.value == "bitwise"
    assert spec.SPEC.max_rows == 32
    assert spec.SPEC.features == {spec.Feature.SPECULATIVE_DECODING}
