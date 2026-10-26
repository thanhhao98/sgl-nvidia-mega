# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of the MiniMax M3.1 megakernel's spec: the M3.1 recipe through SGLang's
startup checks, the refusals of every other setup with each reason named, the text
config checks, the numerics and the slot requests, and the refusals, before the
weights load, of what the slots cannot run on: an SGLang without their ops, a
missing kernel toolchain, a FlashInfer without their names, overlapped weight
loading."""

import importlib.machinery
import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

from minimax_m31_fixtures import hf_config, m31_server_args, text_config  # noqa: E402

try:
    from sglang.kernels.plugin_kernels.nvidia import spec as registry  # noqa: E402
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind  # noqa: E402
    from sglang.srt.kernel_plugins.nvidia import binding as guards  # noqa: E402
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels import guards  # noqa: E402
    from sglang.srt.megakernels import spec as registry  # noqa: E402
    from sglang.srt.megakernels.spec import NumericsKind  # noqa: E402

from sglang.test.test_utils import CustomTestCase  # noqa: E402

from sgl_nvidia_mega import plugin  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import plan as m31_plan  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import runtime as m31_runtime  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import spec as m31_spec  # noqa: E402

# the import-spec probe itself, which the toolchains fixture stands in for
_real_has_module = m31_spec._has_module


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.delenv(plugin.PARITY_SORT_ENV, raising=False)
    plugin.register()


@pytest.fixture(autouse=True)
def toolchains(monkeypatch):
    # the kernel toolchains the slots launch through, which a CPU host lacks
    # (test_a_missing_toolchain_is_refused runs the real probe)
    monkeypatch.setattr(m31_spec, "_has_module", lambda name: True)


class TestSpec(CustomTestCase):
    def test_stock_numerics_are_bitwise(self):
        self.assertEqual(m31_spec.SPEC.numerics.kind, NumericsKind.BITWISE)
        self.assertEqual(m31_spec.SPEC.max_rows, 128)

    def test_the_m31_recipe_resolves_with_no_reason(self):
        for spec_args in (
            {},
            dict(speculative_algorithm=None, speculative_num_draft_tokens=None),
        ):
            spec, reasons = guards.megakernel_disable_reasons(
                cfg=m31_server_args(**spec_args), hf_config=hf_config(), isa="sm_103"
            )
            self.assertIs(spec, m31_spec.SPEC)
            self.assertEqual(reasons, [], spec_args)

    def test_other_recipes_are_refused_with_every_reason(self):
        _, reasons = guards.megakernel_disable_reasons(
            cfg=m31_server_args(
                moe_a2a_backend="megamoe", moe_runner_backend="deep_gemm", tp_size=8
            ),
            hf_config=hf_config(hidden_size=5120, rope_theta=1e6),
            isa="sm_103",
        )
        text = "\n".join(reasons)
        for needle in (
            "tp_size",
            "moe_a2a_backend",
            "moe_runner_backend",
            "hidden_size",
            "rope_theta",
        ):
            self.assertIn(needle, text)

    def test_text_config_checks(self):
        self.assertEqual(m31_spec.text_config_reasons(hf_config()), [])
        v5 = text_config(rope_parameters={"rope_theta": 1e7, "rope_type": "default"})
        self.assertEqual(
            m31_spec.text_config_reasons(SimpleNamespace(text_config=v5)), []
        )
        mrope = hf_config(rope_scaling={"mrope_section": [16, 24, 24]})
        self.assertTrue(m31_spec.text_config_reasons(mrope))
        dense = hf_config(moe_layer_freq=[1] * 60)
        self.assertTrue(m31_spec.text_config_reasons(dense))
        self.assertTrue(m31_spec.text_config_reasons(SimpleNamespace()))

    def test_eager_decode_is_refused_without_the_test_switch(self):
        cfg = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend="disabled")
            ),
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=8,
        )
        with patch.dict(os.environ, {"SGLANG_NVIDIA_MEGAKERNEL_PARITY": ""}):
            self.assertTrue(m31_spec._config_disable_reasons(cfg, hf_config()))
        with patch.dict(os.environ, {"SGLANG_NVIDIA_MEGAKERNEL_PARITY": "1"}):
            self.assertEqual(m31_spec._config_disable_reasons(cfg, hf_config()), [])

    def test_unknown_slots_are_refused(self):
        self.assertEqual(m31_runtime.slot_reasons(frozenset()), [])
        self.assertTrue(m31_runtime.slot_reasons(frozenset({"nonsense"})))
        self.assertEqual(
            m31_runtime.requested_slots({m31_runtime.SLOTS_ENV: " , "}), frozenset()
        )

    def test_every_slot_has_a_plan_field_and_one_numerics_entry(self):
        """A slot is a field of the per-width plan and either bitwise or declares
        its tolerances (never both); what it needs is a slot too."""
        known = (*m31_runtime.SHIPPED_SLOTS, *m31_runtime.OPT_IN_SLOTS)
        self.assertEqual(len(set(known)), len(known))
        self.assertEqual(set(m31_plan.Slots.__struct_fields__), set(known))
        self.assertEqual(set(m31_plan.Facts.__struct_fields__), set(known))
        for slot in known:
            self.assertNotEqual(
                slot in m31_runtime.BITWISE_SLOTS, slot in m31_runtime.TOLERANCES, slot
            )
        needs = set(m31_runtime.NEEDS) | set(m31_runtime.NEEDS.values())
        self.assertLessEqual(needs, set(known))


def _reasons(cfg=None):
    """Why SGLang's argument resolution refuses the M3.1 recipe on GB300 ([] when it
    accepts it)."""
    _, reasons = guards.megakernel_disable_reasons(
        cfg=m31_server_args() if cfg is None else cfg,
        hf_config=hf_config(),
        isa="sm_103",
    )
    return reasons


def test_an_sglang_without_the_epilogue_ops_is_refused(monkeypatch):
    from sglang.kernels.ops.communication import all_reduce_fusion as ops

    monkeypatch.delattr(ops, "load_residual_norm_quant")
    (reason,) = _reasons()
    assert reason.startswith("slot hop_mxfp8 needs the MXFP8 epilogue"), reason
    assert "kernels/ops/communication/all_reduce_fusion.py" in reason
    assert "load_residual_norm_quant" in reason
    # with no slot requested nothing asks for the ops
    monkeypatch.setattr(m31_spec, "SLOTS", frozenset())
    assert _reasons() == []


def test_an_sglang_without_the_entry_trigger_is_refused(monkeypatch):
    """Slot hop_mxfp8 passes the boundary kernel's entry trigger (trigger_at_entry)
    at every launch, so an SGLang whose MXFP8 epilogue lacks it is refused too."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops

    def untriggered(input, residual, weight, eps, *, world_size, **outputs):
        raise AssertionError("resolution launches nothing")

    monkeypatch.setattr(ops, "all_reduce_residual_norm_quant", untriggered)
    (reason,) = _reasons()
    assert reason.startswith("slot hop_mxfp8 needs the MXFP8 epilogue"), reason
    assert reason.endswith(
        "(kernels/ops/communication/all_reduce_fusion.py: no "
        "all_reduce_residual_norm_quant(trigger_at_entry))"
    ), reason


def test_an_sglang_without_the_fused_kv_store_is_refused(monkeypatch):
    """Slot qk_store reaches SGLang's KV pool only through the binding's view of the
    pool's fused store, which launches SGLang's fused op: an SGLang without the view,
    the store, or one of the keyword arguments the slot passes it is refused."""
    try:
        from sglang.srt.kernel_plugins.nvidia import binding
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.megakernels import binding
    from sglang.srt.mem_cache import minimax_m3_nvfp4_pool as pool

    name = "set_kv_index_buffer_fused_qknorm_rope"
    monkeypatch.setattr(m31_spec, "SLOTS", frozenset({"qk_store"}))

    def fused_store(
        self,
        layer,
        loc,
        qkv,
        groups,
        cos_sin_cache,
        positions,
        eps,
        *,
        fp8_a,
        group_a,
        fp8_b,
        group_b,
        k_group,
        index_k_group,
        cache_k,
        cache_v,
        cache_idx_k,
        k_scale=None,
        v_scale=None,
        idx_k_scale=None,
    ):
        raise AssertionError("resolution launches nothing")

    def view(self):
        return getattr(self.pool, name)

    # the view and the store (a stand-in with its keyword arguments): no reason
    monkeypatch.setattr(pool.MiniMaxM3Nvfp4KVPool, name, fused_store, raising=False)
    monkeypatch.setattr(binding.KvPoolView, name, property(view), raising=False)
    assert _reasons() == []

    # a store that takes the K / V / index-K views under other names, and no scales
    def other_store(
        self,
        layer,
        loc,
        qkv,
        groups,
        cos_sin_cache,
        positions,
        eps,
        *,
        fp8_a,
        group_a,
        fp8_b,
        group_b,
        k_group,
        index_k_group,
        k,
        v,
        idx_k,
    ):
        raise AssertionError("resolution launches nothing")

    monkeypatch.setattr(pool.MiniMaxM3Nvfp4KVPool, name, other_store)
    (reason,) = _reasons()
    assert reason.startswith(
        "slot qk_store needs SGLang's fused q/k norm + RoPE + NVFP4 KV store"
    ), reason
    missing = ("cache_k", "cache_v", "cache_idx_k", "k_scale", "v_scale", "idx_k_scale")
    assert (
        "srt/mem_cache/minimax_m3_nvfp4_pool.py: no "
        + ", ".join(f"MiniMaxM3Nvfp4KVPool.{name}({kw})" for kw in missing)
        + ")"
    ) in reason, reason
    # no view of it in the binding
    monkeypatch.setattr(pool.MiniMaxM3Nvfp4KVPool, name, fused_store)
    monkeypatch.delattr(binding.KvPoolView, name)
    (reason,) = _reasons()
    where = binding.__name__.removeprefix("sglang.").replace(".", "/") + ".py"
    assert reason.endswith(f"({where}: no KvPoolView.{name})"), reason


def test_an_sglang_without_the_nvfp4_epilogue_is_refused(monkeypatch):
    """Slot hop_ffn launches the boundary kernel's MXFP8 + NVFP4 form and probes its
    NVFP4 epilogue on rows from memory at build: an SGLang without them is refused."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops

    monkeypatch.setattr(m31_spec, "SLOTS", frozenset({"hop_ffn", "moe_nofill"}))
    if hasattr(ops, "all_reduce_residual_norm_quant_nvfp4"):
        assert _reasons() == []
    for name in ("all_reduce_residual_norm_quant_nvfp4", "nvfp4_row_quant"):
        monkeypatch.delattr(ops, name, raising=False)
    (reason,) = _reasons()
    assert reason.startswith(
        "slot hop_ffn needs the MXFP8 and NVFP4 epilogues of SGLang's push-plane "
        "boundary kernel, which this SGLang lacks"
    ), reason
    assert reason.endswith(
        "(kernels/ops/communication/all_reduce_fusion.py: no "
        "all_reduce_residual_norm_quant_nvfp4, nvfp4_row_quant)"
    ), reason


def test_a_missing_toolchain_is_refused(monkeypatch):
    monkeypatch.setattr(m31_spec, "_has_module", _real_has_module)
    # find_spec imports a dotted name's parent: an absent parent is a missing
    # module, not an import error
    assert m31_spec._has_module("no_such_package_m31.sub") is False
    # the three toolchains by their import names, whatever this host has installed
    present = importlib.machinery.ModuleSpec("present", None)
    found = {"flashinfer": present, "cutlass": None, "cuda.bindings": present}
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        m31_spec.importlib.util,
        "find_spec",
        lambda name, *a: found[name] if name in found else find_spec(name, *a),
    )
    for slot in ("moe_nofill", "attn_ll"):
        assert m31_spec.toolchain_reasons(frozenset({slot})) == [
            "needs CuTe DSL (nvidia-cutlass-dsl)"
        ], slot
    # slot hop_mxfp8 launches SGLang's ops only
    assert m31_spec.toolchain_reasons(frozenset({"hop_mxfp8"})) == []
    # with none installed, each toolchain is named once
    found.update(dict.fromkeys(found))
    slots = frozenset({"moe_nofill", "attn_ll", "hop_mxfp8"})
    assert m31_spec.toolchain_reasons(slots) == [
        "needs FlashInfer",
        "needs CuTe DSL (nvidia-cutlass-dsl)",
        "needs the CUDA Python bindings",
    ]


def test_overlap_weight_loading_is_refused():
    assert _reasons(m31_server_args(startup_weight_load_mode="overlap")) == [
        "startup_weight_load_mode 'overlap' is not one of ['serial']"
    ]


# FlashInfer's nvfp4_quantize.py as moe_nofill reads it: each name in a form FlashInfer
# provides it (an import, an aliased import, an assignment, a definition)
NVFP4_QUANTIZE = """\
from ..nvfp4_quantization_utils import (
    resolve_nvfp4_4over6,
)
from ..utils import env_flag_enabled as _env_flag_enabled

SF_LAYOUT_LINEAR = 2


def _per_token_cta_threads(k, m):
    raise NotImplementedError


def _get_compiled_kernel_nvfp4_per_token(dtype_key, k, *args, **kwargs):
    raise NotImplementedError
"""


def test_a_flashinfer_without_the_quantize_names_is_refused(tmp_path, monkeypatch):
    cfg = m31_server_args()
    root = tmp_path / "flashinfer"
    quantize = root / "quantization" / "kernels" / "nvfp4_quantize.py"
    quantize.parent.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "utils.py").write_text("def device_support_pdl(device):\n    return True\n")
    # find_spec("flashinfer") finds this stand-in; nothing imports it
    stand_in = importlib.machinery.ModuleSpec(
        "flashinfer", None, origin=str(root / "__init__.py")
    )
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        m31_spec.importlib.util,
        "find_spec",
        lambda name, *a: stand_in if name == "flashinfer" else find_spec(name, *a),
    )
    quantize.write_text(NVFP4_QUANTIZE.replace("SF_LAYOUT_LINEAR = 2\n", ""))
    assert _reasons(cfg) == [
        "FlashInfer's quantization/kernels/nvfp4_quantize.py has no SF_LAYOUT_LINEAR "
        "(slot moe_nofill)"
    ]
    # only slot moe_nofill launches through them
    assert m31_spec.flashinfer_api_reasons(frozenset({"hop_mxfp8", "attn_ll"})) == []
    quantize.write_text(NVFP4_QUANTIZE)
    assert _reasons(cfg) == []


if __name__ == "__main__":
    unittest.main()
