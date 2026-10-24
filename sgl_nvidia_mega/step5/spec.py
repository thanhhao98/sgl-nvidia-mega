# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Step-5 megakernel's spec: the setups it serves and its build.

SGLang checks the requirements at argument resolution and the model at load
(``sglang.kernels.plugin_kernels.megakernel``, ``sglang.srt.megakernels`` before
sgl-project/sglang#45389); an unsupported setup stops the server on every rank,
listing every reason. This module stays light: SGLang imports it in the launcher.

The envelope is the deployed setup on GB300 and B300 at TP4: FA4 attention on page
64, FlashInfer TRT-LLM FP8 MoE, the TRT-LLM all-reduce fusion backend (the one the
Step-5 multicast communicator is built for), BF16 KV, EAGLE 3 / 1 / 4 with multi-layer
MTP, decode CUDA graphs FULL.
"""

from __future__ import annotations

import os

try:
    from sglang.kernels.plugin_kernels.megakernel.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

from sgl_nvidia_mega.step5 import (
    ATTENTION_PROFILE,
    DSA_LAYERS,
    MOE_LAYERS,
    QUANTIZATION,
    TEXT_CONFIG,
    VL_MODEL_TYPE,
    runtime,
)
from sgl_nvidia_mega.step5.plan import MAX_ROWS

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/megakernel/spec.py|srt/megakernels/spec.py": (
        "Feature",
        "SPECULATIVE_DECODING",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "BITWISE",
        "TOLERANCE",
        "Requirements",
        "decode_graph_backends",
        "config_disable_reasons",
        "model_disable_reasons",
    ),
    # the decode CUDA graph backend _config_disable_reasons reads
    "srt/model_executor/cuda_graph_config.py": ("decode", "backend", '"disabled"'),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/arg_groups/fields/*.py": (
        "cuda_graph_config",
        "attention_backend",
        "decode_attention_backend",
        "prefill_attention_backend",
        "speculative_attention_mode",
        "page_size",
        "kv_cache_dtype",
        "moe_runner_backend",
        "moe_a2a_backend",
        "flashinfer_allreduce_fusion_backend",
        "enforce_disable_flashinfer_allreduce_fusion",
        "boundary_reduction",
        "bf16_gemm_backend",
        "enable_symm_mem",
        "enable_torch_symm_mem",
        "enable_mscclpp",
        "disable_custom_all_reduce",
        "enable_quant_communications",
        "speculative_algorithm",
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
        "enable_multi_layer_eagle",
        "speculative_adaptive",
        "dp_size",
        "startup_weight_load_mode",
    ),
    "srt/configs/step5.py": (
        "get_step5_text_config",
        "parse_step5_attention_profile",
        "num_hidden_layers",
        "moe_layer_list",
        "layer_types_with_mtp",
        '"full_attention"',
    ),
    "srt/environ.py": ("SGLANG_STEP5_QUANTIZE_BF16_MOE",),
    # the Step-5 view the covered steps read (BINDING_MEMBERS)
    "srt/kernel_plugins/megakernel/binding.py|srt/megakernels/binding.py": (
        "ForwardView",
        "step5",
        "Step5View",
        "swa_kv_write_target",
        "indexer_metadata",
        "swa_head_gate",
        "dsa_head_gate",
    ),
}

# the SGLang megakernel contract this spec is written against
API_VERSION = 1

# Test-only switch that runs a covered step eagerly (decode CUDA graphs off): the
# composition check, runtime.parity.PARITY_ENV
EAGER_TEST_SWITCHES = ("SGLANG_NVIDIA_MEGAKERNEL_PARITY",)


def eager_steps_allowed() -> bool:
    """Eager covered steps run only under a test-only switch: the served path is the
    one captured in the decode CUDA graphs."""
    return any(os.environ.get(k, "") not in ("", "0") for k in EAGER_TEST_SWITCHES)


def serves_this_step() -> bool:
    """A covered step runs the megakernel while SGLang captures its decode CUDA graphs
    (their warm-up runs included), or eagerly under a test-only switch; any other
    step, e.g. an eager one under FULL graphs, runs SGLang's forward."""
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return get_is_capture_mode() or eager_steps_allowed()


def decode_graph_backends() -> frozenset:
    """FULL decode CUDA graphs; eager steps too under a test-only switch."""
    return frozenset({"full", "disabled"} if eager_steps_allowed() else {"full"})


REQUIREMENTS = Requirements(
    # GB300 and B300 (device_isa reports sm_103 on both)
    isas=frozenset({"sm_103"}),
    tp_sizes=frozenset({4}),
    dtypes=frozenset({"bfloat16"}),
    decode_graph_backends=decode_graph_backends(),
    # as resolved before the megakernel hook (pipeline.py), for the deployed launch
    server_args={
        # Step5AttentionBackend over FA4 for prefill and decode alike
        # (attention_registry.py); FA4 applies the SWA head gate in its epilogue
        "attention_backend": frozenset({"fa4"}),
        "decode_attention_backend": frozenset({None, "fa4"}),
        "prefill_attention_backend": frozenset({None, "fa4"}),
        "speculative_attention_mode": frozenset({"prefill"}),
        # the Step-5 KV pool requires 64 (mem_cache/step5_kv_pool.py)
        "page_size": frozenset({64}),
        # "auto" stays "auto" until ModelRunner.configure_kv_cache_dtype: BF16 here
        "kv_cache_dtype": frozenset({"auto", "bf16", "bfloat16"}),
        "moe_runner_backend": frozenset({"flashinfer_trtllm"}),
        "moe_a2a_backend": frozenset({"none"}),
        # the Step-5 multicast communicator exists only for TP4/EP1 + trtllm
        # (step5_allreduce.py, _get_comm); Step-5 is not auto-enabled there
        # (arg_groups/overrides.py)
        "flashinfer_allreduce_fusion_backend": frozenset({"trtllm"}),
        "enforce_disable_flashinfer_allreduce_fusion": frozenset({False}),
        "boundary_reduction": frozenset({"ar"}),
        # the deployed GEMM dispatch (TGV / cuBLAS by shape; the indexer keys on "auto")
        "bf16_gemm_backend": frozenset({"auto"}),
        # the deployed collectives: CustomAllReduceV2 with multicast, no symmetric
        # memory
        "enable_symm_mem": frozenset({False}),
        "enable_torch_symm_mem": frozenset({False}),
        "enable_mscclpp": frozenset({False}),
        "disable_custom_all_reduce": frozenset({False}),
        "enable_quant_communications": frozenset({None, False}),
        # uniform verify steps of 4 rows a request (NEXTN resolves to EAGLE)
        "speculative_algorithm": frozenset({"EAGLE"}),
        "speculative_num_steps": frozenset({3}),
        "speculative_eagle_topk": frozenset({1}),
        "speculative_num_draft_tokens": frozenset({4}),
        "enable_multi_layer_eagle": frozenset({True}),
        "speculative_adaptive": frozenset({False}),
        "dp_size": frozenset({1}),
        # the step code reads the weights SGLang loaded, after its last conversion
        "startup_weight_load_mode": frozenset({"serial"}),
    },
    # the text shapes live in text_config, which config_disable_reasons checks
    hf_config={"model_type": VL_MODEL_TYPE},
    # layers 88-90 keep their BF16 experts (FlashInfer TRT-LLM BF16 MoE)
    env={"SGLANG_STEP5_QUANTIZE_BF16_MOE": False},
)


def numerics(slots) -> Numerics:
    """The claim for the requested slots: bitwise unless a tolerance slot is on (an
    unknown name is refused by runtime.slot_reasons, not here)."""
    if runtime.bitwise(slots):
        return Numerics(kind=NumericsKind.BITWISE)
    return Numerics(kind=NumericsKind.TOLERANCE, tolerances=runtime.tolerances(slots))


NUMERICS = numerics(runtime.SLOTS)


def text_config_reasons(hf_config) -> list[str]:
    """The text_config the step code is written for: its shapes, the layer pattern
    and the attention profile."""
    from sglang.srt.configs.step5 import (
        get_step5_text_config,
        parse_step5_attention_profile,
    )

    text = get_step5_text_config(hf_config)
    if text is None:
        return ["config.text_config is not a Step-5 text config (Step5Config)"]
    reasons = [
        f"config.text_config.{key} {getattr(text, key, None)!r} is not {want!r}"
        for key, want in TEXT_CONFIG.items()
        if getattr(text, key, None) != want
    ]
    if tuple(text.moe_layer_list) != MOE_LAYERS:
        reasons.append("config.text_config: MoE layers are not 3-90")
    dsa = tuple(
        i
        for i, t in enumerate(text.layer_types_with_mtp[: text.num_hidden_layers])
        if t == "full_attention"
    )
    if dsa != DSA_LAYERS:
        reasons.append(f"config.text_config: DSA layers {dsa} are not 3, 7, ..., 91")
    try:
        profile = parse_step5_attention_profile(text)
    except (AssertionError, ValueError) as e:  # it asserts on any other profile
        return reasons + [f"config.text_config: no Step-5 attention profile ({e})"]
    reasons += [
        f"attention profile {key} {getattr(profile, key, None)!r} is not {want!r}"
        for key, want in ATTENTION_PROFILE.items()
        if getattr(profile, key, None) != want
    ]
    return reasons


def quantization_reasons(hf_config) -> list[str]:
    """The FP8 checkpoint (its quantization config is at the VL config's top level)."""
    quant = getattr(hf_config, "quantization_config", None) or {}
    get = quant.get if isinstance(quant, dict) else lambda k: getattr(quant, k, None)
    return [
        f"config.quantization_config.{key} {get(key)!r} is not {want!r}"
        for key, want in QUANTIZATION.items()
        if get(key) != want
    ]


# The SGLang binding members the covered steps read: the Step-5 view
BINDING_MEMBERS = {
    "ForwardView": ("step5",),
    "Step5View": (
        "swa_kv_write_target",
        "indexer_metadata",
        "swa_head_gate",
        "dsa_head_gate",
    ),
}


def binding_reasons() -> list[str]:
    """The Step-5 view of SGLang's megakernel binding, each member present (none is
    read when the attention runs SGLang's module whole: runtime.ATTN_ENV)."""
    if runtime.ATTN_MODE == "module":
        return []
    try:
        import sglang.srt.kernel_plugins.megakernel.binding as binding
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        import sglang.srt.megakernels.binding as binding
    members = dict(BINDING_MEMBERS)
    if "swa_attn" in runtime.SLOTS:
        # the slot reads the FA4 forward metadata (page table, sequence lengths)
        members["Step5View"] = (*members["Step5View"], "swa_metadata")
    missing = [
        f"{owner}.{member}"
        for owner, members_of in members.items()
        for member in members_of
        if not hasattr(getattr(binding, owner, None), member)
    ]
    if missing:
        return [
            "this SGLang's megakernel binding lacks the Step-5 view members "
            f"{', '.join(missing)}; update SGLang"
        ]
    return []


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    reasons = text_config_reasons(hf_config) + quantization_reasons(hf_config)
    reasons += binding_reasons() + runtime.slot_reasons()
    reasons += runtime.attention_mode_reasons()
    if cfg.cuda_graph_config.decode.backend == "disabled" and not eager_steps_allowed():
        reasons.append(
            "decode CUDA graphs are disabled: covered steps run only in captured "
            "decode graphs (eagerly only under SGLANG_NVIDIA_MEGAKERNEL_PARITY)"
        )
    return reasons


def _model_disable_reasons(model, hf_config) -> list[str]:
    return runtime.model_disable_reasons(model)


def _build(model, hf_config, ctx):
    return runtime.build(model, ctx)


SPEC = MegakernelSpec(
    name="Step-5",
    requires=REQUIREMENTS,
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=NUMERICS,
    # 1-8 requests of 4 verify rows
    max_rows=MAX_ROWS,
    build=_build,
    api_version=API_VERSION,
    model_disable_reasons=_model_disable_reasons,
    config_disable_reasons=_config_disable_reasons,
)
