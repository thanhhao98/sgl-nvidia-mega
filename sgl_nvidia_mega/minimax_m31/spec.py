# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The M3.1 megakernel's spec: the setups it serves and its build.

SGLang checks the requirements at argument resolution and the loaded model at load
(``sglang.kernels.plugin_kernels.nvidia``); an unsupported setup stops the server on every rank,
listing every reason. Resolution also checks what the requested slots launch through
(SGLang's ops, the kernel toolchains, FlashInfer's names), before the weights load.
This module stays light: the launcher imports it.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
from typing import Any

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import (
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

from sgl_nvidia_mega.minimax_m31 import (
    DENSE_LAYERS,
    ROPE_THETA,
    SPARSE_ATTENTION,
    TEXT_CONFIG,
)
from sgl_nvidia_mega.minimax_m31.plan import MAX_ROWS
from sgl_nvidia_mega.minimax_m31.runtime import (
    BITWISE_SLOTS,
    SLOTS,
    slot_reasons,
    tolerances,
)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Feature",
        "SPECULATIVE_DECODING",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "BITWISE",
        "TOLERANCE",
        "Requirements",
        "decode_graph_backends",
        "model_disable_reasons",
        "config_disable_reasons",
    ),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    # SGLang's MiniMax M3 VL configs: the text config text_config_reasons reads
    "srt/configs/minimax_vl.py": ("text_config",),
    # the checkpoint's index-value flags, from which SGLang picks the layers whose
    # index V it neither loads nor caches (all of M3.1's)
    "srt/configs/model_config.py": (
        "get_minimax_sparse_disable_value_layer_ids",
        '"sparse_disable_index_value"',
    ),
    # imported by name here; srt/utils/hf_transformers_utils.py only star-imports it
    "srt/utils/hf_transformers/__init__.py": ("get_rope_config",),
    # the decode CUDA graph backend _config_disable_reasons reads
    "srt/model_executor/cuda_graph_config.py": ("decode", "backend", '"disabled"'),
    # the server arguments REQUIREMENTS names and _config_disable_reasons reads
    "srt/arg_groups/fields/*.py": (
        "quantization",
        "kv_cache_dtype",
        "page_size",
        "moe_runner_backend",
        "moe_a2a_backend",
        "speculative_algorithm",
        "speculative_num_draft_tokens",
        "cuda_graph_config",
        "startup_weight_load_mode",
    ),
    # the ops slots hop_mxfp8 and hop_ffn launch, which sglang_reasons checks for
    "kernels/ops/communication/all_reduce_fusion.py": (
        "MXFP8_QUANT_MAX_TOKENS",
        "all_reduce_residual_norm_quant",
        "load_residual_norm_quant",
        "moe_finalize_all_reduce_residual_norm_quant",
        "all_reduce_residual_norm_quant(trigger_at_entry)",
        "moe_finalize_all_reduce_residual_norm_quant(trigger_at_entry)",
        # and those slot hop_ffn launches
        "NVFP4_SF_VEC",
        "mxfp8_scale_bytes",
        "_norm_launch",
        "load_residual_norm_quant_nvfp4",
        "load_residual_norm_quant_nvfp4(ftz)",
        "all_reduce_residual_norm_quant(quantized)",
        "all_reduce_residual_norm_quant(scales)",
        "all_reduce_residual_norm_quant_nvfp4",
        "all_reduce_residual_norm_quant_nvfp4(ftz)",
        "all_reduce_residual_norm_quant_nvfp4(fp4)",
        "all_reduce_residual_norm_quant_nvfp4(fp4_scales)",
        "all_reduce_residual_norm_quant_nvfp4(token_scales)",
        "nvfp4_row_quant",
        "nvfp4_row_quant(ftz)",
        "nvfp4_row_quant(cluster_size)",
        "nvfp4_row_quant(vecs_per_thread)",
    ),
    # what slot qk_store launches through, which sglang_reasons checks for
    "kernels/ops/attention/minimax_qknorm_rope_store.py": (
        "minimax_qknorm_rope_store_fp8",
    ),
    "srt/mem_cache/minimax_m3_nvfp4_pool.py": (
        "MiniMaxM3Nvfp4KVPool",
        "set_kv_index_buffer_fused_qknorm_rope(fp8_a)",
        "set_kv_index_buffer_fused_qknorm_rope(group_a)",
        "set_kv_index_buffer_fused_qknorm_rope(fp8_b)",
        "set_kv_index_buffer_fused_qknorm_rope(group_b)",
        "set_kv_index_buffer_fused_qknorm_rope(k_group)",
        "set_kv_index_buffer_fused_qknorm_rope(index_k_group)",
        "set_kv_index_buffer_fused_qknorm_rope(cache_k)",
        "set_kv_index_buffer_fused_qknorm_rope(cache_v)",
        "set_kv_index_buffer_fused_qknorm_rope(cache_idx_k)",
        "set_kv_index_buffer_fused_qknorm_rope(k_scale)",
        "set_kv_index_buffer_fused_qknorm_rope(v_scale)",
        "set_kv_index_buffer_fused_qknorm_rope(idx_k_scale)",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "KvPoolView",
        "set_kv_index_buffer_fused_qknorm_rope",
    ),
}

# the sglang.kernels.plugin_kernels.nvidia contract this spec is written against
API_VERSION = 1

# Test-only switch that runs a covered step eagerly (decode CUDA graphs off): the
# composition check, ``runtime.executor.PARITY_ENV``
EAGER_TEST_SWITCHES = ("SGLANG_NVIDIA_MEGAKERNEL_PARITY",)


def eager_steps_allowed() -> bool:
    """Eager covered steps run only under a test-only switch: the served path is the
    one captured in the decode CUDA graphs."""
    return any(os.environ.get(k, "") not in ("", "0") for k in EAGER_TEST_SWITCHES)


def serves_this_step() -> bool:
    """A covered step runs the megakernel while SGLang captures its decode CUDA graphs
    (their warm-up runs included), or eagerly under a test-only switch; any other
    step, e.g. an eager one past the largest graph, runs SGLang's forward."""
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return get_is_capture_mode() or eager_steps_allowed()


def decode_graph_backends() -> frozenset:
    """FULL decode CUDA graphs; eager steps too under a test-only switch."""
    return frozenset({"full", "disabled"} if eager_steps_allowed() else {"full"})


REQUIREMENTS = Requirements(
    # B200 (sm_100), B300 and GB300 (sm_103)
    isas=frozenset({"sm_100", "sm_103"}),
    tp_sizes=frozenset({4}),
    dtypes=frozenset({"bfloat16"}),
    decode_graph_backends=decode_graph_backends(),
    server_args={
        # the M3.1 NVFP4 recipe: packed NVFP4 KV + index-K pages of 128 tokens
        "quantization": frozenset({"minimax_m3_nvfp4"}),
        "kv_cache_dtype": frozenset({"fp8_e4m3"}),
        "page_size": frozenset({128}),
        # the routed experts on FlashInfer's trtllm-gen routed MoE at EP 1 (not MegaMoE)
        "moe_runner_backend": frozenset({"flashinfer_trtllm_routed"}),
        "moe_a2a_backend": frozenset({"none"}),
        # DSpark's target steps are uniform verify steps; no speculation, decode steps
        "speculative_algorithm": frozenset({None, "DSPARK"}),
        # the kernels read weights in place, after SGLang's last conversion
        "startup_weight_load_mode": frozenset({"serial"}),
    },
)


def _numerics() -> Numerics:
    tol = tolerances(SLOTS)
    if SLOTS <= BITWISE_SLOTS:
        return Numerics(kind=NumericsKind.BITWISE)
    return Numerics(kind=NumericsKind.TOLERANCE, tolerances=tol)


NUMERICS = _numerics()


def text_config_reasons(hf_config: Any) -> list[str]:
    """What the step code assumes of the text model and the checkpoint does not hold."""
    text = getattr(hf_config, "text_config", None)
    if text is None:
        return ["the config has no text_config"]
    reasons = []
    for key, want in TEXT_CONFIG.items():
        got = getattr(text, key, None)
        if got != want:
            reasons.append(f"config.text_config.{key} {got!r} is not {want!r}")
    from sglang.srt.utils.hf_transformers import get_rope_config

    rope_theta, rope_scaling = get_rope_config(text)
    if rope_theta is None or float(rope_theta) != ROPE_THETA:
        reasons.append(f"rope_theta {rope_theta!r} is not {ROPE_THETA:g}")
    # transformers v5 returns rope_parameters (theta and type) even without scaling
    rope_type = (rope_scaling or {}).get("rope_type", (rope_scaling or {}).get("type"))
    if rope_scaling and (
        "mrope_section" in rope_scaling or rope_type not in (None, "default")
    ):
        reasons.append(f"rope scaling {rope_scaling!r} is set (mrope or scaled RoPE)")
    sparse = getattr(text, "sparse_attention_config", None) or {}
    for key, want in SPARSE_ATTENTION.items():
        if sparse.get(key) != want:
            reasons.append(
                f"sparse_attention_config.{key} {sparse.get(key)!r} is not {want!r}"
            )
    layers = TEXT_CONFIG["num_hidden_layers"]
    for key, want in (
        ("sparse_attention_freq", [1] * layers),
        ("sparse_disable_index_value", [1] * layers),
    ):
        if list(sparse.get(key) or []) != want:
            reasons.append(f"sparse_attention_config.{key} is not all 1")
    freq = [0] * DENSE_LAYERS + [1] * (layers - DENSE_LAYERS)
    if list(getattr(text, "moe_layer_freq", None) or []) != freq:
        reasons.append(
            f"moe_layer_freq is not {DENSE_LAYERS} dense layers then MoE layers"
        )
    return reasons


# The SGLang names each slot launches through that not every SGLang has, checked at
# resolution: slot -> (what they are, {module: names}); "C.m" is C's member m,
# "f(p, q)" f with its parameters p and q
SLOT_SGLANG = {
    # the boundary kernel's MXFP8 form, with the entry trigger it always passes
    "hop_mxfp8": (
        "the MXFP8 epilogue of SGLang's push-plane boundary kernel and its entry "
        "trigger",
        {
            "sglang.kernels.ops.communication.all_reduce_fusion": (
                "MXFP8_QUANT_MAX_TOKENS",
                "load_residual_norm_quant",
                "all_reduce_residual_norm_quant(trigger_at_entry)",
                "moe_finalize_all_reduce_residual_norm_quant(trigger_at_entry)",
            ),
        },
    ),
    # the fused q/k norm + RoPE + NVFP4 KV store: SGLang's op, the KV pool's method
    # that launches it (with the keyword arguments LayerStore passes) and the
    # binding's view of that method, the slot's only way to the pool
    "qk_store": (
        "SGLang's fused q/k norm + RoPE + NVFP4 KV store",
        {
            "sglang.kernels.ops.attention.minimax_qknorm_rope_store": (
                "minimax_qknorm_rope_store_fp8",
            ),
            "sglang.srt.mem_cache.minimax_m3_nvfp4_pool": (
                "MiniMaxM3Nvfp4KVPool.set_kv_index_buffer_fused_qknorm_rope(fp8_a, "
                "group_a, fp8_b, group_b, k_group, index_k_group, cache_k, cache_v, "
                "cache_idx_k, k_scale, v_scale, idx_k_scale)",
            ),
            "sglang.srt.kernel_plugins.nvidia.binding|sglang.srt.megakernels.binding": (
                "KvPoolView.set_kv_index_buffer_fused_qknorm_rope",
            ),
        },
    ),
    # the FFN-input boundary kernel's quantizing forms (MXFP8, and MXFP8 + NVFP4), and
    # its NVFP4 epilogue on rows from memory, which the build probes
    "hop_ffn": (
        "the MXFP8 and NVFP4 epilogues of SGLang's push-plane boundary kernel",
        {
            "sglang.kernels.ops.communication.all_reduce_fusion": (
                "MXFP8_QUANT_MAX_TOKENS",
                "NVFP4_SF_VEC",
                "mxfp8_scale_bytes",
                "_norm_launch",
                "load_residual_norm_quant",
                "load_residual_norm_quant_nvfp4(ftz)",
                "all_reduce_residual_norm_quant(quantized, scales)",
                "all_reduce_residual_norm_quant_nvfp4(ftz, fp4, fp4_scales, "
                "token_scales)",
                "nvfp4_row_quant(ftz, cluster_size, vecs_per_thread)",
            ),
        },
    ),
}

_MISSING = object()


def _lacking(module: Any, names) -> list[str]:
    """The names of ``names`` (see SLOT_SGLANG) that ``module`` lacks, a missing
    member once for all its parameters."""
    lacking = []
    for name in names:
        path, _, params = name.partition("(")
        obj = module
        for part in path.split("."):
            obj = getattr(obj, part, _MISSING)
            if obj is _MISSING:
                break
        if obj is _MISSING:
            if path not in lacking:
                lacking.append(path)
            continue
        try:
            has = inspect.signature(obj).parameters if params else {}
        except (TypeError, ValueError):
            has = {}
        lacking += [
            f"{path}({param})"
            for param in params.rstrip(")").replace(" ", "").split(",")
            if param and param not in has
        ]
    return lacking


def sglang_reasons(slots: frozenset = SLOTS) -> list[str]:
    """The SGLang names the requested slots launch through that this SGLang lacks
    ([] when it has them); it imports only the requested slots' modules."""
    reasons = []
    for slot in sorted(slots):
        what, modules = SLOT_SGLANG.get(slot, ("", {}))
        lacking = []
        for module, names in modules.items():
            # "new|old": the first location this SGLang has (a moved module)
            *newer, module = module.split("|")
            for new in newer:
                try:
                    importlib.import_module(new)
                except ModuleNotFoundError:
                    continue
                module = new
                break
            where = module.removeprefix("sglang.").replace(".", "/") + ".py"
            try:
                found = importlib.import_module(module)
            except ImportError as e:
                lacking.append(f"{where}: {e}")
                continue
            missing = _lacking(found, names)
            if missing:
                lacking.append(f"{where}: no {', '.join(missing)}")
        if lacking:
            reasons.append(
                f"slot {slot} needs {what}, which this SGLang lacks "
                f"({'; '.join(lacking)})"
            )
    return reasons


# slot -> the toolchains its kernels launch through
SLOT_TOOLCHAINS = {
    "moe_nofill": (
        ("flashinfer", "FlashInfer"),
        ("cutlass", "CuTe DSL (nvidia-cutlass-dsl)"),
    ),
    "attn_ll": (
        ("flashinfer", "FlashInfer"),
        ("cutlass", "CuTe DSL (nvidia-cutlass-dsl)"),
        ("cuda.bindings", "the CUDA Python bindings"),
    ),
}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # a dotted name whose parent package is absent
        return False


def toolchain_reasons(slots: frozenset = SLOTS) -> list[str]:
    """The requested slots' toolchains, by import spec only: the launcher imports
    nothing (as deepseek_v41.spec.toolchain_reasons). FlashInfer's own API is checked
    at build (moe_nofill.config_reasons), where importing its CuTe-DSL modules is due
    anyway."""
    needed = {m: what for s in sorted(slots) for m, what in SLOT_TOOLCHAINS.get(s, ())}
    return [f"needs {what}" for m, what in needed.items() if not _has_module(m)]


# moe_nofill launches FlashInfer's per-token NVFP4 kernel through these names;
# moe_nofill.config_reasons checks them again at build, on the imported modules
FLASHINFER_NAMES = {
    "quantization/kernels/nvfp4_quantize.py": (
        "_get_compiled_kernel_nvfp4_per_token",
        "SF_LAYOUT_LINEAR",
        "_env_flag_enabled",
        "resolve_nvfp4_4over6",
        "_per_token_cta_threads",
    ),
    "utils.py": ("device_support_pdl",),
}


def _top_level_names(path: Path) -> frozenset:
    """The names a module defines, assigns or imports at its top level (ast)."""
    names = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return frozenset(names)


def flashinfer_api_reasons(slots: frozenset = SLOTS) -> list[str]:
    """moe_nofill's FlashInfer names, read from the installed sources without
    importing them (as deepseek_v41.spec.flashinfer_ll_gemm_digest reads its GEMM's
    source)."""
    found = importlib.util.find_spec("flashinfer") if "moe_nofill" in slots else None
    if found is None or found.origin is None:
        return []  # not requested, or toolchain_reasons names the missing FlashInfer
    root = Path(found.origin).parent
    reasons = []
    for rel, names in FLASHINFER_NAMES.items():
        path = root / rel
        provided = _top_level_names(path) if path.is_file() else frozenset()
        reasons += [
            f"FlashInfer's {rel} has no {n} (slot moe_nofill)"
            for n in names
            if n not in provided
        ]
    return reasons


def _config_disable_reasons(cfg: Any, hf_config: Any) -> list[str]:
    reasons = (
        text_config_reasons(hf_config)
        + slot_reasons(SLOTS)
        + sglang_reasons(SLOTS)
        + toolchain_reasons(SLOTS)
        + flashinfer_api_reasons(SLOTS)
    )
    graphs = cfg.cuda_graph_config
    if graphs.decode.backend == "disabled" and not eager_steps_allowed():
        reasons.append(
            "decode CUDA graphs are disabled: covered steps run only in decode CUDA "
            "graphs; enable them"
        )
    if (cfg.speculative_algorithm is None) != (
        (cfg.speculative_num_draft_tokens or 1) <= 1
    ):
        reasons.append(
            f"speculative_num_draft_tokens {cfg.speculative_num_draft_tokens!r} does "
            f"not match speculative_algorithm {cfg.speculative_algorithm!r}"
        )
    return reasons


def _model_disable_reasons(model: Any, hf_config: Any) -> list[str]:
    from sgl_nvidia_mega.minimax_m31.runtime import model_disable_reasons

    return model_disable_reasons(model)


def _build(model: Any, hf_config: Any, ctx: Any):
    from sgl_nvidia_mega.minimax_m31.runtime import build

    return build(model, ctx)


SPEC = MegakernelSpec(
    name="MiniMax-M3.1",
    requires=REQUIREMENTS,
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=NUMERICS,
    max_rows=MAX_ROWS,
    build=_build,
    api_version=API_VERSION,
    model_disable_reasons=_model_disable_reasons,
    config_disable_reasons=_config_disable_reasons,
)
