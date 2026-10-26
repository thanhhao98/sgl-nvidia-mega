# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# test/registered/unit/kernel_plugins/nvidia/_fakes.py (server_args).
"""The M3.1 checkpoint's text config and the M3.1 recipe's server arguments, shared
by the minimax_m31 CPU tests (a helper module, not collected)."""

from types import SimpleNamespace

from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.srt.server_args import ServerArgs

from sgl_nvidia_mega import minimax_m31

VL = "MiniMaxM3SparseForConditionalGeneration"


def text_config(**overrides):
    layers = minimax_m31.TEXT_CONFIG["num_hidden_layers"]
    sparse = dict(
        minimax_m31.SPARSE_ATTENTION,
        sparse_attention_freq=[1] * layers,
        sparse_disable_index_value=[1] * layers,
    )
    fields = dict(
        minimax_m31.TEXT_CONFIG,
        rope_theta=10_000_000,
        rope_scaling=None,
        sparse_attention_config=sparse,
        moe_layer_freq=[0] * minimax_m31.DENSE_LAYERS
        + [1] * (layers - minimax_m31.DENSE_LAYERS),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def hf_config(**overrides):
    return SimpleNamespace(architectures=[VL], text_config=text_config(**overrides))


def server_args(**kw):
    """Arguments as the resolution step sees them, with the graph config resolved."""
    defaults = {
        "model_path": "dummy",
        "tp_size": 4,
        "kv_cache_dtype": "fp8_e4m3",
        "speculative_algorithm": "DSPARK",
        "speculative_num_draft_tokens": 6,
        "cuda_graph_config": default_cuda_graph_config(),
    }
    return ServerArgs(**{**defaults, **kw})


def m31_server_args(**kw):
    defaults = dict(
        quantization="minimax_m3_nvfp4",
        page_size=128,
        moe_runner_backend="flashinfer_trtllm_routed",
        moe_a2a_backend="none",
        speculative_num_draft_tokens=8,
    )
    return server_args(**{**defaults, **kw})
