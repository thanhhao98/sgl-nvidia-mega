# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax M3.1 / M3.1-Flash decode megakernel.

Covers the DSpark verify steps (and the no-spec decode steps) of 1-16 requests at TP4
on Blackwell, with the M3.1 NVFP4 recipe: MXFP8 projections, NVFP4 routed experts on
FlashInfer's trtllm-gen routed MoE at EP 1, MSA-v1 sparse attention over the packed
NVFP4 KV pool, and the push-plane all-reduce + residual + norm boundaries.

Modules: the spec and the setups it serves (``spec``), the per-width slot plan
(``plan``), the requested kernel slots and the build (``runtime``), the executor of a
covered step (``runtime.executor``) over its per-step state (``runtime.state``), the
TP ranks' build-time agreement (``runtime.agree``), the step functions (``boundary``,
``attention``, ``moe``, ``logits``) and the package's own kernels (``sm100``). With
every slot off the executor runs SGLang's own modules in SGLang's order, bit for bit.
"""

ARCHITECTURES = ("MiniMaxM3SparseForConditionalGeneration",)

# The text model the step code is written for (M3.1-Flash, config.text_config).
TEXT_CONFIG = dict(
    hidden_size=6144,
    num_hidden_layers=60,
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=128,
    rotary_dim=64,
    rms_norm_eps=1e-6,
    use_gemma_norm=True,
    use_qk_norm=True,
    attention_output_gate=False,
    num_local_experts=128,
    num_experts_per_tok=4,
    n_shared_experts=1,
    intermediate_size=3072,
    shared_intermediate_size=3072,
    dense_intermediate_size=12288,
    scoring_func="sigmoid",
    use_routing_bias=True,
    routed_scaling_factor=2.0,
    hidden_act="swigluoai",
    swiglu_alpha=1.702,
    swiglu_limit=7.0,
    vocab_size=200064,
)
ROPE_THETA = 1e7
# MSA (config.text_config.sparse_attention_config); every layer is sparse and has no
# index value head
SPARSE_ATTENTION = dict(
    use_sparse_attention=True,
    sparse_index_dim=128,
    sparse_num_index_heads=4,
    sparse_topk_blocks=16,
    sparse_block_size=128,
)
# moe_layer_freq: layers 0-2 run the dense MLP, 3-59 the MoE
DENSE_LAYERS = 3
