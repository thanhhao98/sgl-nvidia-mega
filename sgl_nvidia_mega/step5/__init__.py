# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""StepFun Step-5 (Step-5-Preview, FP8) megakernel for the EAGLE target verify step on
GB300 and B300 (sm_103) at TP4: its spec and the setups it serves (``spec``), the rows
each covered step has (``plan``), the executor of a covered step, its parity check and
its build (``runtime``), and the step code per stage (``boundary``, ``attention``,
``ffn``, ``logits``). The three-layer MTP draft (Step5MTP) runs SGLang's forward.

No SGLang import here: the launcher imports this package with the spec.
"""

# The checkpoint the stock step is checked against (stepfun-ai/Step-5-Preview at
# 364b3b03fed2477aa0953fe855a2c4c8c48ee3cc); its config.json nests the text model
# under text_config (model_type step5) inside the VL config (model_type step5_vl).
ARCHITECTURE = "Step5ForConditionalGeneration"
VL_MODEL_TYPE = "step5_vl"

# text_config fields the step code and its slots are written for, as SGLang's
# Step5Config normalizes them (srt/configs/step5.py); spec.text_config_reasons
# refuses any other value
TEXT_CONFIG = dict(
    model_type="step5",
    hidden_size=4096,
    num_hidden_layers=92,
    num_nextn_predict_layers=3,
    num_attention_heads=64,
    num_attention_groups=4,
    head_dim=192,
    intermediate_size=13824,
    moe_num_experts=352,
    moe_top_k=8,
    moe_intermediate_size=1536,
    share_expert_dim=1536,
    moe_router_scaling_factor=3.0,
    need_fp32_gate=True,
    use_moe_router_bias=True,
    use_head_wise_attn_gate=True,
    sliding_window=512,
    vocab_size=128896,
    rms_norm_eps=1e-05,
    zero_centered=True,
)
# The layer pattern: MoE in layers 3-90, DSA (full attention + CSA indexer) every 4th
# layer from 3, SWA elsewhere; the MTP layers 92-94 belong to the draft
MOE_LAYERS = tuple(range(3, 91))
DSA_LAYERS = tuple(range(3, 92, 4))
# parse_step5_attention_profile(text_config), field by field
ATTENTION_PROFILE = dict(
    hidden_size=4096,
    head_dim=192,
    num_heads=64,
    num_kv_heads=4,
    index_head_dim=256,
    index_num_heads=16,
    index_num_kv_heads=1,
    provider_groups=4,
    region_size=8,
    index_rope_dim=32,
    sliding_window=512,
)
# The routed experts of layers 3-87 are FP8 E4M3 with 128 x 128 block scales; the
# checkpoint keeps layers 88-90's experts (and everything else) BF16
QUANTIZATION = dict(quant_method="fp8", fmt="e4m3", weight_block_size=[128, 128])
FP8_EXPERT_LAYERS = tuple(range(3, 88))
BF16_EXPERT_LAYERS = (88, 89, 90)
