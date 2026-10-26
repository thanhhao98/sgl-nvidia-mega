# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MiniMax M3.1-Flash servers the serving tests launch: GB300, TP4, DSpark block 7
(verify steps of 8 rows a request). The checkpoint is ``MiniMaxAI/MiniMax-M3.1-Flash``
or the one ``SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH`` names; SGLang loads the DSpark
draft from its ``dspark/`` folder. The megakernel switch is set per launch.

SGLang resolves the recipe the spec requires from the checkpoint: ``minimax_m3_nvfp4``
(MXFP8 projections, NVFP4 experts), the ``fp8_e4m3`` KV cache (packed NVFP4 KV and
index-K) and pages of 128. The arguments leave them to it, as the validated runs did:
an explicit ``--quantization`` would also become the draft's quantization. A test that
compares tokens across boots (``test_minimax_m31_switch``) adds
``--disable-flashinfer-autotune`` to each server it launches: FlashInfer re-tunes the
MXFP8 GEMMs on every boot. ``SGLANG_NVIDIA_MEGAKERNEL_E2E_EXTRA_ARGS`` appends
arguments to every server."""

import os

MODEL = os.environ.get(
    "SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH", "MiniMaxAI/MiniMax-M3.1-Flash"
)
TP = 4
# DSpark block 7: a request's verify step scores the block plus one token, 8 rows
BLOCK = 7
ROWS_PER_REQUEST = BLOCK + 1
# SGLang refuses DSpark on the M3.1 NVFP4 recipe without the static ragged verify
ENV = {"SGLANG_RAGGED_VERIFY_MODE": "static"}
ARGS = [
    "--trust-remote-code",
    "--tp-size",
    str(TP),
    # the routed experts on FlashInfer's trtllm-gen routed MoE at EP 1, as the spec
    # requires (SGLang's M3.1 default is MegaMoE)
    "--moe-runner-backend",
    "flashinfer_trtllm_routed",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-dspark-block-size",
    str(BLOCK),
    # as validated on GB300: decode graphs up to 128 requests, prefill without graphs
    "--chunked-prefill-size",
    "16384",
    "--max-running-requests",
    "128",
    "--cuda-graph-max-bs-decode",
    "128",
    "--cuda-graph-backend-prefill",
    "disabled",
    "--mem-fraction-static",
    "0.8",
    # MiniMax's chat parsers, named: auto leaves them off when detection fails
    "--reasoning-parser",
    "minimax-m3",
    "--tool-call-parser",
    "minimax-m3",
] + os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_EXTRA_ARGS", "").split()


def warm_remote_code():
    """Import the checkpoint's remote code before a launch: the TP ranks import it at
    the same time, and on a cold module cache one of them can import a half-written
    module."""
    from transformers import AutoConfig, AutoProcessor

    AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
