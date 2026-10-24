# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Step-5-Preview FP8 servers the serving tests launch: GB300 (one 4-GPU tray) or
B300 (4 of 8 GPUs), TP4, EAGLE 3 / 1 / 4 with multi-layer MTP, the deployed flags but
for two: ``--max-running-requests 8`` and ``--disable-flashinfer-autotune``.

At 8 requests every covered width (1-8 requests of 4 verify rows) has its decode graph
and no wider graph exists. That also removes what only steps past 32 rows use:
SGLang builds the MNNVL verify workspace only when the widest decode step
(``--max-running-requests`` x 4 rows) is wider than the Step-5 multicast tier's 32
rows, so these servers log no ``backend=mnnvl ... max_token_num=256`` workspace.
Every serving gate of the megakernel (parity, served token identity, profiles,
accuracy) therefore runs at the deployed ``--max-running-requests 64``; these servers
are the plugin's own pytest only. The FlashInfer tactics re-tune on every boot, so
token-identity A/Bs fix them.

A test runs only with ``SGLANG_NVIDIA_MEGAKERNEL_STEP5_E2E_MODEL_PATH`` set (the
checkpoint's local path or HF id).
"""

import os

MODEL = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_STEP5_E2E_MODEL_PATH")
NO_MODEL = (
    "set SGLANG_NVIDIA_MEGAKERNEL_STEP5_E2E_MODEL_PATH to a Step-5-Preview FP8 "
    "checkpoint"
)
ENV = {}
# the covered widths: 1-8 requests of 4 verify rows
REQUESTS = range(1, 9)
WIDTHS = {4 * n for n in REQUESTS}
ARGS = [
    "--tp",
    "4",
    "--ep-size",
    "1",
    "--attention-backend",
    "fa4",
    "--page-size",
    "64",
    "--moe-runner-backend",
    "flashinfer_trtllm",
    "--flashinfer-allreduce-fusion-backend",
    "trtllm",
    "--swa-full-tokens-ratio",
    "0.1",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--enable-multi-layer-eagle",
    "--chunked-prefill-size",
    "16384",
    "--mem-fraction-static",
    "0.9",
    # every covered width has its decode graph (batch sizes 1-8 are captured)
    "--max-running-requests",
    "8",
    # the FlashInfer tactics re-tune on every boot: identity A/Bs need them fixed
    "--disable-flashinfer-autotune",
] + os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_STEP5_E2E_EXTRA_ARGS", "").split()
