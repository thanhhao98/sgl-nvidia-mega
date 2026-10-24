# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``StepState`` (``st``): what one covered step carries between its step functions.

The executor creates one per forward. Fields fixed for the step are set at creation;
a package keeps values of its own in ``scratch`` under ``"<package>.<name>"`` and its
build result in ``bound["<package>.<name>"]``.
"""

from __future__ import annotations

from typing import Any

import torch


class StepState:
    __slots__ = (
        "vl",
        "lm",
        "model",
        "fb",
        "positions",
        "input_ids",
        "lm_input_ids",
        "rows",
        "slots",
        "bound",
        "binding",
        "view",
        "step5",
        "scratch",
    )

    def __init__(
        self,
        *,
        vl,
        forward_batch,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        slots,
        bound: dict,
        binding: Any,
        view: Any,
        step5: Any,
    ):
        self.vl = vl  # Step5ForConditionalGeneration (the replaced forward's owner)
        self.lm = vl.language_model  # Step5ForCausalLM: lm_head, logits_processor
        self.model = self.lm.model  # Step5Model: embed_tokens, layers, norm
        self.fb = forward_batch
        self.positions = positions  # the forward's positions [rows] int64
        self.input_ids = input_ids  # the forward's token ids [rows]
        # general_mm_embed_routine calls the language model with input_ids=None, so
        # Step5ForCausalLM.forward hands None to the logits processor
        self.lm_input_ids = None
        self.rows = int(input_ids.shape[0])
        self.slots = slots  # plan.Slots: which kernels fill this width's step
        self.bound = bound  # "<package>.<name>" -> its build() result
        self.binding = binding  # SGLang's MegakernelBinding (BuildContext.binding)
        # Step.view: the running forward's ForwardView (built at capture for a graph
        # step), and its Step-5 view: what Step5Attention.forward asks SGLang's
        # Step5AttentionBackend (None when the attention runs SGLang's module whole)
        self.view = view
        self.step5 = step5
        self.scratch: dict[str, Any] = {}
