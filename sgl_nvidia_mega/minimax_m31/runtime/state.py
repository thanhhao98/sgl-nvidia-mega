# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``StepState`` (``st``): what one covered step carries between its step functions.

The executor creates one per forward. Fields fixed for the step are set at creation;
``layer_id`` is rewritten for each layer. A step module keeps values of its own in
``scratch`` under ``"<module>.<name>"`` and its build result in ``bound["<module>"]``.
"""

from __future__ import annotations

from typing import Any


class StepState:
    def __init__(
        self,
        *,
        lm: Any,
        forward_batch: Any,
        positions: Any,
        input_ids: Any,
        step: Any,
        slots: Any,
        bound: dict,
        binding: Any = None,
    ):
        self.lm = lm  # MiniMaxM3SparseForConditionalGeneration
        self.model = lm.model  # MiniMaxM3Model: embedding, layers, final norm
        self.fb = forward_batch
        self.positions = positions
        self.input_ids = input_ids
        self.mode: str = step.mode  # "decode" or "target_verify"
        self.rows: int = step.rows
        # SGLang's ForwardView of this forward (Step.view), and the build's binding
        self.view = step.view
        self.binding = binding
        self.slots = slots  # plan.Slots: the kernels that fill this width's steps
        self.bound = bound  # module -> its build result
        self.layer_id = -1
        self.scratch: dict[str, Any] = {}
