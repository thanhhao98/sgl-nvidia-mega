# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/minimax_m3_vl.py (the tail of MiniMaxM3SparseForConditionalGeneration.forward).
"""The LM head and the logits of a covered step, with the DSpark captures.

Stock: SGLang's logits processor.
"""

from __future__ import annotations

from typing import Any


def logits(st, hidden, aux) -> Any:
    """SGLang's ``LogitsProcessorOutput`` for the step."""
    lm = st.lm
    # MiniMaxM3Model.forward returns the captures only when it took some
    aux_hidden = aux if lm.capture_aux_hidden_states and len(aux) != 0 else None
    return lm.logits_processor(st.input_ids, hidden, lm.lm_head, st.fb, aux_hidden)
