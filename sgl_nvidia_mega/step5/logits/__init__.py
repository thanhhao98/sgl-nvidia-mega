# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/step5.py (Step5ForCausalLM.forward).
"""The logits of a covered step.

Stock: ``Step5ForCausalLM.forward``'s logits processor call on the normed rows, with
the FP32 pre-norm sum as ``hidden_states_before_norm`` and the language model's
``input_ids`` (None: the VL routine calls the language model without ids). The
processor returns ``LogitsProcessorOutput(next_token_logits = FP32 [rows, 128896],
the graph's next_token_logits_buffer when shapes match; hidden_states = the FP32
pre-norm sum [rows, 4096] when the batch captures hidden states, which multi-layer
EAGLE's draft extend reads; mm_input_embeds)``.

Seam: ``logits`` (the LM head GEMM, its split-K reduce, the multimem all-gather and
the FP32 copy into the logits buffer).
"""

from __future__ import annotations

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/step5.py": ("logits_processor", "lm_head"),
    "srt/layers/logits_processor.py": (
        "forward(hidden_states_before_norm)",
        "next_token_logits",
        "hidden_states",
    ),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/step5.py:Step5ForCausalLM.forward": "77d904a49a0dca65",
}


def logits(st, normalized, before_norm):
    """SGLang's ``LogitsProcessorOutput`` for the step."""
    lm = st.lm
    return lm.logits_processor(
        st.lm_input_ids,
        normalized,
        lm.lm_head,
        st.fb,
        hidden_states_before_norm=before_norm,
    )
