# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/step3p7.py (Step3p7ForConditionalGeneration.forward) and
# python/sglang/srt/managers/mm_utils.py (general_mm_embed_routine).
"""The token embedding of a covered step.

Stock: the input preparation ``Step3p7ForConditionalGeneration.forward`` runs through
``general_mm_embed_routine``: ``prepare_mm_inputs`` with the VL model's arguments. A
verify step carries no multimodal input, so it takes the language model's
vocabulary-parallel lookup (``VocabParallelEmbedding``: the masked shard gather and
its TP all-reduce), copied into the batch's embedding buffer when the batch has one.
``Step5Model.forward`` then takes the rows as its ``input_embeds`` (``.bfloat16()`` is
a no-op on BF16 rows; kept for fidelity).
"""

from __future__ import annotations

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/managers/mm_utils.py": (
        "prepare_mm_inputs",
        "prepare_mm_inputs(input_ids)",
        "prepare_mm_inputs(forward_batch)",
        "prepare_mm_inputs(language_model)",
        "prepare_mm_inputs(kwargs)",
        "prepare_mm_inputs(multimodal_model)",
        "prepare_mm_inputs(data_embedding_funcs)",
    ),
    "srt/managers/schedule_batch.py": ("Modality", "IMAGE"),
    "srt/models/step3p7.py": ("get_image_feature", "language_model"),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/step3p7.py:Step3p7ForConditionalGeneration.forward": "92639b840ead748a",
    "srt/managers/mm_utils.py:general_mm_embed_routine": "4230344c2697664e",
}


def embed(st):
    """[rows, 4096] BF16 for ``st.input_ids``: ``general_mm_embed_routine``'s
    ``prepare_mm_inputs`` call with ``Step3p7ForConditionalGeneration.forward``'s
    arguments (no multimodal model, the image embedding function, positions in
    kwargs); the routine's language-model call is the executor's."""
    from sglang.srt.managers.mm_utils import prepare_mm_inputs
    from sglang.srt.managers.schedule_batch import Modality

    vl = st.vl
    kwargs = {"positions": st.positions}
    return prepare_mm_inputs(
        st.input_ids,
        st.fb,
        vl.language_model,
        kwargs,
        None,
        {Modality.IMAGE: vl.get_image_feature},
        None,
        {},
        None,
    )
