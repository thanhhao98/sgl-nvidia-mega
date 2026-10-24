# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/step3p7.py (Step3p7ForConditionalGeneration.forward),
# python/sglang/srt/managers/mm_utils.py (general_mm_embed_routine) and
# python/sglang/srt/models/step5.py (Step5ForCausalLM.forward, Step5Model.forward,
# Step5DecoderLayer.forward).
"""The executor: one covered target-verify step of Step-5.

``Step5Verify.plan`` decides from host state alone, identical on every rank, whether a
step is covered: a target-verify step of a built width (4-32 rows), while SGLang
captures its decode CUDA graphs (their warm-up runs included) or eagerly under the
test-only parity switch. Any other step, e.g. an eager one past the largest graph,
runs SGLang's forward.

``Step5Verify.forward`` takes and returns what ``Step5ForConditionalGeneration.forward``
does (a ``LogitsProcessorOutput``), from the plugin's step functions over SGLang's
loaded modules, never through a model-level or decoder-layer ``forward``:

1. the embedding: ``prepare_mm_inputs`` as ``general_mm_embed_routine`` calls it;
2. ``Step5Model.forward``: ``residual_batch.start``; per layer
   ``x = boundary.attn_input``, ``part = attention.step``,
   ``h = boundary.attn_output``, ``x = boundary.ffn_input``, then inside
   ``boundary.ffn_exit``: ``out = ffn.step(defer=ffn_exit.defer_moe_finalize)`` and
   ``h = ffn_exit.finish(out)``; the tail (``boundary.close_stream``);
3. ``Step5ForCausalLM.forward``'s logits processor call (``logits``).
"""

from __future__ import annotations

import logging
from functools import partial

import msgspec
import torch

from sgl_nvidia_mega.step5 import attention, boundary, ffn, logits, runtime
from sgl_nvidia_mega.step5.plan import TOKENS_PER_REQUEST
from sgl_nvidia_mega.step5.runtime import embedding
from sgl_nvidia_mega.step5.runtime.state import StepState

try:
    from sglang.kernels.plugin_kernels.megakernel.spec import Step
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import Step

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # Step, and the BuiltMegakernel members SGLang reads
    "kernels/plugin_kernels/megakernel/spec.py|srt/megakernels/spec.py": (
        "Step",
        "mode",
        "rows",
        "view",
        "widths",
        "draft_sampler",
        "summary",
        "plan",
        "forward",
    ),
    # Step.view's Step-5 view, which the attention step code reads
    "srt/kernel_plugins/megakernel/binding.py|srt/megakernels/binding.py": ("step5",),
    # the forward this executor stands in for, and what plan reads
    "srt/models/step3p7.py": ("forward(get_embedding)", "language_model"),
    "srt/models/step5.py": ("model", "layers", "forward(input_embeds)"),
    "srt/model_executor/forward_batch_info.py": ("forward_mode", "is_target_verify"),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/*.py": (
        "is_in_breakable_cuda_graph",
    ),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang); the layer loop
# and the decoder layer are boundary.SGLANG_MIRRORS' too, the logits call
# logits.SGLANG_MIRRORS'
SGLANG_MIRRORS = {
    "srt/models/step3p7.py:Step3p7ForConditionalGeneration.forward": "92639b840ead748a",
    "srt/models/step5.py:Step5ForCausalLM.forward": "77d904a49a0dca65",
    "srt/models/step5.py:Step5Model.forward": "b7846fb61d679b85",
}


class Step5Verify:
    """One rank's covered steps: the model, each width's slots, each package's build
    result."""

    # the target offers no draft sampler (BuiltMegakernel)
    draft_sampler = None

    def __init__(self, vl, *, slots, bound, widths, binding, summary, parity):
        self.vl, self.slots, self.bound = vl, slots, bound
        self.widths = tuple(sorted(widths))
        self.binding = binding  # SGLang's MegakernelBinding (BuildContext.binding)
        self._summary = summary
        self.parity = parity  # runtime.parity.Parity under the test-only switch
        self._engaged: set = set()

    def summary(self) -> str:
        """What was built, for SGLang's startup line."""
        return self._summary

    def plan(self, input_ids, forward_batch, get_embedding=False, **kwargs):
        """The step if this forward is covered, else None (SGLang's forward runs). An
        argument this executor does not know also leaves the step to SGLang."""
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            is_in_breakable_cuda_graph,
        )

        from sgl_nvidia_mega.step5.spec import serves_this_step

        if kwargs or get_embedding:
            return None
        if not forward_batch.forward_mode.is_target_verify():
            return None
        # Step5Attention.forward takes its breakable-graph arm there
        if is_in_breakable_cuda_graph() or not serves_this_step():
            return None
        rows = int(input_ids.shape[0])
        return Step(mode="target_verify", rows=rows) if rows in self.widths else None

    def forward(self, step, original, input_ids, positions, forward_batch, *args, **kw):
        """The covered step (``original``: the model's own forward, bound, which the
        parity check runs first)."""
        from sglang.srt.model_executor.runner import get_is_capture_mode

        self._engage(step)
        run = partial(self._forward, step, input_ids, positions, forward_batch)
        # eager steps only: a graph's warm-ups run eagerly too, on placeholder inputs
        if self.parity is not None and not get_is_capture_mode():
            stock = partial(original, input_ids, positions, forward_batch, *args, **kw)
            return self.parity.run(step, stock, run)
        return run()

    @torch.no_grad()
    def _forward(self, step, input_ids, positions, forward_batch):
        """``Step5ForConditionalGeneration.forward`` for a covered step."""
        st = StepState(
            vl=self.vl,
            forward_batch=forward_batch,
            positions=positions,
            input_ids=input_ids,
            slots=self.slots[step.rows],
            bound=self.bound,
            binding=self.binding,
            view=step.view,
            # the Step-5 view the attention step code reads; none when SGLang's
            # attention module runs whole (runtime.ATTN_ENV, test only)
            step5=None if runtime.ATTN_MODE == "module" else step.view.step5,
        )
        # general_mm_embed_routine: the input embeddings, then the language model
        # (Step5ForCausalLM.forward with input_ids=None and these embeddings)
        hidden = embedding.embed(st)
        normalized, before_norm = self._model(st, hidden)
        return logits.logits(st, normalized, before_norm)

    def _model(self, st, input_embeds):
        """``Step5Model.forward`` with ``input_embeds``: (normed rows, FP32 pre-norm
        sum)."""
        hidden = input_embeds.bfloat16()
        boundary.open_stream(st)
        for layer in st.model.layers:
            hidden = self._layer(layer, hidden, st)
        return boundary.close_stream(st, hidden)

    @staticmethod
    def _layer(layer, hidden, st):
        """``Step5DecoderLayer.forward``, TP arm (attention DP is refused at load)."""
        x = boundary.attn_input(layer, hidden, st)
        part = attention.step(layer, x, st)
        hidden = boundary.attn_output(layer, part, st)
        x = boundary.ffn_input(layer, hidden, st)
        with boundary.ffn_exit(layer, st) as ffn_exit:
            out = ffn.step(layer, x, st, defer=ffn_exit.defer_moe_finalize)
        return ffn_exit.finish(out)

    def _engage(self, step) -> None:
        """Log, once per width and path, that the megakernel serves it: in a decode
        graph's capture (its warm-ups included) or in an eager step (parity)."""
        from sglang.srt.model_executor.runner import get_is_capture_mode

        path = "capture" if get_is_capture_mode() else "eager"
        key = (step.rows, path)
        if key in self._engaged:
            return
        self._engaged.add(key)
        on = [n for n, v in msgspec.structs.asdict(self.slots[step.rows]).items() if v]
        logger.info(
            "NVIDIA megakernel: Step-5 serves %s rows=%d (batch %d) in %s; slots %s",
            step.mode,
            step.rows,
            step.rows // TOKENS_PER_REQUEST,
            "a decode graph capture" if path == "capture" else "an eager step",
            ", ".join(sorted(on)) or "none (stock)",
        )
