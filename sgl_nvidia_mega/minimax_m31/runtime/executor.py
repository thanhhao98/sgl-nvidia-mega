# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/minimax_m3_vl.py (MiniMaxM3SparseForConditionalGeneration.forward)
# and python/sglang/srt/models/minimax_m3.py (MiniMaxM3Model.forward and its layer loop,
# MiniMaxM3DecoderLayer.forward).
"""The executor: one covered step of the M3.1 target.

``MegakernelDecode.plan`` decides from host state alone, alike on every rank, whether
a forward is covered: a DSpark verify step (or, without speculation, a decode step)
of a built width, while SGLang captures its decode CUDA graphs.
``MegakernelDecode.forward`` takes and returns what
``MiniMaxM3SparseForConditionalGeneration.forward`` does. It runs the step from the
step functions over SGLang's loaded modules, never through the model's or a decoder
layer's ``forward``:

1. ``boundary.start``: the embeddings and the residual stream;
2. per layer, under the expert-distribution recorder:
   ``x = boundary.attn_input(layer, x, st, captured)``, ``x = attention.step(layer, x, st)``,
   ``x = boundary.ffn_input(layer, x, st)``, then inside ``layer.ffn_boundary.exit``
   ``x = moe.step(layer, x, st)`` and ``x = ffn_exit.finish(x)``;
3. ``boundary.final`` (the final norm and the last DSpark capture), ``logits.logits``.

``SGLANG_NVIDIA_MEGAKERNEL_PARITY`` (test only, eager covered steps) runs SGLang's own
forward on the same step first and logs the largest difference of the logits and the
DSpark hidden states; ``control`` runs SGLang's forward twice (the run-to-run floor).
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Any, Optional

import torch

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import Step
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import Step

from sgl_nvidia_mega.minimax_m31 import attention, boundary, logits, moe
from sgl_nvidia_mega.minimax_m31.runtime.state import StepState
from sgl_nvidia_mega.minimax_m31.spec import serves_this_step

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # the BuiltMegakernel members SGLang reads and the Step it hands forward
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
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
    "srt/model_executor/forward_batch_info.py": (
        "forward_mode",
        "is_target_verify",
        "is_decode",
    ),
    "srt/eplb/expert_distribution.py": (
        "get_global_expert_distribution_recorder",
        "with_current_layer",
    ),
    "srt/layers/aux_hidden_states.py": ("AuxHiddenStateList",),
    "srt/model_executor/cuda_graph_config.py": (
        "Backend",
        "Phase",
        "check_cuda_graph_backend",
        "PREFILL",
        "TC_PIECEWISE",
    ),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/*.py": (
        "is_in_breakable_cuda_graph",
    ),
    "srt/models/minimax_m3.py": (
        "start_layer",
        "end_layer",
        "layers",
        "ffn_boundary",
        "is_layer_sparse",
    ),
    # MiniMaxM3SparseForConditionalGeneration.forward, which the executor stands in for
    # (plan() mirrors its signature); the wrapper attributes the step and
    # logits/__init__.py read; and the capture flag the wrapper sets
    "srt/models/minimax_m3_vl.py": (
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(get_embedding)",
        "forward(pp_proxy_tensors)",
        "model",
        "capture_aux_hidden_states",
        "logits_processor",
        "lm_head",
        "_is_layer_to_capture",
    ),
    "srt/layers/layer_boundary/*.py": ("exit", "finish"),
    "srt/layers/logits_processor.py": ("next_token_logits", "hidden_states"),
}

logger = logging.getLogger(__name__)

PARITY_ENV = "SGLANG_NVIDIA_MEGAKERNEL_PARITY"
PARITY_MODE = os.environ.get(PARITY_ENV, "0")
PARITY = PARITY_MODE in ("1", "control")


class MegakernelDecode:
    """One rank's covered steps: the model, each width's slots and each step module's
    build result."""

    # the target offers no draft sampler (BuiltMegakernel)
    draft_sampler = None

    def __init__(
        self,
        lm,
        *,
        widths: tuple[int, ...],
        tokens_per_request: int,
        plan: dict,
        bound: dict,
        binding: Any = None,
        summary: str = "",
    ):
        self.lm = lm
        self.widths = tuple(sorted(widths))
        self.tokens_per_request = tokens_per_request
        self.slot_plan = plan
        self.bound = bound
        # SGLang's MegakernelBinding (BuildContext.binding)
        self.binding = binding
        self._summary = summary

    def summary(self) -> str:
        """What was built, for SGLang's startup line."""
        return self._summary

    def plan(
        self,
        input_ids,
        forward_batch,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
        **kwargs,
    ) -> Optional[Step]:
        """The step if this forward is covered, else None (SGLang's forward runs)."""
        if get_embedding or pp_proxy_tensors is not None or kwargs:
            return None
        mode = forward_batch.forward_mode
        # a DSpark server's target runs verify steps, a server without speculation
        # decode steps
        if self.tokens_per_request > 1:
            if not mode.is_target_verify():
                return None
            name = "target_verify"
        else:
            if not mode.is_decode():
                return None
            name = "decode"
        rows = int(input_ids.shape[0])
        if rows not in self.widths or not serves_this_step():
            return None
        return Step(mode=name, rows=rows)

    def forward(
        self, step: Step, original, input_ids, positions, forward_batch, *args, **kwargs
    ):
        """The covered step (``original``: the model's own forward, bound, for the
        parity check)."""
        if PARITY and not torch.cuda.is_current_stream_capturing():
            return self._with_parity(
                step, original, input_ids, positions, forward_batch, args, kwargs
            )
        return self._forward(step, input_ids, positions, forward_batch)

    def _forward(self, step: Step, input_ids, positions, forward_batch):
        """``MiniMaxM3SparseForConditionalGeneration.forward`` for a covered step."""
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            Phase,
            check_cuda_graph_backend,
        )
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            is_in_breakable_cuda_graph,
        )

        st = StepState(
            lm=self.lm,
            forward_batch=forward_batch,
            positions=positions,
            input_ids=input_ids,
            step=step,
            slots=self.slot_plan[step.rows],
            bound=self.bound,
            binding=self.binding,
        )
        m = st.model
        x = boundary.start(st)
        aux = AuxHiddenStateList()
        # torch dynamo does not support a graph break in a context manager
        quiet = (
            check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
            or is_in_breakable_cuda_graph()
        )
        for i in range(m.start_layer, m.end_layer):
            layer = m.layers[i]
            st.layer_id = i
            captured = aux if getattr(layer, "_is_layer_to_capture", False) else None
            ctx = (
                nullcontext()
                if quiet
                else get_global_expert_distribution_recorder().with_current_layer(i)
            )
            with ctx:
                x = self._layer(st, layer, x, captured)
        x = boundary.final(st, x, aux)
        return logits.logits(st, x, aux)

    @staticmethod
    def _layer(st: StepState, layer, x, captured):
        """``MiniMaxM3DecoderLayer.forward`` for a covered step."""
        # SGLang skips a sublayer of a rank with no rows; a covered step has st.rows
        # rows on every rank and sublayer (no token scatter in this configuration), and
        # with slot hop_mxfp8 the attention input is an MXFP8 pair, with slot hop_ffn
        # the FFN input an hop_ffn.FfnInput, not a tensor.
        x = boundary.attn_input(layer, x, st, captured)
        if st.rows != 0:
            x = attention.step(layer, x, st)
        x = boundary.ffn_input(layer, x, st)
        with layer.ffn_boundary.exit(st.fb) as ffn_exit:
            if layer.is_layer_sparse or st.rows != 0:
                x = moe.step(layer, x, st)
        return ffn_exit.finish(x)

    def _with_parity(
        self, step, original, input_ids, positions, forward_batch, args, kwargs
    ):
        """SGLang's forward, then the executor's, on the same step; logs the largest
        difference of the logits and the DSpark hidden states."""
        want = original(input_ids, positions, forward_batch, *args, **kwargs)
        want = (_clone(want.next_token_logits), _clone(want.hidden_states))
        if PARITY_MODE == "control":
            got = original(input_ids, positions, forward_batch, *args, **kwargs)
        else:
            got = self._forward(step, input_ids, positions, forward_batch)
        diffs = [
            _max_abs(a, b)
            for a, b in zip(want, (got.next_token_logits, got.hidden_states))
        ]
        logger.info(
            "NVIDIA megakernel parity%s %s rows=%d logits_max_abs=%s hidden_max_abs=%s",
            " control" if PARITY_MODE == "control" else "",
            step.mode,
            step.rows,
            *diffs,
        )
        return got


def _clone(t):
    return None if t is None else t.clone()


def _max_abs(a, b) -> str:
    """max |a - b|, that over max |a|, and the fraction of elements that differ."""
    if a is None or b is None:
        return "n/a" if a is None and b is None else "missing"
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    d = (a.float() - b.float()).abs()
    peak = a.float().abs().max().clamp_min(1e-30)
    return (
        f"{d.max().item():.3g} (rel {(d.max() / peak).item():.2e}, "
        f"{(d > 0).float().mean().item():.2%} differ)"
    )
