# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Test only: the covered step against SGLang's own forward on the same eager step
(``SGLANG_NVIDIA_MEGAKERNEL_PARITY``; decode CUDA graphs off).

``1`` runs SGLang's forward on the step, then the megakernel's, and compares the
logits and the hidden states EAGLE's draft extend consumes (the FP32 pre-norm sum),
bit for bit; the megakernel's output continues the request, so every step is compared
on the candidate's own trajectory. ``control`` runs SGLang's forward twice: the
run-to-run floor, and the proof that running a step twice is safe.

Running a verify step twice writes its KV twice. Every write lands in the same slot
with the same value, so a second run sees what the first saw: the SWA K/V rows at the
backend's write locations and the DSA K/V rows at ``out_cache_loc`` (both forwards
write the same bits); the CSA ring append and R8 compression (csa_ring.py
``_compress`` / ``_write_row``) read only history positions [base // 8 * 8, base),
whose ring cells (pos % 16) no write of this step [base, base + 4) can reach while
the width is at most ring - 7 = 9 (the backend enforces it), keep the tags of the
cells they do not write (or reset them all at base % 8 == 0, when there is no
history), and publish a compressed key that depends on that history and this step's
keys alone; per-forward scratch (selection, scoring, LSE / partials) is written before
it is read in every layer. No state is saved and restored. The ``control`` arm is the
measurement of this argument: SGLang's own forward twice must be bitwise at every
width.

SGLang's logits land in the batch's ``next_token_logits_buffer``, which the second
forward overwrites, so the first output is cloned before the second runs.

The line keeps the format the harness parses (``NVIDIA megakernel parity[ control]
<mode> rows=<n> logits_max_abs=<v> (rel <r>, <p>% differ) ... hidden_max_abs=<v>
...``) and adds the differing-element counts, the bitwise judge (``max_abs`` 0 cannot
see -0 / +0 or NaN payloads); an autotune dummy run returns no logits (``n/a``).
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/logits_processor.py": ("next_token_logits", "hidden_states"),
}

PARITY_ENV = "SGLANG_NVIDIA_MEGAKERNEL_PARITY"
MODE = os.environ.get(PARITY_ENV, "0")
ENABLED = MODE in ("1", "control")

# element size -> the integer type its bits are compared as
_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


class Parity:
    def __init__(self):
        self.label = " control" if MODE == "control" else ""

    def run(self, step, original, megakernel):
        """``original()`` then ``megakernel()`` (``original()`` again for control) on
        the same eager step; returns the second's output."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the parity check runs on eager steps only")
        want = original()
        want = (_clone(want.next_token_logits), _clone(want.hidden_states))
        got = original() if MODE == "control" else megakernel()
        pairs = list(zip(want, (got.next_token_logits, got.hidden_states)))
        logger.info(
            "NVIDIA megakernel parity%s %s rows=%d logits_max_abs=%s "
            "hidden_max_abs=%s logits_bits_differ=%s hidden_bits_differ=%s",
            self.label,
            step.mode,
            step.rows,
            *(_max_abs(a, b) for a, b in pairs),
            *(_bits_differ(a, b) for a, b in pairs),
        )
        return got


def _clone(t):
    return None if t is None else t.clone()


def _max_abs(a, b) -> str:
    """max |a - b|, that over max |a|, and the fraction of elements that differ (the
    DeepSeek-V4.1 package's format, which the harness's summary parses)."""
    if a is None or b is None:
        return "n/a" if a is None and b is None else "missing"
    if a.shape != b.shape or a.dtype != b.dtype:
        return f"shape {tuple(a.shape)}:{a.dtype} vs {tuple(b.shape)}:{b.dtype}"
    d = (a.float() - b.float()).abs()
    peak = a.float().abs().max().clamp_min(1e-30)
    return (
        f"{d.max().item():.3g} (rel {(d.max() / peak).item():.2e}, "
        f"{(d > 0).float().mean().item():.2%} differ)"
    )


def _bits_differ(a, b) -> str:
    """The number of elements whose bits differ, or n/a without the output."""
    if a is None or b is None:
        return "n/a" if a is None and b is None else "missing"
    if a.shape != b.shape or a.dtype != b.dtype:
        return "shape"
    bits = _BITS[a.element_size()]
    return str(int((a.contiguous().view(bits) != b.contiguous().view(bits)).sum()))
