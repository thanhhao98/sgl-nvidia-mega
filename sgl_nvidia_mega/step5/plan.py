# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which steps are covered and which kernel fills each slot of a covered step, per
width (rows), fixed at build.

A covered step is the target verify of 1 to MAX_REQUESTS requests of
TOKENS_PER_REQUEST rows (EAGLE 3 steps, top-1, 4 draft tokens): 4 to 32 rows, SGLang's
decode graphs of batch sizes 1-8. Up to 32 rows SGLang's own Step-5 path is one tier
when its Step-5 multicast communicator exists: both boundaries of every layer run the
Step-5 multicast all-reduce kernel and an FP8 MoE layer hands its finalize to the next
layer's input kernel (srt/layers/step5_allreduce.py: _get_comm, max_fused_rows,
can_defer_finalize). Past 32 rows the boundary is FlashInfer's MNNVL one-shot
workspace (patched FlashInfer only) and the MoE finalizes inside the FFN; those steps
keep SGLang's forward until a slot needs them.

A slot runs only where it is requested (SGLANG_NVIDIA_MEGAKERNEL_SLOTS), built and
inside its row bound (runtime.SLOT_TABLE[...].max_rows); every other op is SGLang's.
"""

from __future__ import annotations

import msgspec

TOKENS_PER_REQUEST = 4  # verify rows a request: --speculative-num-draft-tokens 4
MAX_REQUESTS = 8
MAX_ROWS = MAX_REQUESTS * TOKENS_PER_REQUEST  # 32


class Slots(msgspec.Struct, frozen=True):
    """The kernels that fill one width's step: one bool per slot (none yet, so every
    step function runs its stock code)."""


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on (none yet)."""


def covered(widths: tuple[int, ...], tokens_per_request: int) -> tuple[int, ...]:
    """The widths SGLang offers (BuildContext.widths) this package covers."""
    return tuple(w for w in widths if w % tokens_per_request == 0 and 0 < w <= MAX_ROWS)


def slots(rows: int, facts: Facts) -> Slots:
    """The slots that fill a step of ``rows`` rows."""
    if not 0 < rows <= MAX_ROWS:
        raise ValueError(f"{rows} rows are outside the covered widths 4-{MAX_ROWS}")
    return Slots()


def plan(widths: tuple[int, ...], facts: Facts) -> dict[int, Slots]:
    """Each covered width's slots."""
    return {w: slots(w, facts) for w in widths}
