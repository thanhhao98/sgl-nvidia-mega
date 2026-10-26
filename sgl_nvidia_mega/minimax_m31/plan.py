# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which of the slots fill each width's steps (rows), fixed at build: a slot serves a
width when its build serves it there (``Facts``), within its kernel's row bound. With
every slot off the step functions run SGLang's own modules."""

from __future__ import annotations

import msgspec

# The covered steps: up to MAX_REQUESTS requests (DSpark block 7: 8 rows a request,
# so verify steps of 8-128 rows; no-spec decode steps of 1-16 rows)
MAX_REQUESTS = 16
MAX_ROWS = 128
# slot attn_ll: FlashInfer's low-latency GEMM runner takes one 8-row tile
# (sm100/mxfp8_ll.MAX_ROWS)
ATTN_LL_MAX_ROWS = 8
# slot hop_mxfp8: the boundary kernel's MXFP8 epilogue writes one 128-row scale tile
# (SGLang's MXFP8_QUANT_MAX_TOKENS)
HOP_MXFP8_MAX_ROWS = 128
# slot hop_ffn: the FFN-input boundary kernel's quantizing forms write the same tile
HOP_FFN_MAX_ROWS = 128


class Slots(msgspec.Struct, frozen=True):
    """The kernels that fill one width's steps (``runtime``); with every field False the
    step functions run SGLang's own modules."""

    moe_nofill: bool = False
    attn_ll: bool = False
    hop_mxfp8: bool = False
    qk_store: bool = False
    hop_ffn: bool = False


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank agrees on (``runtime.build`` checks it); all
    empty: stock. Each field is the widths its slot's build serves: the build decides
    them (attn_ll per width from FlashInfer's tactics, hop_mxfp8 and hop_ffn within
    SGLang's MXFP8 quantize bound, qk_store at every width or none)."""

    moe_nofill: frozenset = frozenset()
    attn_ll: frozenset = frozenset()
    hop_mxfp8: frozenset = frozenset()
    qk_store: frozenset = frozenset()
    hop_ffn: frozenset = frozenset()


def served_widths(widths: tuple[int, ...], tokens_per_request: int) -> tuple[int, ...]:
    """The requested widths of at most MAX_REQUESTS requests."""
    return tuple(w for w in widths if w <= MAX_REQUESTS * tokens_per_request)


def slots(rows: int, facts: Facts) -> Slots:
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"{rows} rows are outside the covered widths 1-{MAX_ROWS}")
    return Slots(
        moe_nofill=rows in facts.moe_nofill,
        attn_ll=rows <= ATTN_LL_MAX_ROWS and rows in facts.attn_ll,
        hop_mxfp8=rows <= HOP_MXFP8_MAX_ROWS and rows in facts.hop_mxfp8,
        qk_store=rows in facts.qk_store,
        # its MoE inputs are moe_nofill's (runtime.NEEDS)
        hop_ffn=rows <= HOP_FFN_MAX_ROWS
        and rows in facts.hop_ffn
        and rows in facts.moe_nofill,
    )


def plan(widths: tuple[int, ...], facts: Facts) -> dict[int, Slots]:
    return {w: slots(w, facts) for w in widths}
