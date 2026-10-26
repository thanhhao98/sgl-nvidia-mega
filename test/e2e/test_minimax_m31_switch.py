# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The switch on and off on the served path: GB300 TP4, MiniMax M3.1-Flash with DSpark
(8 rows a request), decode CUDA graphs on.

A server runs the megakernel only inside its decode CUDA graphs; the parity test's
eager covered steps need a test-only switch. ``TestServedDecodesAreSGLangs`` boots the
slots the package ships without that switch, then SGLang's own forward (no
``--enable-nvidia-megakernel``), and sends both the same greedy chat requests: each of
16 prompts alone, then 2, 4, 8 and 16 of them in one request, so the verify steps reach
8, 16, 32, 64 and 128 rows. The shipped slots are bitwise, so every prompt must decode
the same tokens with the same log-probabilities on both servers.

Both servers run with ``--disable-flashinfer-autotune``: FlashInfer re-tunes the MXFP8
GEMMs on every boot, and two tactics of one GEMM can differ in the last bit. The
prompts go as token ids, which SGLang hands to the scheduler together, and each
request starts from an empty radix cache, so both servers run the same batches.

The checkpoint is the one ``minimax_m31_server`` names; the test skips on fewer than 4
GPUs.
"""

import os
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

import pytest

pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

import requests  # noqa: E402
import torch  # noqa: E402
from minimax_m31_server import (  # noqa: E402
    ARGS,
    ENV,
    MODEL,
    TP,
    warm_remote_code,
)
from sglang.test.test_utils import (  # noqa: E402
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_nvidia_mega.minimax_m31.runtime import SHIPPED_SLOTS, SLOTS_ENV  # noqa: E402
from sgl_nvidia_mega.plugin import FLAG_CLI  # noqa: E402

LAUNCH_TIMEOUT = 3600
PARITY_ENV = "SGLANG_NVIDIA_MEGAKERNEL_PARITY"
NO_AUTOTUNE = "--disable-flashinfer-autotune"
PROMPTS = 16
# prompts per request: each prompt alone, then the first 2, 4, 8 and 16 together
BATCHES = (1, 2, 4, 8, 16)
MAX_NEW_TOKENS = 256
# SGLang's startup line and its line once every decode graph is captured, per TP rank
ACTIVE = f"NVIDIA megakernel: active (MiniMax-M3.1: slots {sorted(SHIPPED_SLOTS)} ("
GRAPHS = "NVIDIA megakernel: MiniMax-M3.1 runs the decode graphs of batch sizes ["


def _chat_ids(tokenizer, i: int) -> list[int]:
    """Prompt ``i`` as a user turn of MiniMax's chat template, in token ids."""
    ids = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": f"Question {i}: explain step by step whether "
                f"{i} * 17 + 3 is odd.",
            }
        ],
        add_generation_prompt=True,
        tokenize=True,
    )
    # a list of ids, or a BatchEncoding (transformers 5)
    return list(ids["input_ids"] if isinstance(ids, Mapping) else ids)


def _generate(prompts: list) -> list:
    """Greedy decodes of ``prompts`` (token ids) sent as one request, from an empty
    radix cache: each prompt's output token ids and their log-probabilities."""
    r = requests.post(
        DEFAULT_URL_FOR_TEST + "/flush_cache", params={"timeout": 60}, timeout=120
    )
    r.raise_for_status()
    r = requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            "input_ids": prompts,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": MAX_NEW_TOKENS,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": -1,
        },
        timeout=1800,
    )
    r.raise_for_status()
    out = r.json()
    return [
        (o["output_ids"], [lp for lp, *_ in o["meta_info"]["output_token_logprobs"]])
        for o in (out if isinstance(out, list) else [out])
    ]


def _serve(prompts: list, megakernel: bool):
    """Boot one server, decode every request, shut it down: ({(prompts per request,
    prompt): (token ids, log-probabilities)}, the server's log)."""
    logs = tuple(tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err"))
    process = popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=LAUNCH_TIMEOUT,
        other_args=ARGS + [NO_AUTOTUNE] + ([FLAG_CLI] if megakernel else []),
        # named, not inherited: the launch merges the caller's environment; the
        # served path, without the parity switch
        env={
            **os.environ,
            **ENV,
            SLOTS_ENV: ",".join(SHIPPED_SLOTS),
            PARITY_ENV: "0",
        },
        return_stdout_stderr=logs,
    )
    try:
        outputs = {}
        for n in BATCHES:
            # each prompt alone, or the first n prompts in one request
            for sent in [[i] for i in range(PROMPTS)] if n == 1 else [list(range(n))]:
                decoded = _generate([prompts[i] for i in sent])
                outputs.update(((n, i), d) for i, d in zip(sent, decoded))
    finally:
        terminate_and_kill_process_tree(process)
        # read once the server has exited, with its output flushed
        log = "".join(Path(f.name).read_text(errors="replace") for f in logs)
        for f in logs:
            f.close()
    return outputs, log


@unittest.skipIf(torch.cuda.device_count() < TP, f"needs {TP} GPUs (TP{TP})")
class TestServedDecodesAreSGLangs(CustomTestCase):
    """The shipped slots in the decode CUDA graphs against SGLang's forward."""

    @classmethod
    def setUpClass(cls):
        warm_remote_code()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        prompts = [_chat_ids(tokenizer, i) for i in range(PROMPTS)]
        cls.on = _serve(prompts, megakernel=True)
        cls.off = _serve(prompts, megakernel=False)

    def test_the_megakernel_serves_the_decode_graphs(self):
        """On every rank: the shipped slots built, and the decode graphs of every
        request size captured with the megakernel; no parity check ran. The other
        server never activates it."""
        log = self.on[1]
        self.assertEqual(log.count(ACTIVE), TP, ACTIVE)
        graphs = [
            line.split(GRAPHS, 1)[1].split("]", 1)[0]
            for line in log.splitlines()
            if GRAPHS in line
        ]
        self.assertEqual(len(graphs), TP, graphs)
        for sizes in graphs:
            self.assertLessEqual(set(BATCHES), {int(s) for s in sizes.split(",")})
        self.assertNotIn("NVIDIA megakernel parity", log)
        self.assertNotIn("NVIDIA megakernel: active", self.off[1])

    def test_every_prompt_decodes_sglangs_tokens(self):
        """The same token ids and log-probabilities for every prompt of every
        request."""
        on, off = self.on[0], self.off[0]
        self.assertEqual(sorted(on), sorted(off))
        differ = []
        for key in sorted(off):
            self.assertEqual(len(off[key][0]), MAX_NEW_TOKENS, key)
            for what, got, want in zip(("token", "log-probability"), on[key], off[key]):
                if got != want:
                    first = next(
                        (t for t, (a, b) in enumerate(zip(got, want)) if a != b),
                        min(len(got), len(want)),
                    )
                    differ.append(f"{key}: {what} {first} differs")
        print(
            f"{len(off)} decodes of {MAX_NEW_TOKENS} tokens compared, "
            f"{len(differ)} differ"
        )
        self.assertEqual(differ, [], f"(prompts per request, prompt): {differ[:8]}")


if __name__ == "__main__":
    unittest.main()
