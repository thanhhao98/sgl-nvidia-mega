# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The switch on GB300 / B300 TP4 with EAGLE 3/1/4: served decodes in decode CUDA
graphs, token for token, with and without --enable-nvidia-megakernel (stock: every
slot off).

Both servers are step5_server's (``--max-running-requests 8``, the FlashInfer tactics
fixed with ``--disable-flashinfer-autotune``); they flush their radix cache before
each batch and decode 16 prompts greedily alone, then 2, 4 and 8 at a time (verify
steps of 4-32 rows). The megakernel server must log the active line on every rank and
run the decode graphs of batch sizes 1-8.
"""

import concurrent.futures
import os
import re
import tempfile
import unittest
from pathlib import Path

import requests
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)
from step5_server import ARGS, ENV, MODEL, NO_MODEL

from sgl_nvidia_mega.plugin import FLAG_CLI

LAUNCH_TIMEOUT = 3600
PROMPTS = [
    f"Question {i}: explain step by step whether {i} * 17 + 3 is odd."
    for i in range(16)
]
COVERAGE = re.compile(
    r"NVIDIA megakernel: Step-5 runs the decode graphs of batch sizes "
    r"\[1, 2, 3, 4, 5, 6, 7, 8\]"
)


def _decode(url, batch):
    def one(prompt):
        r = requests.post(
            url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 128},
            },
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["output_ids"]

    out = []
    for start in range(0, len(PROMPTS), batch):
        requests.post(url + "/flush_cache", timeout=60)
        with concurrent.futures.ThreadPoolExecutor(batch) as pool:
            out += list(pool.map(one, PROMPTS[start : start + batch]))
    return out


@unittest.skipUnless(MODEL, NO_MODEL)
class TestStep5ServedTokensAreSGLangs(CustomTestCase):
    def _serve(self, flag):
        logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=LAUNCH_TIMEOUT,
            other_args=ARGS + flag,
            env={**os.environ, **ENV, "SGLANG_NVIDIA_MEGAKERNEL_SLOTS": ""},
            return_stdout_stderr=logs,
        )
        try:
            ids = {b: _decode(DEFAULT_URL_FOR_TEST, b) for b in (1, 2, 4, 8)}
            text = "".join(Path(f.name).read_text(errors="replace") for f in logs)
        finally:
            terminate_and_kill_process_tree(process)
            for f in logs:
                f.close()
        return ids, text

    def test_the_stock_megakernel_serves_sglangs_tokens(self):
        off, off_log = self._serve([])
        on, on_log = self._serve([FLAG_CLI])
        self.assertNotIn("NVIDIA megakernel: active", off_log)
        self.assertEqual(on_log.count("NVIDIA megakernel: active (Step-5:"), 4)
        self.assertRegex(on_log, COVERAGE)
        for batch in off:
            self.assertEqual(on[batch], off[batch], f"batch {batch}")


if __name__ == "__main__":
    unittest.main()
