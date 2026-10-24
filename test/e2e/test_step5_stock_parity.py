# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The stock step against SGLang's, bit for bit, on GB300 / B300 TP4 with EAGLE 3/1/4.

With every slot off the covered step runs each package's stock code, which must
reproduce SGLang's verify step exactly at every covered width (4-32 rows). The parity
switch (test only) runs SGLang's forward, then the megakernel's, on every eager verify
step and logs the elements that differ in the logits and in the FP32 hidden states
EAGLE's draft extend reads. ``control`` runs SGLang's forward twice: the run-to-run
floor and the measurement that a verify step run twice (its KV, CSA ring and
compressed-key writes) sees the same state. Batches of 1-8 concurrent requests reach
every width. The servers are step5_server's (``--max-running-requests 8``).
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
from step5_server import ARGS, ENV, MODEL, NO_MODEL, REQUESTS, WIDTHS

from sgl_nvidia_mega.plugin import FLAG_CLI

LAUNCH_TIMEOUT = 3600
LINE = re.compile(
    r"NVIDIA megakernel parity( control)? target_verify rows=(\d+) "
    r"logits_max_abs=.* logits_bits_differ=(\S+) hidden_bits_differ=(\S+)"
)


class _Parity(CustomTestCase):
    MODE = "1"

    @classmethod
    def setUpClass(cls):
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=LAUNCH_TIMEOUT,
            # eager verify steps: the parity check runs outside CUDA graphs only
            other_args=ARGS + [FLAG_CLI, "--disable-decode-cuda-graph"],
            env={
                **os.environ,
                **ENV,
                # the stock step unless a test names slots
                "SGLANG_NVIDIA_MEGAKERNEL_SLOTS": os.environ.get(
                    "SGLANG_NVIDIA_MEGAKERNEL_STEP5_E2E_SLOTS", ""
                ),
                "SGLANG_NVIDIA_MEGAKERNEL_PARITY": cls.MODE,
            },
            return_stdout_stderr=cls.logs,
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process)
        for log in cls.logs:
            log.close()

    def _generate(self, i: int) -> str:
        r = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": f"Question {i}: explain step by step whether {i} * 17 + 3 is odd.",
                "sampling_params": {"temperature": 0, "max_new_tokens": 64},
            },
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["text"]

    def check(self):
        for batch in REQUESTS:
            with concurrent.futures.ThreadPoolExecutor(batch) as pool:
                list(pool.map(self._generate, range(batch)))
        text = "".join(Path(f.name).read_text(errors="replace") for f in self.logs)
        # n/a: a forward without logits (SGLang's autotune dummy run)
        found = [m for m in LINE.findall(text) if m[2] != "n/a"]
        self.assertTrue(found, "no parity line: the covered steps did not run")
        differ = [m for m in found if (m[2], m[3]) != ("0", "0")]
        self.assertEqual(differ, [], f"{len(differ)} of {len(found)} steps differ")
        self.assertLessEqual(WIDTHS, {int(m[1]) for m in found})


@unittest.skipUnless(MODEL, NO_MODEL)
class TestStep5StockStepIsSGLangs(_Parity):
    def test_every_covered_width_is_bitwise(self):
        self.check()


@unittest.skipUnless(MODEL, NO_MODEL)
class TestStep5SGLangsStepRunsTwiceAlike(_Parity):
    MODE = "control"

    def test_every_covered_width_is_bitwise_run_to_run(self):
        self.check()


if __name__ == "__main__":
    unittest.main()
