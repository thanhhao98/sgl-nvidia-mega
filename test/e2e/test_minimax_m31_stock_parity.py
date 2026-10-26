# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel's covered steps against SGLang's verify step, bit for bit, on GB300
TP4 with MiniMax M3.1-Flash and DSpark (8 rows a request).

``SGLANG_NVIDIA_MEGAKERNEL_PARITY=1`` (test only) runs SGLang's forward and then the
megakernel's on every eager covered step and logs the largest difference of the logits
and of the DSpark hidden states. ``TestStockStepIsSGLangs``: with every slot off the
covered step runs each package's stock code, which must reproduce SGLang's step
exactly. ``TestShippedSlotsStepIsSGLangs``: the slots the package ships are bitwise,
so the same holds with them on. Each server's startup line shows what it built; chat
requests 1, 2, 4, 8 and 16 at a time make verify steps of 8, 16, 32, 64 and 128 rows,
and requests that finish first leave the widths in between.

The checkpoint is the one ``minimax_m31_server`` names; the tests skip on fewer than 4
GPUs.
"""

import concurrent.futures
import os
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

import requests  # noqa: E402
import torch  # noqa: E402
from minimax_m31_server import (  # noqa: E402
    ARGS,
    ENV,
    MODEL,
    ROWS_PER_REQUEST,
    TP,
    warm_remote_code,
)
from sglang.test.test_utils import (  # noqa: E402
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_nvidia_mega.minimax_m31.plan import MAX_ROWS  # noqa: E402
from sgl_nvidia_mega.minimax_m31.runtime import SHIPPED_SLOTS, SLOTS_ENV  # noqa: E402
from sgl_nvidia_mega.plugin import FLAG_CLI  # noqa: E402

LAUNCH_TIMEOUT = 3600
PARITY_ENV = "SGLANG_NVIDIA_MEGAKERNEL_PARITY"
# The executor's parity line (runtime/executor.py): "NVIDIA megakernel parity
# [control] <mode> rows=<n> logits_max_abs=<difference> hidden_max_abs=<difference>";
# groups: control, mode, rows, logits, rel, % differ, hidden
LINE = re.compile(
    r"NVIDIA megakernel parity( control)? (\S+) rows=(\d+) "
    r"logits_max_abs=(\S+)(?: \(rel ([^,]+), ([\d.]+)% differ\))?"
    r".*?hidden_max_abs=(\S+)"
)
# a difference renders as "<max> (rel ...)", equal tensors as "0 (rel ...)", a tensor
# neither forward returns as "n/a"
EQUAL = ("0", "n/a")
CONCURRENCY = (1, 2, 4, 8, 16)
# the verify steps that hold all of one level's requests
REQUIRED_WIDTHS = {ROWS_PER_REQUEST * n for n in CONCURRENCY}
# every covered width: the verify steps of 1-16 requests
WIDTHS = list(range(ROWS_PER_REQUEST, MAX_ROWS + 1, ROWS_PER_REQUEST))
MAX_TOKENS = 256


def _active(slots) -> re.Pattern:
    """SGLang's startup line, once per TP rank: the requested slots built (the widths
    each serves are left open), every covered width, bitwise numerics."""
    names = sorted(slots) or "none (stock modules)"
    return re.compile(
        re.escape(f"NVIDIA megakernel: active (MiniMax-M3.1: slots {names} (")
        + r".*?"
        + re.escape(
            f"); verify steps of {WIDTHS[0]}-{WIDTHS[-1]} rows ({ROWS_PER_REQUEST} a "
            f"request); widths {WIDTHS}; bitwise numerics;"
        )
    )


@unittest.skipIf(torch.cuda.device_count() < TP, f"needs {TP} GPUs (TP{TP})")
class _StockParity(CustomTestCase):
    """One server with the parity switch: its eager covered steps compared bit for
    bit."""

    SLOTS: tuple = ()

    @classmethod
    def setUpClass(cls):
        cls.process, cls.logs = None, ()
        warm_remote_code()
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=LAUNCH_TIMEOUT,
            # eager verify steps: the parity check runs outside CUDA graphs only
            other_args=ARGS + [FLAG_CLI, "--disable-decode-cuda-graph"],
            # the slots named, not inherited: the launch merges the caller's environment
            env={
                **os.environ,
                **ENV,
                SLOTS_ENV: ",".join(cls.SLOTS),
                PARITY_ENV: "1",
            },
            return_stdout_stderr=cls.logs,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
        for log in cls.logs:
            log.close()

    def _chat(self, i: int) -> None:
        r = requests.post(
            DEFAULT_URL_FOR_TEST + "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Question {i}: explain step by step whether "
                        f"{i} * 17 + 3 is odd.",
                    }
                ],
                "temperature": 0,
                "max_tokens": MAX_TOKENS,
            },
            timeout=600,
        )
        r.raise_for_status()

    def _log(self) -> str:
        return "".join(Path(f.name).read_text(errors="replace") for f in self.logs)

    def check(self):
        """The startup line on every rank, then the parity lines of the requests'
        steps."""
        startup = self._log()
        self.assertEqual(
            len(_active(self.SLOTS).findall(startup)),
            TP,
            re.findall(r"NVIDIA megakernel: active.*", startup)[:TP],
        )
        for n in CONCURRENCY:
            with concurrent.futures.ThreadPoolExecutor(n) as pool:
                list(pool.map(self._chat, range(n)))
        found = LINE.findall(self._log())
        self.assertTrue(found, "no parity line: the covered steps did not run")
        differ = [
            line for line in found if line[3] not in EQUAL or line[6] not in EQUAL
        ]
        self.assertEqual(
            differ,
            [],
            f"{len(differ)} of {len(found)} parity lines differ, e.g. {differ[:4]}",
        )
        # lines per width whose logits were compared
        widths = Counter(int(line[2]) for line in found if line[3] == "0")
        print(
            f"{type(self).__name__}: {len(found)} parity lines, all bitwise; lines "
            f"per width: {dict(sorted(widths.items()))}"
        )
        self.assertLessEqual(REQUIRED_WIDTHS, set(widths), sorted(widths))


class TestStockStepIsSGLangs(_StockParity):
    def test_every_covered_step_is_bitwise(self):
        self.check()


class TestShippedSlotsStepIsSGLangs(_StockParity):
    SLOTS = SHIPPED_SLOTS

    def test_every_covered_step_is_bitwise(self):
        self.check()


if __name__ == "__main__":
    unittest.main()
