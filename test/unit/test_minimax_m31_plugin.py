# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The plugin's registration of the MiniMax M3.1 megakernel on CPU: the package's
architecture resolves to its spec through SGLang's registry, and the plugin's rows
for the spec are exactly the package's architectures."""

import pytest

pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

try:
    from sglang.kernels.plugin_kernels.nvidia import spec as registry  # noqa: E402
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels import spec as registry  # noqa: E402

from sgl_nvidia_mega import minimax_m31, plugin  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import spec  # noqa: E402

PATH = "sgl_nvidia_mega.minimax_m31.spec:SPEC"


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.delenv(plugin.PARITY_SORT_ENV, raising=False)
    plugin.register()


def test_the_architecture_resolves_through_the_plugin():
    for architecture in minimax_m31.ARCHITECTURES:
        for isa in ("sm_100", "sm_103"):
            found, why = registry.find_spec(architectures=[architecture], isa=isa)
            assert found is spec.SPEC, why
        # the spec serves the SM100 family only, and the refusal names the ISA
        found, why = registry.find_spec(architectures=[architecture], isa="sm_90")
        assert found is None
        assert "sm_90" in why


def test_the_plugin_registers_the_package_architectures():
    rows = [architecture for architecture, path in plugin.SPECS if path == PATH]
    # minimax_m31.ARCHITECTURES is the one list: each of them once, no other
    assert sorted(rows) == sorted(set(minimax_m31.ARCHITECTURES))
