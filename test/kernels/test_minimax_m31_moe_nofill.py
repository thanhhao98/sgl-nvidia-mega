# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot moe_nofill of the MiniMax M3.1 megakernel on one SM100-family GPU (B200, B300,
GB300), at one TP4 rank's shapes (hidden 6144; 128 experts, top-4, intermediate 768) and
every served width (1-16 rows, 8-128 rows step 8):

- PerTokenQuant against FlashInfer's nvfp4_quantize(per_token_activation=True,
  backend="cute-dsl", sfLayout=layout_linear) as SGLang's routed MoE calls it: the
  kernel FlashInfer compiles for that call is the one the slot launches (its key), and
  the FP4 bytes, the block-scale bytes and the per-token scales match bit for bit,
  written over garbage-filled buffers (so no buffer needs a zero fill), eagerly and
  replayed from a CUDA graph, under FlashInfer's default,
  exact-math and 4over6 environments; and the build's own check (PerTokenQuant.verify);
- the slot's routed core (moe_nofill.routed_fp4) against SGLang's
  fused_experts_none_to_flashinfer_trtllm_fp4 under the deferred finalize, on random
  NVFP4 experts processed by MiniMaxM3Nvfp4MoE as SGLang loads them: the GEMM2 row
  each (token, expert) slot points to bit for bit, and the slot map when SGLang's own
  is run-to-run stable, eagerly and replayed from a CUDA graph. The expert weights are
  the caller's top-k weights handed back (FlashInfer's FP4 routed MoE borrows that
  buffer): each path runs on its own copy of the routing, and the slot must hand back
  its own exactly when SGLang does, with the same values;
- the served path (moe_nofill.forward) against SGLang's MiniMaxM3MoE.forward: two
  consecutive MoE layers captured into one CUDA graph per path as a covered step
  captures them, with SGLang's router and TopK (the Triton moe_fused_gate, a PDL kernel
  the PDL quantize now directly follows), the shared expert on the main stream and the
  routed experts on one alternate stream, the second layer reading the first layer's
  finalized output, and both layers on one set of slot buffers: every handoff field and
  the final output bit for bit, replayed over garbage-filled buffers;
- the launches: the slot's chain makes exactly one fill launch fewer than SGLang's (its
  global scale's torch.full; FlashInfer 0.7.0 also zero-filled the block scales).

A bitwise check needs SGLang's own result to repeat bit for bit run to run; when it does
not, the check fails and says so (it never passes on a tolerance).

Run inside the serving container:

    python3 -m pytest -q -s test/kernels/test_minimax_m31_moe_nofill.py
"""

import contextlib
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

# The quantization package before the MoE runner module (the gate imports it): the
# reverse order is a circular import on CUDA.
pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

# isort: off
from sglang.srt.layers.quantization import minimax_m3_nvfp4  # noqa: E402
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm  # noqa: E402

# isort: on
from sglang.srt.configs.shared_experts import SharedExpertsConfig  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # noqa: E402
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig  # noqa: E402
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner  # noqa: E402
from sglang.srt.layers.moe.token_dispatcher.standard import (  # noqa: E402
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK  # noqa: E402
from sglang.srt.layers.moe.utils import MoeRunnerBackend  # noqa: E402
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E402
    CompressedTensorsFusedMoEMethod,
)
from sglang.srt.models import minimax_m3  # noqa: E402
from sglang.srt.models.deepseek_common.utils import (  # noqa: E402
    tiny_router_gemm_max_tokens,
)
from sglang.srt.runtime_context import (  # noqa: E402
    get_context,
    get_model,
    get_parallel,
)

from sgl_nvidia_mega.minimax_m31.moe import moe_nofill  # noqa: E402

HIDDEN, EXPERTS, TOP_K, INTER = 6144, 128, 4, 768
ALPHA, LIMIT = 1.702, 7.0
# M3.1's routed_scaling_factor, which its TopK applies to the output weights
ROUTED_SCALE = 2.0
WIDTHS = sorted(set(range(1, 17)) | set(range(8, 129, 8)))
MAX_ROWS = max(WIDTHS)
FI_ENVS = (
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH",
    "FLASHINFER_NVFP4_4OVER6",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH",
)
ENV_CASES = {
    "default": {},
    "exact-math": {"FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1"},
    "4over6": {"FLASHINFER_NVFP4_4OVER6": "1"},
    "4over6-e4m3-256": {
        "FLASHINFER_NVFP4_4OVER6": "1",
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
    },
}
QUANT_OUTPUTS = ("fp4 bytes", "block-scale bytes", "per-token scales")


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


needs_sm100 = pytest.mark.skipif(
    not _sm100(), reason="needs an SM100-family GPU (B200 / B300 / GB300)"
)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1).view(torch.uint8)


def _assert_bitwise(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert (got.dtype, tuple(got.shape)) == (want.dtype, tuple(want.shape)), (
        what,
        got.dtype,
        want.dtype,
        tuple(got.shape),
        tuple(want.shape),
    )
    differ = int((_bits(got) != _bits(want)).sum())
    assert differ == 0, f"{what}: {differ} of {_bits(want).numel()} bytes differ"


def _require_stable(want: torch.Tensor, control: torch.Tensor, what: str) -> None:
    """SGLang's result ``want`` and the same run again, ``control``: a bitwise claim
    against SGLang needs the two to agree bit for bit."""
    differ = int((_bits(want) != _bits(control)).sum())
    if differ:
        pytest.fail(
            f"{what}: SGLang's own result differs between two runs of the same "
            f"inputs ({differ} of {_bits(want).numel()} bytes), so the slot's "
            "bitwise claim cannot be checked here"
        )


def _rows(rows: int, seed: int) -> torch.Tensor:
    """Normed hidden rows, with a zero row and an outlier from 4 rows on."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda")
    if rows >= 4:
        x[1] = 0
        x[2, 100] = 300.0
    return x.to(torch.bfloat16)


def _routing(rows: int, seed: int):
    """Distinct experts per row; positive weights summing to the routed scale 2."""
    g = torch.Generator(device="cuda").manual_seed(seed + 1000)
    scores = torch.rand(rows, EXPERTS, generator=g, device="cuda")
    top, ids = scores.topk(TOP_K, dim=-1)
    return ids.to(torch.int32), (top / top.sum(-1, keepdim=True) * 2.0).float()


def _garbage(quant: moe_nofill.PerTokenQuant) -> None:
    """Stale bytes in every buffer: the kernel must write all that is read."""
    quant.fp4.fill_(0x5A)
    quant.block_scale.fill_(0xA5)
    quant.per_token_scale.fill_(float("nan"))


@contextlib.contextmanager
def _flashinfer_env(monkeypatch, env: dict):
    for name in FI_ENVS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    yield


def _sglang_quantize(x: torch.Tensor):
    """The per-token branch of fused_experts_none_to_flashinfer_trtllm_fp4, as written
    there: the global scale from torch.full, then FlashInfer's nvfp4_quantize."""
    from flashinfer import SfLayout, nvfp4_quantize

    e4m3_max = 448.0
    if (
        envs.FLASHINFER_NVFP4_4OVER6.get()
        and envs.FLASHINFER_NVFP4_4OVER6_E4M3_USE_256.get()
    ):
        e4m3_max = 256.0
    global_scale_inv = torch.full(
        (1,), 1.0 / (e4m3_max * 6.0), dtype=torch.float32, device=x.device
    )
    hs_fp4_bytes, hs_sf_bytes, per_token_scale = nvfp4_quantize(
        x,
        global_scale_inv,
        sfLayout=SfLayout.layout_linear,
        per_token_activation=True,
        backend="cute-dsl",
    )
    seq_len, hidden_size = x.shape
    hs_fp4 = hs_fp4_bytes.reshape(seq_len, hidden_size // 2)
    hs_scale = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(seq_len, hidden_size // 16)
    return hs_fp4, hs_scale, per_token_scale


# ------------------------------------------------------------------- the quantize


@needs_sm100
@pytest.mark.parametrize("env", list(ENV_CASES), ids=list(ENV_CASES))
def test_quantize_matches_flashinfer_bitwise(env, monkeypatch):
    with _flashinfer_env(monkeypatch, ENV_CASES[env]):
        quant = moe_nofill.PerTokenQuant(
            max_rows=MAX_ROWS, hidden=HIDDEN, device="cuda"
        )
        assert quant.verify() == []
        # ascending, then descending: the buffers hold a wider step's bytes too
        for i, rows in enumerate(WIDTHS + WIDTHS[::-1]):
            x = moe_nofill.check_rows(rows, HIDDEN) if i % 2 else _rows(rows, i)
            with moe_nofill.recorded_kernel_keys() as keys:
                want = _sglang_quantize(x)
            # FlashInfer compiled, for SGLang's call, the kernel the slot launches
            assert keys == [quant.kernel_key(x)], (env, rows, keys)
            _garbage(quant)
            got = quant(x)
            for name, a, b in zip(QUANT_OUTPUTS, got, want):
                _assert_bitwise(a, b, f"{env}: {name} at {rows} rows")
        print(f"{env}: kernel key and quantize bitwise at {len(WIDTHS)} widths")


@needs_sm100
def test_quantize_replays_from_a_cuda_graph():
    quant = moe_nofill.PerTokenQuant(max_rows=MAX_ROWS, hidden=HIDDEN, device="cuda")
    for rows in (1, 8, 13, 64, 128):
        x = _rows(rows, 0)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            quant(x)  # compiles outside the graph
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = quant(x)
        for seed in (1, 2):
            x.copy_(_rows(rows, seed))
            _garbage(quant)
            graph.replay()
            want = _sglang_quantize(x)
            for name, a, b in zip(QUANT_OUTPUTS, out, want):
                _assert_bitwise(a, b, f"replayed {name} at {rows} rows")


# ----------------------------------------------------------------- the routed MoE


def _nvfp4_params(seed: int) -> dict:
    """One TP4 rank's routed experts in compressed-tensors' NVFP4 layout, random."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    e, h, i = EXPERTS, HIDDEN, INTER

    def packed(*shape):
        return torch.randint(
            0, 256, shape, generator=g, dtype=torch.uint8, device="cuda"
        )

    def scales(*shape):
        t = torch.empty(*shape, device="cuda").uniform_(0.25, 2.0, generator=g)
        return t.to(torch.float8_e4m3fn)

    return dict(
        w13_weight_packed=packed(e, 2 * i, h // 2),
        w13_weight_scale=scales(e, 2 * i, h // 16),
        w13_weight_global_scale=torch.full((e, 2), 448.0 * 6, device="cuda"),
        w13_input_global_scale=torch.ones(e, 2, device="cuda"),
        w2_weight_packed=packed(e, h, i // 2),
        w2_weight_scale=scales(e, h, i // 16),
        w2_weight_global_scale=torch.full((e,), 448.0 * 6, device="cuda"),
        w2_input_global_scale=torch.ones(e, device="cuda"),
    )


def _load_experts(layer: torch.nn.Module, seed: int):
    """``layer`` with one TP4 rank's routed experts (EP 1) as SGLang loads them for the
    TRT-LLM routed runner: random NVFP4 weights processed by MiniMaxM3Nvfp4MoE. Returns
    the scheme."""
    for name, t in _nvfp4_params(seed).items():
        layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))
    layer.num_experts = layer.num_local_experts = EXPERTS
    layer.moe_ep_rank = 0
    layer.intermediate_size_per_partition = INTER
    layer.routing_method_type = None
    layer._minimax_activation = "swigluoai"
    layer._mega_moe_trtllm_layout = False
    layer.moe_runner_config = MoeRunnerConfig(
        num_experts=EXPERTS,
        num_local_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTER,
        top_k=TOP_K,
        gemm1_alpha=ALPHA,
        gemm1_beta=1.0,
        gemm1_clamp_limit=LIMIT,
        inplace=False,
    )
    with mock.patch.object(
        minimax_m3_nvfp4,
        "get_moe_runner_backend",
        return_value=MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
    ):
        scheme = minimax_m3_nvfp4.MiniMaxM3Nvfp4MoE()
    assert scheme.use_flashinfer_trtllm and not scheme.use_mega_moe_prefill
    scheme._process_trtllm_weights(layer)
    return scheme


@pytest.fixture(scope="module")
def experts():
    """One TP4 rank's routed experts as SGLang loads them, and the quant info its
    apply_weights hands the runner."""
    layer = torch.nn.Module()
    scheme = _load_experts(layer, seed=0)
    captured = []
    scheme.runner = SimpleNamespace(run=lambda d, q: captured.append(q))
    scheme.apply_weights(layer, None)
    (quant_info,) = captured
    # the slot builds the same quant info from the layer
    mine = moe_nofill.quant_info(layer)
    for name, want in vars(quant_info).items():
        got = getattr(mine, name)
        if isinstance(want, torch.Tensor):
            assert (got.data_ptr(), got.shape, got.dtype) == (
                want.data_ptr(),
                want.shape,
                want.dtype,
            ), name
        else:
            assert got == want, name
    return SimpleNamespace(layer=layer, quant_info=quant_info)


def _dispatch(x, ids, weights):
    return StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=weights, topk_ids=ids, router_logits=None
        ),
    )


def _stock(experts, x, ids, weights):
    """SGLang's routed core under the deferred finalize (FusedMoE's
    forward_deferred_finalize around the runner's fused function)."""
    with flashinfer_trtllm.flashinfer_trtllm_deferred_finalize_context():
        return flashinfer_trtllm.fused_experts_none_to_flashinfer_trtllm_fp4(
            _dispatch(x, ids, weights),
            experts.quant_info,
            experts.layer.moe_runner_config,
            use_routed_topk=True,
        ).hidden_states


def _slot(experts, quant, x, ids, weights):
    with flashinfer_trtllm.flashinfer_trtllm_deferred_finalize_context():
        return moe_nofill.routed_fp4(
            _dispatch(x, ids, weights),
            experts.quant_info,
            experts.layer.moe_runner_config,
            quant,
        ).hidden_states


def _slot_rows(out) -> torch.Tensor:
    """The GEMM2 row each (token, expert) slot points to: what the finalize reads."""
    idx = out.expanded_idx_to_permuted_idx.reshape(-1).long()
    return out.gemm2_out.reshape(-1, out.gemm2_out.shape[-1]).index_select(0, idx)


@contextlib.contextmanager
def _spy_quantize(store: list):
    """FlashInfer's nvfp4_quantize as SGLang's chain calls it, its outputs kept."""
    import flashinfer

    real = flashinfer.nvfp4_quantize

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        store.append(tuple(t.clone() for t in out))
        return out

    with mock.patch.object(flashinfer, "nvfp4_quantize", spy):
        yield


def _compare(got, want, control, what: str, *, got_weights, want_weights) -> str:
    """The deferred triples of the slot (``got``) and of SGLang (``want``;
    ``control`` is SGLang's run again), each path on its own copy of the top-k
    weights (``got_weights``, ``want_weights``)."""
    assert got.top_k == want.top_k == TOP_K, what
    # FlashInfer's FP4 routed MoE hands the caller's top-k weights back as the expert
    # weights (it borrows the buffer): the slot must hand back its own exactly when
    # SGLang does, and the values must agree
    borrowed = want.expert_weights.data_ptr() == want_weights.data_ptr()
    assert (got.expert_weights.data_ptr() == got_weights.data_ptr()) == borrowed, what
    _assert_bitwise(got.expert_weights, want.expert_weights, f"{what}: expert weights")
    want_rows = _slot_rows(want)
    _require_stable(want_rows, _slot_rows(control), f"{what}: GEMM2 rows")
    _assert_bitwise(_slot_rows(got), want_rows, f"{what}: GEMM2 rows")
    notes = ["the expert weights are the caller's, handed back"] if borrowed else []
    # The slot map is FlashInfer's order of each expert's rows, which its routing
    # assigns in launch-dependent order (FlashInfer 0.7.2, from 32 rows): what the
    # finalize reads is the rows it selects, asserted bitwise above.
    if not torch.equal(
        want.expanded_idx_to_permuted_idx, control.expanded_idx_to_permuted_idx
    ):
        notes.append("SGLang's slot map varies run to run, the rows it selects do not")
    elif not torch.equal(
        got.expanded_idx_to_permuted_idx, want.expanded_idx_to_permuted_idx
    ):
        notes.append("the slot map orders an expert's rows otherwise, the rows agree")
    return f"{what}: deferred triple bitwise" + (
        f" ({'; '.join(notes)})" if notes else ""
    )


@needs_sm100
def test_routed_core_matches_sglang_bitwise(experts, monkeypatch):
    with _flashinfer_env(monkeypatch, {}):
        quant = moe_nofill.PerTokenQuant(
            max_rows=MAX_ROWS, hidden=HIDDEN, device="cuda"
        )
        for rows in WIDTHS:
            x = _rows(rows, rows)
            ids, weights = _routing(rows, rows)
            # each path on its own copy of the routing
            stock_weights = weights.clone()
            stock_quant = []
            with _spy_quantize(stock_quant), moe_nofill.recorded_kernel_keys() as keys:
                want = _stock(experts, x, ids.clone(), stock_weights)
            control = _stock(experts, x, ids.clone(), weights.clone())
            _garbage(quant)
            got = _slot(experts, quant, x, ids, weights)
            assert keys == [quant.kernel_key(x)], (rows, keys)
            (stock_out,) = stock_quant
            buffers = (
                quant.fp4[:rows],
                quant.block_scale[: rows * HIDDEN // 16].view(torch.uint8),
                quant.per_token_scale[:rows],
            )
            want_quant = (
                stock_out[0].reshape(rows, -1),
                stock_out[1].reshape(-1).view(torch.uint8),
                stock_out[2],
            )
            for name, a, b in zip(QUANT_OUTPUTS, buffers, want_quant):
                _assert_bitwise(a, b, f"{name} at {rows} rows")
            print(
                _compare(
                    got,
                    want,
                    control,
                    f"{rows} rows",
                    got_weights=weights,
                    want_weights=stock_weights,
                )
            )


@needs_sm100
@pytest.mark.parametrize("rows", [1, 8, 16, 64, 128])
def test_routed_core_replays_from_a_cuda_graph(experts, rows):
    quant = moe_nofill.PerTokenQuant(max_rows=MAX_ROWS, hidden=HIDDEN, device="cuda")
    x = _rows(rows, 0)
    ids, weights = _routing(rows, 0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _slot(experts, quant, x, ids, weights)  # compiles and loads outside the graph
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _slot(experts, quant, x, ids, weights)
    for seed in (1, 2):
        x.copy_(_rows(rows, seed))
        new_ids, new_weights = _routing(rows, seed)
        ids.copy_(new_ids)
        weights.copy_(new_weights)
        _garbage(quant)
        graph.replay()
        stock_weights = weights.clone()
        want = _stock(experts, x, ids.clone(), stock_weights)
        control = _stock(experts, x, ids.clone(), weights.clone())
        what = f"replayed {rows} rows, seed {seed}"
        print(
            _compare(
                out,
                want,
                control,
                what,
                got_weights=weights,
                want_weights=stock_weights,
            )
        )


@needs_sm100
def test_the_slot_launches_one_fewer_fill(experts):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    quant = moe_nofill.PerTokenQuant(max_rows=MAX_ROWS, hidden=HIDDEN, device="cuda")
    rows = 8
    x = _rows(rows, 0)
    ids, weights = _routing(rows, 0)

    def device_ops(fn):
        fn()  # warm
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        return [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]

    stock = device_ops(lambda: _stock(experts, x, ids, weights))
    slot = device_ops(lambda: _slot(experts, quant, x, ids, weights))
    if not stock:
        pytest.skip("the profiler recorded no device activity (CUPTI unavailable)")

    def fills(names):
        return [n for n in names if "FillFunctor" in n or "memset" in n.lower()]

    print(f"SGLang's chain: {len(stock)} device ops, fills {fills(stock)}")
    print(f"the slot's chain: {len(slot)} device ops, fills {fills(slot)}")
    # SGLang's torch.full of the global scale (FlashInfer 0.7.2 no longer zero-fills
    # the block scales, which 0.7.0 did)
    assert len(fills(stock)) - len(fills(slot)) == 1
    assert len(stock) - len(slot) == 1


# ---------------------------------------------------------------- the served path


def _weight(rows: int, cols: int, g: torch.Generator) -> torch.Tensor:
    return (torch.randn(rows, cols, generator=g, device="cuda") / cols**0.5).to(
        torch.bfloat16
    )


class _Dispatcher:
    """StandardDispatcher at EP 1 without the FP4 all-gather (M3.1's configuration):
    the rows and the top-k pass through."""

    def dispatch(self, hidden_states, topk_output):
        return StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

    def combine(self, combine_input):
        (hidden_states,) = combine_input
        return hidden_states


class _Experts(FusedMoE):
    """A FusedMoE with ``_load_experts``'s routed experts behind
    CompressedTensorsFusedMoEMethod, on the runner the served configuration builds
    (MoeRunner over fused_experts_none_to_flashinfer_trtllm_routed). FusedMoE.__init__
    needs a server; SGLang's forward methods run unchanged."""

    def __init__(self, seed: int):
        torch.nn.Module.__init__(self)
        self.scheme = _load_experts(self, seed)
        # as FusedMoE.__init__ resolves it from the quant method
        self.runner_backend = MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
        self.moe_runner_config.runner_backend = self.runner_backend
        runner = object.__new__(MoeRunner)
        runner.config = self.moe_runner_config
        runner.fused_func = (
            flashinfer_trtllm.fused_experts_none_to_flashinfer_trtllm_routed
        )
        runner.lora_enabled = False
        self.scheme.runner = runner
        self.quant_method = CompressedTensorsFusedMoEMethod(None)
        self.dispatcher = _Dispatcher()
        self.hidden_size = HIDDEN
        self.moe_ep_size = 1
        self.supports_deferred_finalize = True


class _SharedExpert(torch.nn.Module):
    """A shared expert's work on the main stream: bf16 GEMMs (the served MiniMaxM3MLP
    is MXFP8; both paths only call it)."""

    def __init__(self, g: torch.Generator):
        super().__init__()
        self.gate_up = _weight(2 * INTER, HIDDEN, g)
        self.down = _weight(HIDDEN, INTER, g)

    def forward(self, x):
        gate, up = torch.nn.functional.linear(x, self.gate_up).chunk(2, dim=-1)
        return torch.nn.functional.linear(
            torch.nn.functional.silu(gate) * up, self.down
        )


class _Moe(minimax_m3.MiniMaxM3MoE):
    """A MiniMaxM3MoE of one TP4 rank (MiniMaxM3MoE.__init__ needs a server): SGLang's
    forward, router (the bf16 gate of the served configuration) and shared-expert call
    run unchanged, over SGLang's TopK as M3.1 builds it (sigmoid top-4 with the
    correction bias, renormalized, the routed scale on the output weights) and
    ``_Experts``."""

    def __init__(self, layer_id: int, alt_stream: torch.cuda.Stream):
        torch.nn.Module.__init__(self)
        g = torch.Generator(device="cuda").manual_seed(1000 + layer_id)
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.num_fused_shared_experts = 0
        self.shared_experts = _SharedExpert(g)
        self.bf16_router_gemm = True
        self.gate = SimpleNamespace(weight=_weight(EXPERTS, HIDDEN, g))
        self.tiny_router_gemm_max_tokens = tiny_router_gemm_max_tokens(
            num_experts=EXPERTS, hidden_size=HIDDEN, weight_dtype=torch.bfloat16
        )
        # TopK reads the model's shared-expert configuration: M3.1's shared expert is
        # not fused into the routed slots
        with (
            get_context().override_server_args(),
            get_model().override(shared_experts=SharedExpertsConfig()),
        ):
            self.topk = TopK(
                top_k=TOP_K,
                layer_id=layer_id,
                renormalize=True,
                scoring_func="sigmoid",
                correction_bias=torch.randn(EXPERTS, generator=g, device="cuda") / 8,
                routed_scaling_factor=ROUTED_SCALE,
                apply_routed_scaling_factor_on_output=True,
            )
        self.experts = _Experts(seed=layer_id)
        self.topk.bind_runner_backend(self.experts.runner_backend)
        self.supports_deferred_finalize = True


@pytest.fixture(scope="module")
def moe_layers():
    """Two consecutive MoE layers of one TP4 rank on one alternate stream, as the model
    hands every layer get_stream("alt")."""
    alt = torch.cuda.Stream()
    return [_Moe(layer_id, alt) for layer_id in (3, 4)]


@contextlib.contextmanager
def _served_step():
    """What MiniMaxM3MoE.forward reads of the server in a captured step that defers
    the MoE finalize: capture mode, the deferred finalize, deterministic inference
    off; the TP group TopK's allocation context names (one rank: no symmetric
    memory); and SGLang's resolved server arguments, which TopK's routing reads
    (no expert-location dispatch)."""
    deterministic = SimpleNamespace(enable_deterministic_inference=False)
    with contextlib.ExitStack() as stack:
        stack.enter_context(get_context().override_server_args())
        for name, value in (
            ("get_is_capture_mode", lambda: True),
            ("get_forward", lambda: SimpleNamespace(defer_moe_finalize=True)),
            ("get_exec", lambda: SimpleNamespace(deterministic=deterministic)),
        ):
            stack.enter_context(mock.patch.object(minimax_m3, name, value))
        stack.enter_context(
            get_parallel().override(tp_group=SimpleNamespace(world_size=1))
        )
        yield


def _handoff_rows(handoff) -> torch.Tensor:
    """The GEMM2 row each (token, expert) slot of a deferred handoff points to."""
    idx = handoff.permuted_indices.reshape(-1).long()
    return handoff.routed_output.index_select(0, idx)


def _finalize(handoff) -> torch.Tensor:
    """What the next boundary takes of a deferred handoff (SGLang fuses it into the
    all-reduce): each row's GEMM2 rows by its expert weights plus the gated shared
    output, on the current stream, after the layer's join."""
    rows = _handoff_rows(handoff).float().view(handoff.m, TOP_K, -1)
    routed = (rows * handoff.expert_weights.float()[..., None]).sum(1)
    return (routed + handoff.gated_shared_output.float()).to(torch.bfloat16)


def _two_layers(moe, layers, x):
    """Two consecutive MoE layers of a covered step: the second reads the first's
    finalized output."""
    first = moe(layers[0], x)
    second = moe(layers[1], _finalize(first))
    return first, second, _finalize(second)


def _capture(fn):
    """``fn()`` captured into a CUDA graph after one run outside it (which compiles
    and loads the kernels), as SGLang captures a decode step: (graph, its outputs)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def _replay(graph, out) -> dict:
    """Replay ``graph``; copies of what its two layers hand on, in step order (so a
    failed comparison names the first piece that differs)."""
    graph.replay()
    first, second, output = out
    snapshot = {}
    for name, handoff in (("layer 1", first), ("layer 2", second)):
        snapshot[f"{name} expert weights"] = handoff.expert_weights.clone()
        snapshot[f"{name} slot map"] = handoff.permuted_indices.clone()
        snapshot[f"{name} GEMM2 rows"] = _handoff_rows(handoff)
        snapshot[f"{name} shared output"] = handoff.gated_shared_output.clone()
    snapshot["output"] = output.clone()
    return snapshot


@needs_sm100
@pytest.mark.parametrize("rows", [1, 8, 13, 16, 64, 128])
def test_two_moe_layers_in_one_graph_match_sglang_bitwise(moe_layers, rows):
    quant = moe_nofill.PerTokenQuant(max_rows=MAX_ROWS, hidden=HIDDEN, device="cuda")
    x = _rows(rows, 0)

    def sglang(mlp, h):
        return mlp(h, forward_batch=None)

    def slot(mlp, h):
        return moe_nofill.forward(mlp, h, quant)

    with _served_step():
        graphs = {
            name: _capture(lambda moe=moe: _two_layers(moe, moe_layers, x))
            for name, moe in (("SGLang", sglang), ("slot", slot))
        }
    for seed in (1, 2):
        x.copy_(_rows(rows, seed))
        _garbage(quant)
        got = _replay(*graphs["slot"])
        want = _replay(*graphs["SGLang"])
        control = _replay(*graphs["SGLang"])
        what = f"{rows} rows, seed {seed}"
        output = want["output"].float()
        assert torch.isfinite(output).all() and output.abs().max() > 0, what
        notes = []
        for name, value in want.items():
            if name.endswith("slot map") and not torch.equal(value, control[name]):
                notes.append(f"SGLang's {name} varies run to run")
                continue
            _require_stable(value, control[name], f"{what}: {name}")
            _assert_bitwise(got[name], value, f"{what}: {name}")
        print(
            f"{what}: two MoE layers in one graph bitwise"
            + (f" ({'; '.join(notes)})" if notes else "")
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
