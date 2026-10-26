# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# This file has been modified from its original version (the per-token NVFP4 quantize
# launched into buffers built once).
# Adapted from FlashInfer's flashinfer/quantization/kernels/nvfp4_quantize.py
# (nvfp4_quantize_per_token_cute_dsl; flashinfer-ai/flashinfer v0.7.2).
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/minimax_m3.py
# (MiniMaxM3MoE.forward_normal, MiniMaxM3MoE._forward_router_experts),
# python/sglang/srt/layers/moe/fused_moe_triton/layer.py
# (FusedMoE.forward_deferred_finalize),
# python/sglang/srt/layers/quantization/minimax_m3_nvfp4.py
# (MiniMaxM3Nvfp4MoE.apply_weights) and
# python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py
# (fused_experts_none_to_flashinfer_trtllm_fp4).
"""Slot ``moe_nofill``: an MoE layer's routed experts without the fill launch and the
allocations SGLang's chain makes per layer.

SGLang quantizes the routed experts' input to NVFP4 per token before FlashInfer's
TRT-LLM routed MoE (``fused_experts_none_to_flashinfer_trtllm_fp4``, branch
``use_per_token_activation``). One of that chain's launches only builds an input:
``torch.full`` of the global scale ``1 / (e4m3_max * 6)`` (``FillFunctor<float>``);
FlashInfer's ``nvfp4_quantize_per_token_cute_dsl`` then allocates its three outputs
(FlashInfer 0.7.0 also zeroed the block scales, a second fill; 0.7.2 leaves them to
the kernel, which in the linear layout writes all ``K / 16`` scale bytes of each row).

``PerTokenQuant`` builds the global scale once and launches FlashInfer's compiled
per-token kernel itself, under the key ``nvfp4_quantize_per_token_cute_dsl`` computes,
into buffers built once. ``forward`` runs ``MiniMaxM3MoE.forward_normal``'s deferred
finalize branch with SGLang's own calls in SGLang's order (shared expert, router,
top-k, dispatch, the TRT-LLM routed MoE with the stock arguments, combine, on the same
streams), so the layer's output is SGLang's bit for bit.

With slot hop_ffn the FFN-input boundary has already written the routed experts'
NVFP4 input and the shared expert's MXFP8 input: ``forward`` then takes them
(``Prequantized`` as the routed experts' quantize, ``shared_input`` for the shared
expert), so neither quantize launches.

``build`` checks every MoE layer for the chain the slot follows (``layer_reasons``;
module hooks count against it, so ``--enable-layerwise-nvtx-marker`` and
``--debug-tensor-dump-output-folder`` do not combine with the slot) and that
FlashInfer's own call compiles and runs the kernel the slot launches
(``PerTokenQuant.verify``). It serves every width when every rank finds both, else
none. ``fallback_reason`` checks one step's preconditions; a step that fails one runs
SGLang's module for that layer.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
from typing import Any, Iterator, Optional

import msgspec
import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # SGLang's TRT-LLM MoE runner module, reached as fit.<name> at call time
    # (importlib, in the stock import order)
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "FlashInferTrtllmFp4MoeQuantInfo",
        "is_deferred_finalize_enabled",
        "get_activation_type",
        "get_activation_type(is_gated)",
        "_get_routing_for_flashinfer_routed",
        "_routing_top_k",
        "next_power_of_2",
        "trtllm_moe_enable_pdl",
        "_make_deferred_finalize_output",
        "_make_deferred_finalize_output(top_k)",
        "flashinfer_trtllm_deferred_finalize_context",
        "fused_experts_none_to_flashinfer_trtllm_routed",
        "fused_experts_none_to_flashinfer_trtllm_fp4",
        # the FlashInferTrtllmFp4MoeQuantInfo fields quant_info passes and
        # routed_fp4 reads
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "g1_scale_c",
        "g1_alphas",
        "g2_alphas",
        "w13_input_scale_quant",
        "global_num_experts",
        "local_expert_offset",
        "local_num_experts",
        "intermediate_size_per_partition",
        "routing_method_type",
        "use_per_token_activation",
        "padded_hidden_size",
        "gemm1_alpha",
        "gemm1_beta",
        "gemm1_clamp_limit",
    ),
    "srt/layers/quantization/minimax_m3_nvfp4.py": (
        "MiniMaxM3Nvfp4MoE",
        "use_marlin",
        "use_flashinfer_trtllm",
        "use_mega_moe_prefill",
        "runner",
        "apply_weights",
        # the experts' gemm1 activation parameters, set by the scheme
        "gemm1_alpha",
        "gemm1_beta",
        "gemm1_clamp_limit",
    ),
    "srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py": (
        "w13_input_scale_quant",
        "g1_scale_c",
        "g1_alphas",
        "g2_alphas",
        "w13_weight_scale",
        "w2_weight_scale",
    ),
    "srt/environ.py": (
        "envs",
        "FLASHINFER_NVFP4_4OVER6",
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256",
    ),
    # the dispatch output routed_fp4 reads, the combine input it builds and the
    # dispatcher's combine
    "srt/layers/moe/token_dispatcher/standard.py": (
        "StandardCombineInput",
        "hidden_states",
        "hidden_states_scale",
        "topk_output",
        "combine(combine_input)",
    ),
    "srt/layers/moe/utils.py": (
        "sum_post_experts_output",
        "should_use_flashinfer_cutlass_moe_fp4_allgather",
        # get_moe_a2a_backend()'s MoeA2ABackend
        "is_none",
    ),
    # read through the model module's namespace, as MiniMaxM3MoE.forward_normal does
    "srt/models/minimax_m3.py": (
        "MiniMaxM3MoE",
        "num_fused_shared_experts",
        "get_is_capture_mode",
        "is_in_breakable_cuda_graph",
        "MoeDeferredFinalize",
        "get_moe_a2a_backend",
        "get_forward",
        "alt_stream",
        "shared_experts",
        "_forward_shared_experts",
        "_compute_router_logits",
        "topk",
        "experts",
        "forward_normal",
        "_forward_router_experts",
        "supports_deferred_finalize",
        "is_layer_sparse",
        "layer_id",
        "mlp",
        "forward",
        "start_layer",
        "end_layer",
        "layers",
    ),
    "srt/models/minimax_m3_vl.py": ("model",),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "FusedMoE",
        "forward_deferred_finalize",
        "_dispatch_with_pre_quant",
        "run_moe_core",
        "moe_ep_size",
        "num_local_experts",
        "num_experts",
        "quant_method",
        "hidden_size",
        '"w13_weight"',
        '"w2_weight"',
        "dispatcher",
        "moe_ep_rank",
        "scheme",
        "intermediate_size_per_partition",
        "routing_method_type",
        "supports_deferred_finalize",
    ),
    "srt/layers/moe/moe_runner/runner.py": (
        "MoeRunner",
        "lora_enabled",
        "fused_func",
        "config",
        "run",
    ),
    "srt/layers/quantization/compressed_tensors/compressed_tensors.py": (
        "CompressedTensorsFusedMoEMethod",
        "apply",
    ),
    "srt/layers/layer_boundary/fusions/*.py": (
        "from_flashinfer",
        "from_flashinfer(gated_shared_output)",
        "from_flashinfer(m)",
        "from_flashinfer(reduce)",
    ),
    "srt/runtime_context.py": ('"defer_moe_finalize"',),
    "srt/layers/moe/moe_runner/base.py": ("activation", "is_gated"),
}

logger = logging.getLogger(__name__)

# NVFP4's block size: one E4M3 scale per 16 values (FlashInfer's NVFP4_SF_VEC_SIZE)
SF_VEC = 16
# The values SGLang's per-token global scale 1 / (e4m3_max * 6) takes (e4m3_max)
E4M3_MAXES = (448.0, 256.0)
# The activations the stock chain serves (_SUPPORTED_FP4_ACTIVATIONS of
# fused_experts_none_to_flashinfer_trtllm_fp4)
FP4_ACTIVATIONS = frozenset({"silu", "relu2", "gelu", "situ"})
# The activations' dtype the build checks the kernel on (the spec serves bfloat16)
DTYPE = torch.bfloat16


def _flashinfer_trtllm():
    """SGLang's FlashInfer TRT-LLM MoE runner module. The stock chain resolves its
    helpers there at call time, and so does the slot (patches included)."""
    # The quantization package first: the reverse order is a circular import on CUDA.
    importlib.import_module("sglang.srt.layers.quantization.minimax_m3_nvfp4")
    return importlib.import_module("sglang.srt.layers.moe.moe_runner.flashinfer_trtllm")


def e4m3_max() -> float:
    """The stock chain's ``e4m3_max``: 256 when SGLang reads both
    ``FLASHINFER_NVFP4_4OVER6`` and ``FLASHINFER_NVFP4_4OVER6_E4M3_USE_256`` as set,
    else 448 (SGLang's reading, which may differ from FlashInfer's)."""
    from sglang.srt.environ import envs

    if (
        envs.FLASHINFER_NVFP4_4OVER6.get()
        and envs.FLASHINFER_NVFP4_4OVER6_E4M3_USE_256.get()
    ):
        return 256.0
    return 448.0


@contextlib.contextmanager
def recorded_kernel_keys() -> Iterator[list[tuple]]:
    """The keys FlashInfer compiles its per-token NVFP4 kernel under while in the
    context, in call order: each call's ``_get_compiled_kernel_nvfp4_per_token``
    arguments bound to its signature, defaults applied (the order ``kernel_key``
    returns). ``nvfp4_quantize_per_token_cute_dsl`` looks the getter up in its module
    on every call, so wrapping that name sees what FlashInfer's own call asks for."""
    import inspect

    from flashinfer.quantization.kernels import nvfp4_quantize as nq

    real = nq._get_compiled_kernel_nvfp4_per_token
    signature = inspect.signature(real)
    keys: list[tuple] = []

    def recording(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        keys.append(tuple(bound.arguments.values()))
        return real(*args, **kwargs)

    nq._get_compiled_kernel_nvfp4_per_token = recording
    try:
        yield keys
    finally:
        nq._get_compiled_kernel_nvfp4_per_token = real


class PerTokenQuant:
    """FlashInfer's per-token NVFP4 quantize of the routed experts' input as SGLang
    calls it (``nvfp4_quantize(x, 1 / (e4m3_max * 6), sfLayout=layout_linear,
    per_token_activation=True, backend="cute-dsl")``), into buffers built once.

    ``nvfp4_quantize`` hands the call to ``nvfp4_quantize_per_token_cute_dsl``, which
    allocates the packed FP4 output, the block scales and the per-token scales and
    launches the kernel compiled for ``(dtype, K, layout, PDL, fast math, 4over6, folded
    output scale, CTA width)``, the CTA width chosen from ``K`` and the row count.
    ``__call__`` computes that key the same way, from the same input and environment,
    per call (covered steps run while the decode graphs are captured, so this host work
    is never replayed), and launches the same kernel on slices of the buffers below
    with a global scale built as the stock ``torch.full`` builds it. The kernel writes
    every byte its consumers read: rows ``[0, m)`` of the FP4 output and of the
    per-token scales, and in the linear layout all ``m * K / 16`` block scales.

    One set of buffers serves every MoE layer and every width. Within a covered step,
    layer L's quantize writes them and layer L's TRT-LLM MoE launches read them, on one
    stream (``mlp.alt_stream`` while capturing, the current stream otherwise); nothing
    else reads them (the deferred triple the next boundary consumes is FlashInfer's own
    output) and the main stream never touches them. Layer L + 1 issues its quantize on
    the alternate stream only after its ``alt_stream.wait_stream(current)``, which
    the main stream records after layer L's ``current.wait_stream(alt_stream)`` join,
    so layer L + 1's writes follow every read of layer L. A captured graph keeps both
    joins as full dependencies; PDL overlaps kernels only within the stream segment
    after a join, and the kernel waits on its predecessor (``griddepcontrol.wait``)
    before it touches memory. Eager steps run every launch on one stream, in order, and
    steps (graph replays or eager) run one after another on the forward stream.
    """

    def __init__(self, *, max_rows: int, hidden: int, device):
        self.max_rows, self.hidden = int(max_rows), int(hidden)
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            # "cuda" names the current device; inputs carry an explicit index
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.fp4 = torch.empty(
            self.max_rows, self.hidden // 2, dtype=torch.uint8, device=self.device
        )
        self.block_scale = torch.empty(
            self.max_rows * (self.hidden // SF_VEC),
            dtype=torch.uint8,
            device=self.device,
        )
        self.per_token_scale = torch.empty(
            self.max_rows, dtype=torch.float32, device=self.device
        )
        # the stock torch.full, once for each e4m3_max the environment may select
        self.global_scale_inv = {
            e: torch.full(
                (1,), 1.0 / (e * 6.0), dtype=torch.float32, device=self.device
            )
            for e in E4M3_MAXES
        }

    def nbytes(self) -> int:
        tensors = (self.fp4, self.block_scale, self.per_token_scale)
        return sum(t.numel() * t.element_size() for t in tensors) + 4 * len(
            self.global_scale_inv
        )

    @staticmethod
    def kernel_key(x: torch.Tensor) -> tuple:
        """``_get_compiled_kernel_nvfp4_per_token``'s arguments as
        ``nvfp4_quantize_per_token_cute_dsl`` computes them for ``x`` now. The stock
        ``nvfp4_quantize`` passes ``enable_pdl=None`` (the device decides), no
        ``out_scale`` (no folded output scale) and the 4over6 setting the environment
        resolves to."""
        from flashinfer import utils as fi_utils
        from flashinfer.quantization.kernels import nvfp4_quantize as nq

        dtype_key = {torch.float16: "float16", torch.bfloat16: "bfloat16"}[x.dtype]
        m, k = (int(d) for d in x.shape)
        return (
            dtype_key,
            k,
            nq.SF_LAYOUT_LINEAR,
            fi_utils.device_support_pdl(x.device),
            nq._env_flag_enabled("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"),
            nq.resolve_nvfp4_4over6(),
            False,
            nq._per_token_cta_threads(k, m),
        )

    def __call__(self, x: torch.Tensor):
        """``(hs_fp4 [m, K / 2] uint8, hs_scale [m, K / 16] float8_e4m3fn,
        per_token_scale [m] float32)``: views of the buffers, shaped as the stock
        chain hands them to the MoE."""
        from flashinfer.quantization.kernels import nvfp4_quantize as nq

        m, k = x.shape
        if not (
            0 < m <= self.max_rows
            and k == self.hidden
            and x.is_contiguous()
            and x.device == self.device
        ):
            raise ValueError(
                f"moe_nofill: input {tuple(x.shape)} on {x.device} (contiguous "
                f"{x.is_contiguous()}) does not fit buffers of {self.max_rows} x "
                f"{self.hidden} on {self.device}"
            )
        launch = nq._get_compiled_kernel_nvfp4_per_token(*self.kernel_key(x))
        fp4 = self.fp4[:m]
        block_scale = self.block_scale[: m * (k // SF_VEC)]
        per_token_scale = self.per_token_scale[:m]
        global_scale_inv = self.global_scale_inv[e4m3_max()]
        # (input, fp4, block scales, per-token scales, rows, padded rows (the linear
        # layout pads none), global scale, output scale (the global scale when none is
        # folded)), as nvfp4_quantize_per_token_cute_dsl launches it
        launch(
            x,
            fp4,
            block_scale,
            per_token_scale,
            m,
            m,
            global_scale_inv,
            global_scale_inv,
        )
        hs_scale = block_scale.view(torch.float8_e4m3fn).reshape(m, k // SF_VEC)
        return fp4, hs_scale, per_token_scale

    def verify(self, dtype: torch.dtype = DTYPE) -> list[str]:
        """Launch against FlashInfer's own ``nvfp4_quantize`` as SGLang calls it, on
        synthetic rows at one row and at every buffer row (at build, outside any
        graph; it compiles the kernel): why the slot's launch is not that call's ([]
        when it is). FlashInfer's call must compile its kernel under the slot's
        ``kernel_key`` (``recorded_kernel_keys``: a variant that writes the same
        bytes, such as PDL on or off, shows only there), and the outputs must agree
        bit for bit."""
        from flashinfer import SfLayout, nvfp4_quantize

        reasons = []
        for m in sorted({1, self.max_rows}):
            x = check_rows(m, self.hidden, dtype=dtype, device=self.device)
            scale = torch.full(
                (1,), 1.0 / (e4m3_max() * 6.0), dtype=torch.float32, device=self.device
            )
            with recorded_kernel_keys() as keys:
                fp4, sf, pts = nvfp4_quantize(
                    x,
                    scale,
                    sfLayout=SfLayout.layout_linear,
                    per_token_activation=True,
                    backend="cute-dsl",
                )
            key = self.kernel_key(x)
            if keys != [key]:
                reasons.append(
                    f"at {m} rows FlashInfer's nvfp4_quantize compiles its per-token "
                    f"kernel under {keys}, the slot under {[key]}"
                )
            want = (
                fp4.reshape(m, -1),
                sf.view(torch.float8_e4m3fn).reshape(m, -1),
                pts,
            )
            got = self(x)
            for name, a, b in zip(
                ("fp4", "block scales", "per-token scales"), got, want
            ):
                if a.shape != b.shape or not torch.equal(
                    a.view(torch.uint8), b.view(torch.uint8)
                ):
                    reasons.append(
                        f"the {name} at {m} rows differ from FlashInfer's "
                        "nvfp4_quantize"
                    )
        return reasons


class Prequantized:
    """The routed experts' NVFP4 input already written for rows ``x`` (slot hop_ffn's
    FFN-input boundary): ``routed_fp4``'s quantize, handing out ``nvfp4`` (``(hs_fp4,
    hs_scale, per_token_scale)``, as ``PerTokenQuant`` returns them) for exactly those
    rows and quantizing any other rows with ``quant`` (``PerTokenQuant``).

    The dispatch hands the routed core the rows it was given (no a2a backend, no FP4
    all-gather: ``config_reasons``), so the triple stands for ``x``'s quantize. Its
    buffers are hop_ffn's, written on the main stream before the routed chain's
    ``alt_stream.wait_stream`` (hop_ffn's docstring has their reuse)."""

    def __init__(self, x: torch.Tensor, nvfp4: tuple, quant: PerTokenQuant):
        self.x, self.nvfp4, self.quant = x, nvfp4, quant

    def __call__(self, hidden_states: torch.Tensor):
        if hidden_states is self.x:
            return self.nvfp4
        return self.quant(hidden_states)


def check_rows(
    m: int, hidden: int, *, dtype: torch.dtype = DTYPE, device="cuda"
) -> torch.Tensor:
    """Deterministic rows for ``PerTokenQuant.verify``: Gaussian rows of magnitudes
    1e-3 to 1e3, and from row 1 on rows that reach the kernel's other branches: all
    zeros (zero amax), one value throughout, one large outlier, tiny values."""
    g = torch.Generator(device="cpu").manual_seed(31)
    x = torch.randn(m, hidden, generator=g, dtype=torch.float32)
    x *= torch.logspace(-3, 3, m, dtype=torch.float32)[:, None]
    special = (
        torch.zeros(hidden),
        torch.full((hidden,), -1.5),
        torch.zeros(hidden).index_fill_(0, torch.tensor([hidden // 3]), 3.0e4),
        torch.randn(hidden, generator=g) * 1e-6,
    )
    for i, row in enumerate(special[: m - 1]):
        x[i + 1] = row
    return x.to(dtype=dtype, device=device).contiguous()


# --------------------------------------------------------------------- the chain


def quant_info(layer) -> Any:
    """The ``FlashInferTrtllmFp4MoeQuantInfo`` ``MiniMaxM3Nvfp4MoE.apply_weights``
    hands its runner (the TRT-LLM branch) for the experts ``layer`` as they are now."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmFp4MoeQuantInfo,
    )

    return FlashInferTrtllmFp4MoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_weight_scale=layer.w13_weight_scale,
        w2_weight_scale=layer.w2_weight_scale,
        g1_scale_c=layer.g1_scale_c,
        g1_alphas=layer.g1_alphas,
        g2_alphas=layer.g2_alphas,
        w13_input_scale_quant=layer.w13_input_scale_quant[:1],
        global_num_experts=layer.num_experts,
        local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
        local_num_experts=layer.num_local_experts,
        intermediate_size_per_partition=layer.intermediate_size_per_partition,
        routing_method_type=layer.routing_method_type,
        use_per_token_activation=True,
        gemm1_alpha=layer.gemm1_alpha,
        gemm1_beta=layer.gemm1_beta,
        gemm1_clamp_limit=layer.gemm1_clamp_limit,
    )


def routed_fp4(dispatch_output, quant_info, runner_config, quant: PerTokenQuant):
    """``fused_experts_none_to_flashinfer_trtllm_fp4(..., use_routed_topk=True)``
    under the deferred finalize, for the branch every served layer takes: the
    dispatch hands bf16 rows, per-token activations, no hidden padding. The quantize
    runs on ``quant``'s buffers (or ``quant`` is a ``Prequantized``); the MoE call
    and its wrapping are the stock ones."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    fit = _flashinfer_trtllm()
    assert runner_config.activation in FP4_ACTIVATIONS, runner_config.activation
    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    if not (
        dispatch_output.hidden_states_scale is None
        and quant_info.padded_hidden_size is None
        and quant_info.use_per_token_activation
        and fit.is_deferred_finalize_enabled()
    ):
        raise RuntimeError(
            "moe_nofill: the routed MoE left the branch the slot was built for "
            "(bf16 dispatch, unpadded, per-token activations, deferred finalize)"
        )
    hs_fp4, hs_scale, per_token_scale = quant(hidden_states)
    activation_type = fit.get_activation_type(
        runner_config.activation, is_gated=runner_config.is_gated
    )
    routing = fit._get_routing_for_flashinfer_routed(topk_output)
    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=routing,
        routing_bias=None,
        hidden_states=hs_fp4,
        hidden_states_scale=hs_scale,
        gemm1_weights=quant_info.w13_weight,
        gemm1_weights_scale=quant_info.w13_weight_scale.view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=quant_info.gemm1_alpha,
        gemm1_beta=quant_info.gemm1_beta,
        gemm1_clamp_limit=quant_info.gemm1_clamp_limit,
        gemm2_weights=quant_info.w2_weight,
        gemm2_weights_scale=quant_info.w2_weight_scale.view(torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=quant_info.g1_scale_c,
        output1_scale_gate_scalar=quant_info.g1_alphas,
        output2_scale_scalar=quant_info.g2_alphas,
        per_token_scale=per_token_scale,
        num_experts=quant_info.global_num_experts,
        top_k=fit._routing_top_k(routing),
        n_group=0,
        topk_group=0,
        intermediate_size=quant_info.intermediate_size_per_partition,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=1,  # unused, but must be 1 to pass validation
        do_finalize=False,
        activation_type=activation_type,
        tune_max_num_tokens=fit.next_power_of_2(hs_fp4.shape[0]),
        enable_pdl=fit.trtllm_moe_enable_pdl(hs_fp4.shape[0]),
    )
    result = fit._make_deferred_finalize_output(
        result, top_k=fit._routing_top_k(routing)
    )
    return StandardCombineInput(hidden_states=result)


def experts_deferred(experts, hidden_states, topk_output, quant: PerTokenQuant):
    """``FusedMoE.forward_deferred_finalize(hidden_states, topk_output)``: SGLang's
    dispatch and combine around the routed core (``run_moe_core``: the quant method's
    ``apply``, the scheme's ``apply_weights``, its runner's fused function)."""
    fit = _flashinfer_trtllm()
    dispatch_output = experts._dispatch_with_pre_quant(hidden_states, topk_output, None)
    with fit.flashinfer_trtllm_deferred_finalize_context():
        combine_input = routed_fp4(
            dispatch_output,
            quant_info(experts),
            experts.scheme.runner.config,
            quant,
        )
    return experts.dispatcher.combine(combine_input=combine_input)


def _router_experts(mlp, x, quant: PerTokenQuant):
    """``MiniMaxM3MoE._forward_router_experts(x, defer_finalize=True)``."""
    router_logits = mlp._compute_router_logits(x)
    topk_output = mlp.topk(x, router_logits)
    return experts_deferred(mlp.experts, x, topk_output, quant)


def _shared_experts(mlp, x: torch.Tensor, shared_input):
    """``mlp._forward_shared_experts(x)``, its shared expert called on
    ``shared_input``: ``x``, or ``x``'s MXFP8 pair (slot hop_ffn), which the shared
    expert's gate_up takes in place of quantizing ``x``."""
    if shared_input is x:
        return mlp._forward_shared_experts(x)
    if x.shape[0] > 0 and mlp.num_fused_shared_experts == 0:
        return mlp.shared_experts(shared_input)
    return None


def forward(mlp, x: torch.Tensor, quant: PerTokenQuant, *, shared_input=None):
    """``MiniMaxM3MoE.forward(x)`` (no a2a backend: ``forward_normal``) of a step that
    defers the finalize, on rows ``x``: the shared expert on the current stream and the
    routed experts on ``mlp.alt_stream`` while capturing (SGLang's condition),
    otherwise one after the other; the deferred-finalize handoff. ``shared_input`` is
    what the shared expert is called on (default ``x``; slot hop_ffn: the MXFP8 pair),
    ``quant`` the routed experts' quantize (slot hop_ffn: a ``Prequantized``)."""
    from sglang.srt.layers.moe.utils import sum_post_experts_output
    from sglang.srt.models.minimax_m3 import (
        MoeDeferredFinalize,
        get_is_capture_mode,
        is_in_breakable_cuda_graph,
    )

    shared_input = x if shared_input is None else shared_input
    if (
        mlp.alt_stream is not None
        and mlp.shared_experts is not None
        and get_is_capture_mode()
        and not is_in_breakable_cuda_graph()
    ):
        current_stream = torch.cuda.current_stream()
        mlp.alt_stream.wait_stream(current_stream)
        shared_output = _shared_experts(mlp, x, shared_input)
        with torch.cuda.stream(mlp.alt_stream):
            routed = _router_experts(mlp, x, quant)
        current_stream.wait_stream(mlp.alt_stream)
    else:
        shared_output = _shared_experts(mlp, x, shared_input)
        routed = _router_experts(mlp, x, quant)
    return MoeDeferredFinalize.from_flashinfer(
        routed,
        gated_shared_output=shared_output,
        m=x.shape[0],
        reduce=sum_post_experts_output,
    )


# ------------------------------------------------------------ build and fallback


class Bound(msgspec.Struct):
    """``build``'s result: the quantize buffers and the widths the slot serves (every
    built width, or none), each MoE layer's module as checked, by layer id, and why
    the slot serves none ("" when it serves)."""

    widths: tuple = ()
    quant: Any = None
    mlps: dict = msgspec.field(default_factory=dict)
    reason: str = ""

    def serves(self, rows: int) -> bool:
        return rows in self.widths


def _overridden(obj, cls, names) -> list[str]:
    return [
        f"{type(obj).__name__}.{name} replaces {cls.__name__}.{name}"
        for name in names
        if getattr(type(obj), name, None) is not getattr(cls, name)
        or name in getattr(obj, "__dict__", {})
    ]


def call_reason(module: torch.nn.Module) -> str:
    """Why calling ``module`` would run more than its class's ``forward`` (hooks, a
    compiled or replaced forward), "" when it would not: the slot calls the MoE's
    pieces in place of ``mlp(x)``, so it would skip them."""
    import torch.nn.modules.module as nn_module

    if type(module).__call__ is not torch.nn.Module.__call__:
        return f"{type(module).__name__} replaces __call__"
    if (
        "forward" in module.__dict__
        or getattr(module, "_compiled_call_impl", None) is not None
    ):
        return "the module's forward is replaced or compiled"
    if (
        module._forward_hooks
        or module._forward_pre_hooks
        or module._backward_hooks
        or module._backward_pre_hooks
        or nn_module._global_forward_hooks
        or nn_module._global_forward_pre_hooks
        or nn_module._global_backward_hooks
        or nn_module._global_backward_pre_hooks
    ):
        return (
            "the module has forward or backward hooks, which the slot would skip "
            "(--enable-layerwise-nvtx-marker and --debug-tensor-dump-output-folder "
            "register them at load)"
        )
    return ""


def layer_reasons(layer) -> list[str]:
    """Why ``layer``'s MoE does not take the chain the slot follows ([] when it
    does): ``MiniMaxM3MoE.forward_normal`` over ``FusedMoE.forward_deferred_finalize``,
    the packed NVFP4 experts on ``MiniMaxM3Nvfp4MoE``'s TRT-LLM routed runner
    (per-token activations), its weights unpadded, at EP 1.

    Hooks on the MoE module count too (``call_reason``). The ones present here were
    registered at load and are never removed, so with them the slot could only skip
    them or fall back at every step; it serves no width instead, and ``runtime.build``
    refuses a server that requests it: ``--enable-layerwise-nvtx-marker`` and
    ``--debug-tensor-dump-output-folder`` do not combine with ``moe_nofill``."""
    fit = _flashinfer_trtllm()
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsFusedMoEMethod,
    )
    from sglang.srt.layers.quantization.minimax_m3_nvfp4 import MiniMaxM3Nvfp4MoE
    from sglang.srt.models.minimax_m3 import MiniMaxM3MoE

    mlp = layer.mlp
    if not isinstance(mlp, MiniMaxM3MoE):
        return [f"the MLP is {type(mlp).__name__}, not MiniMaxM3MoE"]
    reasons = _overridden(
        mlp, MiniMaxM3MoE, ("forward", "forward_normal", "_forward_router_experts")
    )
    if why := call_reason(mlp):
        reasons.append(why)
    if not mlp.supports_deferred_finalize:
        reasons.append("the MoE does not defer its finalize")
    experts = mlp.experts
    if not isinstance(experts, FusedMoE):
        return reasons + [f"the experts are {type(experts).__name__}, not FusedMoE"]
    reasons += _overridden(
        experts,
        FusedMoE,
        ("forward_deferred_finalize", "_dispatch_with_pre_quant", "run_moe_core"),
    )
    if not experts.supports_deferred_finalize:
        reasons.append("the experts do not defer their finalize")
    if experts.moe_ep_size != 1 or experts.num_local_experts != experts.num_experts:
        reasons.append(
            f"expert parallelism (EP {experts.moe_ep_size}, "
            f"{experts.num_local_experts} of {experts.num_experts} experts local)"
        )
    method = experts.quant_method
    if not isinstance(method, CompressedTensorsFusedMoEMethod):
        reasons.append(
            f"the experts run on {type(method).__name__}, not "
            "CompressedTensorsFusedMoEMethod"
        )
    else:
        reasons += _overridden(method, CompressedTensorsFusedMoEMethod, ("apply",))
    scheme = getattr(experts, "scheme", None)
    if not isinstance(scheme, MiniMaxM3Nvfp4MoE):
        return reasons + [
            f"the experts' scheme is {type(scheme).__name__}, not MiniMaxM3Nvfp4MoE"
        ]
    reasons += _overridden(scheme, MiniMaxM3Nvfp4MoE, ("apply_weights",))
    if (
        scheme.use_marlin
        or not scheme.use_flashinfer_trtllm
        or scheme.use_mega_moe_prefill
    ):
        reasons.append(
            "the NVFP4 experts do not run on the TRT-LLM routed runner alone "
            f"(use_flashinfer_trtllm {scheme.use_flashinfer_trtllm}, marlin "
            f"{scheme.use_marlin}, MegaMoE prefill {scheme.use_mega_moe_prefill})"
        )
    runner = getattr(scheme, "runner", None)
    if not isinstance(runner, MoeRunner):
        reasons.append(f"the scheme's runner is {type(runner).__name__}")
    else:
        reasons += _overridden(runner, MoeRunner, ("run",))
        routed = fit.fused_experts_none_to_flashinfer_trtllm_routed
        if runner.lora_enabled or runner.fused_func is not routed:
            reasons.append(
                "the runner does not run fused_experts_none_to_flashinfer_trtllm_routed"
            )
        if runner.config.activation not in FP4_ACTIVATIONS:
            reasons.append(f"the activation {runner.config.activation!r}")
    hidden = experts.hidden_size
    if (
        experts.w13_weight.shape[-1] * 2 != hidden
        or experts.w2_weight.shape[-2] != hidden
    ):
        reasons.append(
            f"the experts' weights {tuple(experts.w13_weight.shape)} / "
            f"{tuple(experts.w2_weight.shape)} are padded past hidden size {hidden}"
        )
    return reasons


def config_reasons() -> list[str]:
    """What the slot needs of the server and of FlashInfer ([] when it holds)."""
    from sglang.srt.layers.moe.utils import (
        should_use_flashinfer_cutlass_moe_fp4_allgather,
    )
    from sglang.srt.models.minimax_m3 import get_moe_a2a_backend

    reasons = []
    if not get_moe_a2a_backend().is_none():
        reasons.append(f"the MoE a2a backend is {get_moe_a2a_backend()}")
    if should_use_flashinfer_cutlass_moe_fp4_allgather():
        reasons.append("the dispatcher quantizes to FP4 before an all-gather")
    try:
        from flashinfer import utils as fi_utils
        from flashinfer.quantization.kernels import nvfp4_quantize as nq
    except ImportError as e:
        return reasons + [f"FlashInfer's CuTe-DSL NVFP4 quantize does not import: {e}"]
    names = (
        "_get_compiled_kernel_nvfp4_per_token",
        "SF_LAYOUT_LINEAR",
        "_env_flag_enabled",
        "resolve_nvfp4_4over6",
        "_per_token_cta_threads",
    )
    reasons += [
        f"flashinfer.quantization.kernels.nvfp4_quantize has no {name}"
        for name in names
        if not hasattr(nq, name)
    ]
    if not hasattr(fi_utils, "device_support_pdl"):
        reasons.append("flashinfer.utils has no device_support_pdl")
    return reasons


def _gather(obj) -> list:
    """``obj`` from every TP rank, through the megakernel's startup gather."""
    from sgl_nvidia_mega.minimax_m31.runtime.agree import gather_tp

    return gather_tp(obj)


def build(model, widths: tuple[int, ...]) -> Bound:
    """The slot at every width of ``widths`` when every rank finds the chain on
    every MoE layer of ``model`` and FlashInfer's kernel matches its own call there
    (``PerTokenQuant.verify``), else at none; a collective on the TP group."""
    m = model.model
    layers = [m.layers[i] for i in range(m.start_layer, m.end_layer)]
    moe = [layer for layer in layers if layer.is_layer_sparse]
    quant, reasons = None, []
    try:
        reasons += config_reasons()
        if not moe:
            reasons.append("the model has no MoE layer")
        for layer in moe:
            reasons += [f"layer {layer.layer_id}: {r}" for r in layer_reasons(layer)]
        if not reasons:
            experts = moe[0].mlp.experts
            quant = PerTokenQuant(
                max_rows=max(widths),
                hidden=experts.hidden_size,
                device=experts.w13_weight.device,
            )
            reasons += quant.verify()
    except Exception as e:  # gathered, then the slot is off on every rank
        reasons.append(f"{type(e).__name__}: {e}")
    failed = [
        f"rank {rank}: {why}"
        for rank, rank_reasons in enumerate(_gather(reasons))
        for why in rank_reasons
    ]
    if failed:
        logger.warning(
            "NVIDIA megakernel: slot moe_nofill serves no width:\n  - %s",
            "\n  - ".join(failed),
        )
        return Bound(reason="; ".join(failed))
    logger.info(
        "NVIDIA megakernel: slot moe_nofill on %d MoE layers at %d-%d rows: the "
        "routed experts' NVFP4 input quantized into %.1f KiB built once",
        len(moe),
        min(widths),
        max(widths),
        quant.nbytes() / 1024,
    )
    return Bound(
        widths=tuple(widths),
        quant=quant,
        mlps={layer.layer_id: layer.mlp for layer in moe},
    )


def fallback_reason(bound: Optional[Bound], layer, x) -> str:
    """Why this step runs SGLang's module for ``layer``'s MoE ("" when the slot runs
    it): host state only, alike on every rank."""
    from sglang.srt.models.minimax_m3 import get_forward, get_moe_a2a_backend

    if bound is None or not bound.widths:
        return "the slot is not built"
    mlp, quant = layer.mlp, bound.quant
    if bound.mlps.get(layer.layer_id) is not mlp:
        return "the layer's MoE is not the module checked at build"
    if why := call_reason(mlp):
        return why
    if not get_moe_a2a_backend().is_none():
        return "the MoE a2a backend is set"
    if not (
        isinstance(x, torch.Tensor)
        and x.dim() == 2
        and x.dtype == DTYPE
        and x.is_contiguous()
        and x.device == quant.device
        and x.shape[1] == quant.hidden
        and 0 < x.shape[0] <= quant.max_rows
    ):
        return "the input is not contiguous bf16 rows the buffers take"
    if not get_forward().defer_moe_finalize:
        return "the step does not defer the MoE finalize"
    return ""


_LOGGED: set = set()


def log_fallback(layer_id: int, why: str) -> None:
    """Log a fallback once per reason (a covered step runs once per captured width)."""
    if why not in _LOGGED:
        _LOGGED.add(why)
        logger.warning(
            "NVIDIA megakernel: slot moe_nofill runs SGLang's MoE at layer %d (and "
            "any other layer for the same reason): %s",
            layer_id,
            why,
        )
