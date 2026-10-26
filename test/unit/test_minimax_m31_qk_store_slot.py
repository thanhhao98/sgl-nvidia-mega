# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of the M3.1 megakernel slot qk_store (the q/k/index norm + RoPE with the
FP8 q epilogue and the NVFP4 K/V/index-K store in one kernel, which SGLang's KV pool
launches): the decomposed attention against SGLang's MiniMaxM3Attention.forward through
SGLang's real RadixAttention, NVFP4 attention backend and KV pool, only the CUDA kernels
faked (the backend skips its store exactly when the fused kernel ran, and the fused
kernel gets the store's arguments); the step reaching the pool only through the
running forward's view (SGLang's binding), with the arguments of the backend's store
call; its composition with attn_ll; slot hop_mxfp8's MXFP8 pair as the input; the
step's fallbacks; the build (SGLang's fused-store switch, the check against SGLang's
kernels on the pool's page-row layout); and the fused kernel's source order (PDL).
The pool's fused store itself (its arguments to the kernel, its per-buffer refusal)
is SGLang's test, with the pool."""

import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.nn import Parameter

pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.attention.minimax_sparse_nvfp4_backend import (  # noqa: E402
    MiniMaxSparseNvfp4AttnBackend,
)
from sglang.srt.layers.linear import RowParallelLinear  # noqa: E402
from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput  # noqa: E402
from sglang.srt.layers.radix_attention import RadixAttention  # noqa: E402

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind  # noqa: E402
    from sglang.srt.kernel_plugins.nvidia.binding import ForwardView  # noqa: E402
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.binding import ForwardView  # noqa: E402
    from sglang.srt.megakernels.spec import NumericsKind  # noqa: E402

from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import (  # noqa: E402
    MiniMaxM3Nvfp4KVPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.models.minimax_m3 import (  # noqa: E402
    MiniMaxM3Attention,
    _FusedQKVIndexProj,
)
from sglang.test.test_utils import CustomTestCase  # noqa: E402

from sgl_nvidia_mega.minimax_m31 import attention as m31_attention  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import plan as m31_plan  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import runtime as m31_runtime  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import spec as m31_spec  # noqa: E402
from sgl_nvidia_mega.minimax_m31.attention import qk_store as m31_qk_store  # noqa: E402
from sgl_nvidia_mega.minimax_m31.runtime import agree as m31_agree  # noqa: E402
from sgl_nvidia_mega.minimax_m31.sm100 import mxfp8_ll  # noqa: E402

E4M3 = torch.float8_e4m3fn
HIDDEN = 256  # the fake projections' input and output width (no check reads it)
QKV_WIDTH = 2560  # one TP4 rank's q|k|v|index q|index k row: 20 heads of 128
PAGE = 128
NUM_LAYERS = 2
POOL_TOKENS = 5 * PAGE


# --------------------------------------------------------------- fakes of the kernels


def _desc(t):
    """A tensor's layout and bytes (or the value itself)."""
    if not isinstance(t, torch.Tensor):
        return t
    raw = t.detach().contiguous().view(-1).view(torch.uint8).clone()
    return ("tensor", tuple(t.shape), tuple(t.stride()), t.dtype, raw)


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.shape == b.shape
            and torch.equal(a, b)
        )
    if isinstance(a, (tuple, list)):
        return (
            isinstance(b, (tuple, list))
            and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b))
        )
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


class _Trace:
    def __init__(self):
        self.calls = []

    def add(self, name, *payload):
        self.calls.append((name, payload))

    def named(self, name):
        return [payload for n, payload in self.calls if n == name]

    def names(self):
        return [n for n, _ in self.calls]


def _codes(x):
    """Stand-in NVFP4 codes of [..., 128] rows: [..., 64] bytes, two values a byte."""
    q = (x.float() * 3).round().clamp(-8, 7).to(torch.int64) & 0xF
    q = q.unflatten(-1, (-1, 2))
    return (q[..., 0] | (q[..., 1] << 4)).to(torch.uint8)


def _scales(x):
    """Stand-in block scales of [..., 128] rows: [..., 8] bytes, one per 16 values."""
    amax = x.float().unflatten(-1, (-1, 16)).abs().amax(-1)
    return (amax * 8).clamp(0, 255).to(torch.uint8)


def _store_rows(
    k,
    v,
    idx_k,
    k_payload,
    k_scale,
    v_payload,
    v_scale,
    idx_k_payload,
    idx_k_scale,
    indices,
    global_scale,
    *,
    page_size,
):
    """nvfp4_store_kv_index's writes (its addressing, stand-in codes): every row,
    padded rows included, at the slot its index names."""
    loc = indices.long()
    for t in range(loc.shape[0]):
        page, row = int(loc[t]) // page_size, int(loc[t]) % page_size
        k_payload[page, :, row] = _codes(k[t])
        k_scale[page, :, row] = _scales(k[t])
        v_payload[page, :, row] = _codes(v[t])
        v_scale[page, :, row] = _scales(v[t])
        idx_k_payload[page, row] = _codes(idx_k[t])
        idx_k_scale[page, row] = _scales(idx_k[t])


def _norm_rope(qkv, groups, cos_sin_cache, positions, eps, fp8_a, a, fp8_b, b):
    """minimax_qknorm_rope_grouped_fp8's writes (stand-in math): every group's heads
    in place, the FP8 copies of groups a and b; V and the other heads untouched."""
    heads = qkv.view(qkv.shape[0], -1, 128)
    rope = cos_sin_cache[positions.long()][:, None, :1]
    for w, off, cnt in groups:
        h = heads[:, off : off + cnt]
        h.copy_((h.float() * (1 + w.float()) * 0.5 + rope + eps).to(h.dtype))
    for out, g in ((fp8_a, a), (fp8_b, b)):
        _, off, cnt = groups[g]
        out.copy_(heads[:, off : off + cnt].to(E4M3))


def _storage_desc(t):
    """A view's place in its storage (offset, shape, stride, dtype)."""
    offset = t.data_ptr() - t.untyped_storage().data_ptr()
    return (offset, tuple(t.shape), tuple(t.stride()), t.dtype)


def _store_args_desc(args):
    """The store kernel's arguments: K / V / index K by their place in the q|k|v|index
    rows (the fused kernel gets them before it norms K and index K, SGLang's store
    after: the stored bytes are compared instead), the cache buffers and the slots by
    address (one pool serves both forwards) and the slots' bytes."""
    desc = {name: _storage_desc(args[name]) for name in ("k", "v", "idx_k")}
    for name in (
        "k_payload",
        "k_scale",
        "v_payload",
        "v_scale",
        "idx_k_payload",
        "idx_k_scale",
        "indices",
        "global_scale",
    ):
        t = args[name]
        desc[name] = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
    desc["indices_bytes"] = _desc(args["indices"])
    desc["page_size"] = args["page_size"]
    return desc


_STORE_NAMES = (
    "k",
    "v",
    "idx_k",
    "k_payload",
    "k_scale",
    "v_payload",
    "v_scale",
    "idx_k_payload",
    "idx_k_scale",
    "indices",
    "global_scale",
)


def _fake_store(trace):
    """nvfp4_store_kv_index, with its signature."""

    def store(
        k,
        v,
        idx_k,
        k_payload,
        k_scale,
        v_payload,
        v_scale,
        idx_k_payload,
        idx_k_scale,
        indices,
        global_scale,
        *,
        page_size,
    ):
        args = (k, v, idx_k, k_payload, k_scale, v_payload, v_scale)
        args += (idx_k_payload, idx_k_scale, indices, global_scale)
        named = dict(zip(_STORE_NAMES, args), page_size=page_size)
        trace.add("store", _store_args_desc(named))
        _store_rows(*args, page_size=page_size)

    return store


def _fake_qknorm(trace):
    def qknorm(
        qkv, groups, cos_sin_cache, positions, eps, *, fp8_a, group_a, fp8_b, group_b
    ):
        trace.add(
            "qknorm",
            _desc(qkv),
            [(id(w), off, cnt) for w, off, cnt in groups],
            id(cos_sin_cache),
            _desc(positions),
            eps,
            group_a,
            group_b,
        )
        _norm_rope(
            qkv, groups, cos_sin_cache, positions, eps, fp8_a, group_a, fp8_b, group_b
        )
        return qkv

    return qknorm


def _fake_fused(trace):
    """minimax_qknorm_rope_store_fp8: its host checks that K and index K are the
    normed groups' heads in place, then the two kernels' writes."""

    def fused(
        qkv,
        groups,
        cos_sin_cache,
        positions,
        eps,
        *,
        fp8_a,
        group_a,
        fp8_b,
        group_b,
        k_group,
        index_k_group,
        **store,
    ):
        page_size = store.pop("page_size")
        args = [store.pop(name) for name in _STORE_NAMES]
        assert not store, store
        named = dict(zip(_STORE_NAMES, args), page_size=page_size)
        trace.add(
            "fused",
            _desc(qkv),
            [(id(w), off, cnt) for w, off, cnt in groups],
            id(cos_sin_cache),
            _desc(positions),
            eps,
            group_a,
            group_b,
            k_group,
            index_k_group,
            _store_args_desc(named),
        )
        base, head_bytes = qkv.data_ptr(), 128 * qkv.element_size()
        k, idx_k = named["k"], named["idx_k"]
        assert k.data_ptr() == base + groups[k_group][1] * head_bytes
        assert idx_k.data_ptr() == base + groups[index_k_group][1] * head_bytes
        # a single row's stride is never used
        assert qkv.shape[0] == 1 or k.stride(0) == idx_k.stride(0) == qkv.stride(0)
        _norm_rope(
            qkv, groups, cos_sin_cache, positions, eps, fp8_a, group_a, fp8_b, group_b
        )
        _store_rows(*args, page_size=page_size)

    return fused


# ------------------------------------------------- SGLang's modules, kernels faked


class _Gemm:
    """A linear method whose apply is x @ weight.T in BF16; x BF16 rows or slot
    hop_mxfp8's MXFP8 pair (its E4M3 rows; both of its tensors traced)."""

    def __init__(self, trace, name, weight):
        self.trace, self.name, self.weight = trace, name, weight

    def apply(self, layer, x, bias=None):
        if isinstance(x, Mxfp8SwizzledInput):
            self.trace.add(self.name, _desc(x.data), _desc(x.scales))
            x = x.data
        else:
            self.trace.add(self.name, _desc(x))
        return (x.float() @ self.weight.float().t()).to(torch.bfloat16)


def _weight(n, k, seed):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(n, k, generator=gen) / k**0.5).to(torch.bfloat16)


def _fused_proj(trace):
    p = _FusedQKVIndexProj.__new__(_FusedQKVIndexProj)
    torch.nn.Module.__init__(p)
    p.weight = Parameter(_weight(QKV_WIDTH, HIDDEN, 1), requires_grad=False)
    # attn_ll's low-latency linear reads these as the MXFP8 linear's operands
    p.weight_scale_inv_swizzled = Parameter(
        torch.zeros(QKV_WIDTH, dtype=torch.uint8), requires_grad=False
    )
    p._qm = _Gemm(trace, "qkv_gemm", p.weight)
    p.input_scale = None
    return p


def _o_proj(trace):
    o = RowParallelLinear.__new__(RowParallelLinear)
    torch.nn.Module.__init__(o)
    o.weight = Parameter(_weight(HIDDEN, 2048, 2), requires_grad=False)
    o.weight_scale_inv_swizzled = Parameter(
        torch.zeros(HIDDEN, dtype=torch.uint8), requires_grad=False
    )
    o.quant_method = _Gemm(trace, "o_gemm", o.weight)
    o.tp_group = SimpleNamespace(rank_in_group=0, world_size=4)
    o.parallel_group = "attn_tp"
    o.input_is_parallel = True
    o.reduce_results = False
    o.skip_bias_add = False
    o.use_dp_attention_reduce = False
    o.use_decode_attn_tp = False
    o.register_parameter("bias", None)
    return o


class _FakeAttention(MiniMaxM3Attention):
    """A MiniMaxM3Attention of the M3.1 TP4 rank (the real class and forward) with
    SGLang's RadixAttention, fake projections where kernels run."""

    def __init__(self, trace, layer_id):
        torch.nn.Module.__init__(self)
        self.is_sparse_attention_layer = True
        self.disable_index_value = True
        self.index_o_proj = None
        self.num_heads, self.num_kv_heads, self.head_dim = 16, 1, 128
        self.num_idx_heads, self.idx_head_dim, self.idx_replica_size = 1, 128, 1
        self.q_size, self.kv_size = 2048, 128
        self._fused_main_size = self.q_size + 2 * self.kv_size
        self._can_use_aiter_fused_qknorm_static = False
        self._combined_qknorm_ok = True
        self._fp8_q_epilogue_ok = True
        self._qknorm_group_meta = ((0, 16), (16, 1), (18, 1), (19, 1))
        gen = torch.Generator().manual_seed(3 + layer_id)
        for name in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
            setattr(
                self,
                name,
                SimpleNamespace(
                    weight=(0.1 * torch.randn(128, generator=gen)).bfloat16(),
                    variance_epsilon=1e-6,
                ),
            )
        self.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.randn(4096, 64, generator=gen)
        )
        self.add_module("fused_qkv_index_proj", _fused_proj(trace))
        self._fused_qkv_index = self.fused_qkv_index_proj
        self.o_proj = _o_proj(trace)
        self.attn = RadixAttention(
            16, 128, 128**-0.5, num_kv_heads=1, layer_id=layer_id
        )


def _layer(trace, layer_id=0):
    return SimpleNamespace(self_attn=_FakeAttention(trace, layer_id))


def _model(trace, num_layers=NUM_LAYERS):
    layers = [_layer(trace, i) for i in range(num_layers)]
    return SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=num_layers, layers=layers)
    )


def _pool():
    """SGLang's M3.1 KV pool (NVFP4 main K/V and index K) on the CPU, every byte
    random, with its fused store on (the GPU's setting)."""
    pool = MiniMaxM3Nvfp4KVPool(
        size=POOL_TOKENS,
        page_size=PAGE,
        dtype=E4M3,
        head_num=1,
        head_dim=128,
        idx_head_dim=128,
        dense_layer_ids=[],
        sparse_layer_ids=list(range(NUM_LAYERS)),
        disable_value_sparse_layer_ids=list(range(NUM_LAYERS)),
        device="cpu",
        start_layer=0,
        end_layer=NUM_LAYERS,
    )
    gen = torch.Generator().manual_seed(5)
    for slab in (pool.main_pool.slab, pool.index_k_pool.slab):
        slab.copy_(torch.randint(0, 256, slab.shape, dtype=torch.uint8, generator=gen))
    pool.use_nvfp4_fused_quantize_store = True
    return pool


def _slabs(pool):
    return (pool.main_pool.slab.clone(), pool.index_k_pool.slab.clone())


def _restore(pool, slabs):
    pool.main_pool.slab.copy_(slabs[0])
    pool.index_k_pool.slab.copy_(slabs[1])


def _backend(trace, pool, rows, tokens_per_request):
    """SGLang's NVFP4 attention backend over ``pool``, its MSA-v1 indexer and
    attention runs faked: both read the caches, so their outputs depend on what the
    step stored."""
    backend = MiniMaxSparseNvfp4AttnBackend.__new__(MiniMaxSparseNvfp4AttnBackend)
    backend.kv_pool = backend.token_to_kv_pool = pool
    backend.forward_metadata = SimpleNamespace(
        msa_v1_decode_plan=SimpleNamespace(
            batch_size=rows // tokens_per_request, query_length=tokens_per_request
        ),
        msa_v1_attention_plan=None,
    )

    def indexer(idx_q, layer):
        trace.add("indexer", layer.layer_id, _desc(idx_q))
        keys = pool.get_index_k_buffer(layer.layer_id).float()
        score = keys.sum() * 1e-6 + idx_q.float().sum(dim=(1, 2))
        return (score[:, None, None].long() + torch.arange(16)) % 16

    def attention(q, topk, layer):
        trace.add("attention", layer.layer_id, _desc(q), _desc(topk))
        k_cache, v_cache = pool.get_kv_buffer(layer.layer_id)
        index = pool.get_index_k_buffer(layer.layer_id)
        cached = (
            k_cache.float().sum() * 1e-5
            + v_cache.float().mean()
            + index.float().mean() * 1e-2
        )
        out = q.float() * 0.25 + topk.float().mean() + cached
        return out.to(torch.bfloat16)

    backend._run_msa_v1_decode_indexer = indexer
    backend._run_msa_v1_decode_attention = attention
    return backend


_TP_GROUP = SimpleNamespace(world_size=4, rank_in_group=0)
_PARALLEL = SimpleNamespace(tp_group=_TP_GROUP, attn_tp_group=_TP_GROUP)


def _fake_mxfp8_quantize(trace):
    def quantize(input, is_sf_swizzled_layout=True, alignment=32, backend="cute-dsl"):
        trace.add("quantize", _desc(input))
        return input.to(E4M3), torch.zeros(4, dtype=torch.uint8)

    return quantize


def _fake_ll_linear(trace):
    def linear(a, a_scale, weight, weight_scale, *, tactic, out=None):
        trace.add("ll_gemm", _desc(a), id(weight), tactic)
        return (a.float() @ weight.float().t()).to(torch.bfloat16)

    return linear


@contextlib.contextmanager
def _sglang_faked(trace, backend):
    """SGLang's CUDA kernels and runtime queries faked alike for every forward."""
    from sglang.kernels.ops.attention import (
        minimax_qknorm_rope,
        minimax_qknorm_rope_store,
    )
    from sglang.kernels.ops.kvcache import nvfp4_store_kv_index
    from sglang.srt import runtime_context
    from sglang.srt.distributed.device_communicators import pynccl_allocator
    from sglang.srt.layers import linear, radix_attention
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.model_executor import forward_context

    def use_symmetric_memory(group, disabled=False):
        trace.add("allocation", id(group), disabled)
        return contextlib.nullcontext()

    def no_backend_read():
        # RadixAttention reaches the backend through its own name (patched below);
        # the slot reaches the pool only through Step.view
        raise AssertionError("the slot read the attention backend")

    forward = SimpleNamespace(sp_active=False)
    with contextlib.ExitStack() as stack:
        for target, name, value in (
            (
                minimax_qknorm_rope,
                "minimax_qknorm_rope_grouped_fp8",
                _fake_qknorm(trace),
            ),
            (
                minimax_qknorm_rope_store,
                "minimax_qknorm_rope_store_fp8",
                _fake_fused(trace),
            ),
            (nvfp4_store_kv_index, "nvfp4_store_kv_index", _fake_store(trace)),
            (radix_attention, "get_attn_backend", lambda: backend),
            (forward_context, "get_attn_backend", no_backend_read),
            (forward_context, "has_forward_context", no_backend_read),
            (fp8_utils, "flashinfer_mxfp8_quantize", _fake_mxfp8_quantize(trace)),
            (mxfp8_ll, "linear", _fake_ll_linear(trace)),
            (linear, "get_parallel", lambda: _PARALLEL),
            (runtime_context, "get_parallel", lambda: _PARALLEL),
            (linear, "get_forward", lambda: forward),
            (runtime_context, "get_forward", lambda: forward),
            (linear, "use_symmetric_memory", use_symmetric_memory),
            (pynccl_allocator, "use_symmetric_memory", use_symmetric_memory),
        ):
            stack.enter_context(patch.object(target, name, value, create=True))
        yield


def _bound(model, rows):
    """qk_store's build result binding every layer of ``model`` at ``rows``."""
    layers = model.model.layers
    return m31_qk_store.Bound(
        widths=(rows,), attns={i: layer.self_attn for i, layer in enumerate(layers)}
    )


def _ll_bound(model, rows, tactic=(128, 8, 1, 4), gemms=m31_attention.DEFAULT_GEMMS):
    """attn_ll's build result over the same layers (any tactic: the GEMM is faked),
    with ``gemms`` on the low-latency kernel: the slot's DEFAULT_GEMMS, as it builds,
    unless a test names others (as ``attention.GEMMS_ENV`` would)."""
    layers = {}
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        linears = {"qkv_index": attn.fused_qkv_index_proj, "o_proj": attn.o_proj}
        layers[i] = m31_attention.LayerGemms(
            **{
                name: m31_attention.LowLatencyLinear(linears[name], {rows: tactic})
                for name in gemms
            }
        )
    return m31_attention.AttnLowLatency(
        layers=layers,
        widths=(rows,),
        tactics={},
        reasons=[],
        skipped={},
        gemms=tuple(gemms),
    )


def _inputs(rows, seed=0):
    gen = torch.Generator().manual_seed(seed + rows)
    return (torch.randn(rows, HIDDEN, generator=gen) * 2).to(torch.bfloat16)


def _pair(rows, seed=0):
    """Slot hop_mxfp8's attention input: the rows in E4M3 and one 128-row atom of
    UE8M0 scales in FlashInfer's 128x4 layout (what attn_ll's GEMM takes)."""
    scales = torch.arange(mxfp8_ll.scale_bytes(rows, HIDDEN)) % 5 + 125
    return Mxfp8SwizzledInput(_inputs(rows, seed).to(E4M3), scales.to(torch.uint8))


def _forward_batch(rows, tokens_per_request, *, padded=0):
    """A verify (or decode) step's forward batch: each request's rows on consecutive
    cache slots, the first request's crossing a page boundary; ``padded`` trailing
    rows at slot 0, as a graph pads them."""
    gen = torch.Generator().manual_seed(rows)
    # pages 3-5 for the requests, the first one across pages 1 and 2
    starts = torch.randperm(3 * PAGE // tokens_per_request, generator=gen)
    starts = starts[: rows // tokens_per_request] * tokens_per_request + 3 * PAGE
    starts[0] = 2 * PAGE - tokens_per_request // 2
    loc = (starts[:, None] + torch.arange(tokens_per_request)).reshape(-1)
    if padded:
        loc[-padded:] = 0
    mode = ForwardMode.TARGET_VERIFY if tokens_per_request > 1 else ForwardMode.DECODE
    return SimpleNamespace(
        forward_mode=mode,
        out_cache_loc=loc.to(torch.int64),
        global_num_token_non_padded_cpu=None,
    )


def _positions(rows, padded=0):
    pos = torch.arange(rows, dtype=torch.long) * 37 % 4000 + 50
    if padded:
        pos[-padded:] = 0
    return pos


def _state(fb, positions, layer_id, *, slots, bound, view=None):
    """A step's state; ``view``, the running forward's ForwardView (Step.view)."""
    return SimpleNamespace(
        slots=slots,
        bound=bound,
        layer_id=layer_id,
        rows=int(positions.shape[0]),
        positions=positions,
        fb=fb,
        view=view,
    )


# ------------------------------------------------------------------------- tests


class TestNumerics(CustomTestCase):
    def test_qk_store_is_an_opt_in_bitwise_slot(self):
        self.assertIn("qk_store", m31_runtime.OPT_IN_SLOTS)
        self.assertNotIn("qk_store", m31_runtime.SHIPPED_SLOTS)
        self.assertIn("qk_store", m31_runtime.BITWISE_SLOTS)
        self.assertNotIn("qk_store", m31_runtime.TOLERANCES)
        self.assertEqual(m31_runtime.slot_reasons(frozenset({"qk_store"})), [])
        shipped = frozenset(m31_runtime.SHIPPED_SLOTS)
        with patch.object(m31_spec, "SLOTS", shipped | {"qk_store"}):
            self.assertEqual(m31_spec._numerics().kind, NumericsKind.BITWISE)
        # with attn_ll the tolerances are attn_ll's alone
        with patch.object(m31_spec, "SLOTS", frozenset({"qk_store", "attn_ll"})):
            numerics = m31_spec._numerics()
        self.assertEqual(numerics.kind, NumericsKind.TOLERANCE)
        self.assertEqual(
            dict(numerics.tolerances), m31_runtime.tolerances(frozenset({"attn_ll"}))
        )


class TestStepMatchesSGLangsAttention(CustomTestCase):
    """The slot's step against MiniMaxM3Attention.forward on the same layer and the
    same KV pool: SGLang's RadixAttention, NVFP4 backend and pool run for real."""

    def _step(self, rows, tokens, padded=0, mxfp8=False):
        """One step's forward batch, positions and attention input (BF16 rows, or
        with ``mxfp8`` slot hop_mxfp8's MXFP8 pair)."""
        return (
            _forward_batch(rows, tokens, padded=padded),
            _positions(rows, padded),
            _pair(rows) if mxfp8 else _inputs(rows),
        )

    def _run(self, model, pool, step, tokens, *, slots, bound=None):
        """Both layers of ``model`` on ``step``, one after the other (the stock module
        call when ``slots`` is None, else the executor's attention step): the trace,
        the outputs, and the pool's bytes after."""
        fb, positions, x = step
        rows = positions.shape[0]
        trace = _Trace()
        backend = _backend(trace, pool, rows, tokens)
        outs = []
        with _sglang_faked(trace, backend):
            for layer_id, layer in enumerate(model.model.layers):
                for module in (
                    layer.self_attn.fused_qkv_index_proj._qm,
                    layer.self_attn.o_proj.quant_method,
                ):
                    module.trace = trace
                if slots is None:
                    out = layer.self_attn(
                        positions=positions, hidden_states=x, forward_batch=fb
                    )
                else:
                    st = _state(
                        fb,
                        positions,
                        layer_id,
                        slots=slots,
                        bound=bound,
                        view=ForwardView(backend),
                    )
                    out = m31_attention.step(layer, x, st)
                outs.append(out)
        return trace, outs, _slabs(pool)

    def _check(self, rows, tokens, padded=0, mxfp8=False):
        model, pool = _model(_Trace()), _pool()
        step = self._step(rows, tokens, padded, mxfp8)
        before = _slabs(pool)
        stock, want, want_bytes = self._run(model, pool, step, tokens, slots=None)
        _restore(pool, before)
        slot, got, got_bytes = self._run(
            model,
            pool,
            step,
            tokens,
            slots=m31_plan.Slots(qk_store=True),
            bound={"qk_store": _bound(model, rows)},
        )
        what = f"{rows} rows"
        for a, b in zip(got, want):
            self.assertTrue(torch.equal(a, b), what)
        for a, b in zip(got_bytes, want_bytes):
            self.assertTrue(torch.equal(a, b), what)
        self.assertFalse(torch.equal(want_bytes[0], before[0]), what)
        per_layer = [
            "qkv_gemm",
            "store",
            "indexer",
            "attention",
            "allocation",
            "o_gemm",
        ]
        self.assertEqual(
            [n for n in stock.names() if n != "qknorm"], per_layer * NUM_LAYERS
        )
        # the fused kernel runs where SGLang's q/k norm does, and the backend's store
        # (exactly once a layer in SGLang's forward) never runs
        self.assertEqual(slot.names().count("store"), 0)
        self.assertEqual(stock.names().count("store"), NUM_LAYERS)
        self.assertEqual(
            [n if n != "fused" else "qknorm" for n in slot.names()],
            [n for n in stock.names() if n != "store"],
        )
        for qknorm, store, fused in zip(
            stock.named("qknorm"), stock.named("store"), slot.named("fused")
        ):
            # SGLang's q/k norm call, the group ids of the K / index-K heads, and
            # the store's arguments, call for call
            self.assertTrue(_same(fused[:7], qknorm), what)
            self.assertEqual(fused[7:9], (1, 3))
            self.assertTrue(_same(fused[9], store[0]), what)
        for name in ("qkv_gemm", "indexer", "attention", "o_gemm"):
            self.assertTrue(_same(slot.named(name), stock.named(name)), name)

    def test_verify_steps(self):
        for rows in (8, 16, 64, 128):
            self._check(rows, 8)

    def test_decode_steps(self):
        for rows in (1, 5, 16):
            self._check(rows, 1)

    def test_padded_rows_store_where_sglang_stores_them(self):
        self._check(32, 8, padded=8)
        self._check(16, 1, padded=3)

    def test_mxfp8_input_from_hop_mxfp8(self):
        """The default composition (the shipped slots and qk_store): layers 1-59 get
        slot hop_mxfp8's MXFP8 pair, which SGLang's q|k|v|index linear takes as it is,
        in the slot's step as in SGLang's forward."""
        for rows, tokens in ((8, 8), (128, 8), (1, 1), (16, 1)):
            self._check(rows, tokens, mxfp8=True)

    def test_with_attn_ll_the_low_latency_gemms_run_around_the_fused_kernel(self):
        """The two slots compose: attn_ll + qk_store against attn_ll alone, the same
        GEMMs, outputs and stored bytes; the q/k norm and the backend's store replaced
        by the fused kernel. Here attn_ll runs o_proj on the low-latency kernel too, as
        ``attention.GEMMS_ENV=qkv_index,o_proj`` builds it; its default build (the
        q|k|v|index GEMM alone) is checked by
        test_with_attn_ll_default_its_qkv_gemm_runs_before_the_fused_kernel."""
        rows, tokens = 8, 8
        both_gemms = ("qkv_index", "o_proj")
        model, pool = _model(_Trace()), _pool()
        before = _slabs(pool)
        # BF16 rows, and slot hop_mxfp8's pair (attn_ll's GEMM takes it unquantized)
        for mxfp8 in (False, True):
            step = self._step(rows, tokens, mxfp8=mxfp8)
            runs = {}
            for name, slots in (
                ("attn_ll", m31_plan.Slots(attn_ll=True)),
                ("both", m31_plan.Slots(attn_ll=True, qk_store=True)),
            ):
                _restore(pool, before)
                bound = {
                    "attention": _ll_bound(model, rows, gemms=both_gemms),
                    "qk_store": _bound(model, rows),
                }
                runs[name] = self._run(
                    model, pool, step, tokens, slots=slots, bound=bound
                )
            (ll, want, want_bytes) = runs["attn_ll"]
            (both, got, got_bytes) = runs["both"]
            for a, b in zip(got + list(got_bytes), want + list(want_bytes)):
                self.assertTrue(torch.equal(a, b), mxfp8)
            self.assertNotIn("qkv_gemm", both.names())
            self.assertEqual(both.names().count("ll_gemm"), 2 * NUM_LAYERS)
            self.assertEqual(both.names().count("fused"), NUM_LAYERS)
            self.assertEqual(both.names().count("store"), 0)
            # the pair skips the q|k|v|index input's quantize; o_proj's remains
            self.assertEqual(
                both.names().count("quantize"), NUM_LAYERS * (1 if mxfp8 else 2)
            )
            self.assertEqual(
                [n if n != "fused" else "qknorm" for n in both.names()],
                [n for n in ll.names() if n != "store"],
            )
            for name in ("ll_gemm", "quantize", "indexer", "attention"):
                self.assertTrue(_same(both.named(name), ll.named(name)), name)

    def test_with_attn_ll_default_its_qkv_gemm_runs_before_the_fused_kernel(self):
        """attn_ll's default build (the q|k|v|index GEMM on the low-latency kernel,
        o_proj SGLang's) with qk_store, against attn_ll alone: the same GEMMs, outputs
        and stored bytes. On slot hop_mxfp8's pair that GEMM is still the layer's
        first kernel, as attention.low_latency_qkv tells the boundary kernel before
        it signals the GEMM at its entry."""
        rows, tokens = 8, 8
        model, pool = _model(_Trace()), _pool()
        before = _slabs(pool)
        both_slots = m31_plan.Slots(attn_ll=True, qk_store=True)

        def bound():
            return {
                "attention": _ll_bound(model, rows),
                "qk_store": _bound(model, rows),
            }

        for mxfp8 in (False, True):
            step = self._step(rows, tokens, mxfp8=mxfp8)
            runs = {}
            for name, slots in (
                ("attn_ll", m31_plan.Slots(attn_ll=True)),
                ("both", both_slots),
            ):
                _restore(pool, before)
                runs[name] = self._run(
                    model, pool, step, tokens, slots=slots, bound=bound()
                )
            (ll, want, want_bytes) = runs["attn_ll"]
            (both, got, got_bytes) = runs["both"]
            for a, b in zip(got + list(got_bytes), want + list(want_bytes)):
                self.assertTrue(torch.equal(a, b), mxfp8)
            self.assertEqual(both.names().count("ll_gemm"), NUM_LAYERS)
            self.assertEqual(both.names().count("o_gemm"), NUM_LAYERS)
            self.assertNotIn("qkv_gemm", both.names())
            self.assertEqual(both.names().count("fused"), NUM_LAYERS)
            self.assertEqual(both.names().count("store"), 0)
            self.assertEqual(
                [n if n != "fused" else "qknorm" for n in both.names()],
                [n for n in ll.names() if n != "store"],
            )
        # the boundary's question on the pair, then the step it asks about
        fb, positions, x = self._step(rows, tokens, mxfp8=True)
        layer = model.model.layers[0]
        trace = _Trace()
        backend = _backend(trace, pool, rows, tokens)
        st = _state(
            fb, positions, 0, slots=both_slots, bound=bound(), view=ForwardView(backend)
        )
        with _sglang_faked(trace, backend):
            for module in (
                layer.self_attn.fused_qkv_index_proj._qm,
                layer.self_attn.o_proj.quant_method,
            ):
                module.trace = trace
            self.assertTrue(m31_attention.low_latency_qkv(layer, x, st))
            m31_attention.step(layer, x, st)
        self.assertEqual(trace.names()[:2], ["ll_gemm", "fused"])

    def test_slot_off_runs_the_stock_module_call(self):
        rows, tokens = 8, 8
        model, pool = _model(_Trace()), _pool()
        step = self._step(rows, tokens)
        before = _slabs(pool)
        want = self._run(model, pool, step, tokens, slots=None)
        _restore(pool, before)
        got = self._run(
            model,
            pool,
            step,
            tokens,
            slots=m31_plan.Slots(),
            bound={"qk_store": _bound(model, rows)},
        )
        self.assertTrue(_same(got[0].calls, want[0].calls))
        self.assertTrue(_same(list(got[1:]), list(want[1:])))


class TestFallbacks(CustomTestCase):
    """A step the slot does not reproduce runs SGLang's two kernels for the layer
    (no fused launch), with the same result."""

    def _run_case(self, change, rows=8, tokens=8):
        model, pool = _model(_Trace()), _pool()
        layer = model.model.layers[0]
        trace = _Trace()
        backend = _backend(trace, pool, rows, tokens)
        fb = _forward_batch(rows, tokens)
        positions = _positions(rows)
        bound = _bound(model, rows)
        st = _state(
            fb,
            positions,
            0,
            slots=m31_plan.Slots(qk_store=True),
            bound={"qk_store": bound},
            view=ForwardView(backend),
        )
        change(layer, st, backend, pool, bound)
        for module in (
            layer.self_attn.fused_qkv_index_proj._qm,
            layer.self_attn.o_proj.quant_method,
        ):
            module.trace = trace
        with (
            _sglang_faked(trace, backend),
            patch.object(
                MiniMaxM3Nvfp4KVPool,
                "set_kv_buffer",
                lambda self, *a, **k: trace.add("per_buffer_kv"),
            ),
            patch.object(
                MiniMaxM3Nvfp4KVPool,
                "set_index_k_buffer",
                lambda self, *a, **k: trace.add("per_buffer_index_k"),
            ),
        ):
            m31_attention.step(layer, _inputs(rows), st)
        return trace

    def test_declines(self):
        def unbound(layer, st, backend, pool, bound):
            bound.attns.pop(0)

        def attention_hook(layer, st, backend, pool, bound):
            layer.self_attn.register_forward_hook(lambda *a: None)

        def extend_step(layer, st, backend, pool, bound):
            st.fb.forward_mode = ForwardMode.EXTEND

        def int32_positions(layer, st, backend, pool, bound):
            st.positions = st.positions.int()

        def no_forward_view(layer, st, backend, pool, bound):
            st.view = None

        def a_pool_without_the_fused_store(layer, st, backend, pool, bound):
            st.view = SimpleNamespace(kv_pool=SimpleNamespace())

        def per_buffer_pool(layer, st, backend, pool, bound):
            # the pool's fused store declines, launching nothing
            pool.use_nvfp4_fused_quantize_store = False

        for case in (
            unbound,
            attention_hook,
            extend_step,
            int32_positions,
            no_forward_view,
            a_pool_without_the_fused_store,
            per_buffer_pool,
        ):
            trace = self._run_case(case)
            names = trace.names()
            self.assertNotIn("fused", names, case.__name__)
            self.assertEqual(names.count("qknorm"), 1, case.__name__)
            stores = names.count("store") + names.count("per_buffer_kv")
            self.assertEqual(stores, 1, case.__name__)

    def test_rows_the_pool_stores_per_buffer_decline_after_the_gemm(self):
        """K / V the pool would not take for its fused store (here FP16 rows): the
        slot launches nothing and SGLang's q/k norm and per-buffer store run."""

        def fp16_gemm(layer, st, backend, pool, bound):
            qm = layer.self_attn.fused_qkv_index_proj._qm
            apply = qm.apply
            qm.apply = lambda layer, x, bias=None: apply(layer, x, bias).half()

        trace = self._run_case(fp16_gemm)
        self.assertEqual(
            [n for n in trace.names() if n in ("qkv_gemm", "qknorm", "per_buffer_kv")],
            ["qkv_gemm", "qknorm", "per_buffer_kv"],
        )
        self.assertNotIn("fused", trace.names())


class _FakeKernels:
    def __init__(self, reasons=(), mismatch=()):
        self._reasons, self._mismatch = list(reasons), list(mismatch)
        self.verified = []

    def reasons(self):
        return list(self._reasons)

    def verify(self, attn, rows):
        self.verified.append((attn, list(rows)))
        return list(self._mismatch)


def _one_rank():
    return patch.object(m31_agree, "gather_tp", side_effect=lambda x, group=None: [x])


class TestBuild(CustomTestCase):
    def test_not_requested(self):
        self.assertIsNone(m31_qk_store.build(None, (8,), frozenset()))
        self.assertFalse(m31_qk_store.serves(None, 8))

    def test_binds_the_layers_on_the_branch_and_checks_the_kernel_once(self):
        model = _model(_Trace(), num_layers=6)
        layers = model.model.layers

        def groups_moved(attn):
            attn._qknorm_group_meta = ((0, 16), (17, 1), (18, 1), (19, 1))

        def another_attention(attn):
            attn.attn = type("Other", (RadixAttention,), {})(
                16, 128, 1.0, num_kv_heads=1, layer_id=0
            )

        def fp8_q_off(attn):
            attn._fp8_q_epilogue_ok = False

        def overridden(attn):
            attn.forward_core = lambda s: None

        def index_value(attn):
            attn.disable_index_value = False

        changes = (groups_moved, another_attention, fp8_q_off, overridden, index_value)
        for i, change in enumerate(changes):
            change(layers[i].self_attn)
            self.assertTrue(m31_qk_store.layer_reasons(layers[i]), change.__name__)
        clean = len(changes)
        self.assertEqual(m31_qk_store.layer_reasons(layers[clean]), [])
        kernels = _FakeKernels()
        with _one_rank():
            built = m31_qk_store.build_qk_store(model, tuple(range(8, 129, 8)), kernels)
        self.assertEqual(built.widths, tuple(range(8, 129, 8)))
        self.assertEqual(list(built.attns), [clean])
        self.assertIs(built.attns[clean], layers[clean].self_attn)
        self.assertEqual(sorted(built.skipped), list(range(clean)))
        self.assertEqual(kernels.verified, [(layers[clean].self_attn, [1, 8, 128])])
        self.assertTrue(m31_qk_store.serves(built, 64))
        self.assertFalse(m31_qk_store.serves(built, 4))

    def test_a_failed_check_or_an_unsupported_gpu_serves_no_width_on_any_rank(self):
        model = _model(_Trace())
        for kernels in (
            _FakeKernels(mismatch=["at 8 rows the fused kernel's qkv differs"]),
            _FakeKernels(reasons=["not SM100"]),
        ):
            with _one_rank():
                built = m31_qk_store.build_qk_store(model, (1, 2, 16), kernels)
            self.assertEqual(built.widths, ())
            self.assertIn("rank 0", built.reason)
        # another rank's failure turns the slot off here too
        with patch.object(
            m31_agree,
            "gather_tp",
            side_effect=lambda x, group=None: [x, (["rank 1 failed"], x[1])],
        ):
            built = m31_qk_store.build_qk_store(model, (8,), _FakeKernels())
        self.assertEqual(built.widths, ())
        self.assertIn("rank 1: rank 1 failed", built.reason)
        with patch.object(
            m31_agree,
            "gather_tp",
            side_effect=lambda x, group=None: [x, ([], [0])],
        ):
            built = m31_qk_store.build_qk_store(model, (8,), _FakeKernels())
        self.assertEqual(built.widths, ())
        self.assertIn("different layers", built.reason)

    def test_the_plan_runs_the_slot_at_every_width_or_refuses_the_server(self):
        model = _model(_Trace())
        ctx = SimpleNamespace(
            widths=tuple(range(8, 129, 8)), tokens_per_request=8, binding=None
        )
        with (
            patch.object(m31_runtime, "SLOTS", frozenset({"qk_store"})),
            patch.object(m31_qk_store, "_GpuKernels", _FakeKernels),
            _one_rank(),
        ):
            built = m31_runtime.build(model, ctx)
        for w in range(8, 129, 8):
            self.assertEqual(built.slot_plan[w], m31_plan.Slots(qk_store=True))
        self.assertIs(built.bound["qk_store"].attns[0], model.model.layers[0].self_attn)
        self.assertIn("qk_store at [8, 16,", built.summary())
        with (
            patch.object(m31_runtime, "SLOTS", frozenset({"qk_store"})),
            patch.object(
                m31_qk_store, "_GpuKernels", lambda: _FakeKernels(reasons=["no"])
            ),
            _one_rank(),
            self.assertRaisesRegex(RuntimeError, "qk_store"),
        ):
            m31_runtime.build(model, ctx)

    def test_sglangs_per_buffer_store_serves_no_width_and_refuses_the_server(self):
        """With SGLang's fused NVFP4 store switched off, the pool stores per buffer and
        the slot would replace no store at any step: it serves no width (and launches
        no check), so a request for it refuses the server at build."""
        switch = envs.SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE
        model = _model(_Trace())
        kernels = _FakeKernels()
        with switch.override(False), _one_rank():
            self.assertTrue(m31_qk_store.store_reasons())
            built = m31_qk_store.build_qk_store(model, (8, 16), kernels)
        self.assertEqual(built.widths, ())
        self.assertIn(switch.name, built.reason)
        self.assertEqual(kernels.verified, [])
        ctx = SimpleNamespace(widths=(8, 16), tokens_per_request=8, binding=None)
        with (
            switch.override(False),
            patch.object(m31_runtime, "SLOTS", frozenset({"qk_store"})),
            patch.object(m31_qk_store, "_GpuKernels", _FakeKernels),
            _one_rank(),
            self.assertRaisesRegex(RuntimeError, "qk_store"),
        ):
            m31_runtime.build(model, ctx)
        # switched on (its default) the same model is served
        with switch.override(True), _one_rank():
            self.assertEqual(m31_qk_store.store_reasons(), [])
            built = m31_qk_store.build_qk_store(model, (8, 16), _FakeKernels())
        self.assertEqual(built.widths, (8, 16))


def _fused_paging_with_the_head_stride(trace):
    """A fused kernel that steps the main K / V caches' pages by their head stride:
    the same bytes where the two strides agree (dense caches of one KV head), others
    in the pool's page rows, whose page stride is the row width."""
    fused = _fake_fused(trace)

    def mixed_up(*args, **kw):
        for name in ("k_payload", "k_scale", "v_payload", "v_scale"):
            t = kw[name]
            kw[name] = t.as_strided(t.shape, (t.stride(1), *t.stride()[1:]))
        fused(*args, **kw)

    return mixed_up


class TestBuildCheck(CustomTestCase):
    """The build's check (verify_against_stock) with the CUDA kernels faked: both arms
    get caches laid out as SGLang's pool lays out a layer (its page rows, so the page
    stride is not the head stride), and any byte the fused kernel writes otherwise
    is reported."""

    def _verify(self, fused_of):
        """The check at 1, 8 and 128 rows, the fused kernel ``fused_of(trace)``: the
        trace and the check's reasons."""
        from sglang.kernels.ops.attention import (
            minimax_qknorm_rope,
            minimax_qknorm_rope_store,
        )
        from sglang.kernels.ops.kvcache import nvfp4_store_kv_index

        trace = _Trace()
        attn = _FakeAttention(trace, 0)
        with contextlib.ExitStack() as stack:
            for target, name, value in (
                (
                    minimax_qknorm_rope,
                    "minimax_qknorm_rope_grouped_fp8",
                    _fake_qknorm(trace),
                ),
                (
                    minimax_qknorm_rope_store,
                    "minimax_qknorm_rope_store_fp8",
                    fused_of(trace),
                ),
                (nvfp4_store_kv_index, "nvfp4_store_kv_index", _fake_store(trace)),
            ):
                stack.enter_context(patch.object(target, name, value))
            reasons = m31_qk_store.verify_against_stock(attn, (1, 8, 128))
        return trace, reasons

    def test_the_caches_are_the_pools_layer_views(self):
        trace, reasons = self._verify(_fake_fused)
        self.assertEqual(reasons, [])
        stores = [args[0] for args in trace.named("store")]
        fused = [args[-1] for args in trace.named("fused")]
        self.assertEqual(len(stores), 3)
        self.assertEqual(len(fused), 3)
        pool = _pool()
        main = pool.main_pool.get_raw_kv_buffers(0)
        index = pool.index_k_pool.get_raw_k_buffers(0)
        want = dict(zip(("k_payload", "v_payload", "k_scale", "v_scale"), main))
        want |= dict(zip(("idx_k_payload", "idx_k_scale"), index))
        for desc in stores + fused:
            for name, view in want.items():
                _, shape, stride, dtype = desc[name]
                self.assertEqual(
                    (shape[1:], stride, dtype),
                    (tuple(view.shape[1:]), tuple(view.stride()), view.dtype),
                    name,
                )
        # the layout this check needs: a main cache's page stride is its row width,
        # not its head stride (equal in dense caches of the one KV head of TP4)
        self.assertNotEqual(want["k_payload"].stride(0), want["k_payload"].stride(1))
        self.assertNotEqual(want["k_scale"].stride(0), want["k_scale"].stride(1))

    def test_reports_bytes_written_elsewhere(self):
        _, reasons = self._verify(_fused_paging_with_the_head_stride)
        self.assertTrue(reasons)
        self.assertTrue(all("main K / V page rows" in why for why in reasons))


class _PoolView:
    """The running forward's KvPoolView as the step sees it: its fused store records
    each call and answers ``stored`` (launching nothing)."""

    def __init__(self, stored):
        self.stored, self.calls = stored, []

    def set_kv_index_buffer_fused_qknorm_rope(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.stored


class TestTheStepReachesThePoolThroughTheView(CustomTestCase):
    """The step reaches SGLang's KV pool only through the running forward's view
    (Step.view.kv_pool, SGLang's binding; _sglang_faked fails any read of the
    attention backend): it hands the pool's fused store the backend's store call (the
    layer, its cache slots, the K / V / index-K views and their scales) with the q/k
    norm's arguments, and where that store declines, SGLang's q/k norm runs and the
    backend stores."""

    def _run(self, stored, rows=8, tokens=8):
        model, pool = _model(_Trace()), _pool()
        layer = model.model.layers[0]
        fb, positions, x = (
            _forward_batch(rows, tokens),
            _positions(rows),
            _inputs(rows),
        )
        trace = _Trace()
        backend = _backend(trace, pool, rows, tokens)
        view = SimpleNamespace(kv_pool=_PoolView(stored))
        st = _state(
            fb,
            positions,
            0,
            slots=m31_plan.Slots(qk_store=True),
            bound={"qk_store": _bound(model, rows)},
            view=view,
        )
        for module in (
            layer.self_attn.fused_qkv_index_proj._qm,
            layer.self_attn.o_proj.quant_method,
        ):
            module.trace = trace
        with _sglang_faked(trace, backend):
            m31_attention.step(layer, x, st)
        return SimpleNamespace(
            attn=layer.self_attn,
            fb=fb,
            positions=positions,
            trace=trace,
            calls=view.kv_pool.calls,
        )

    def test_the_pools_fused_store_gets_the_backends_store_call(self):
        from sglang.srt.models.minimax_m3 import (
            _QKNORM_GROUP_INDEX_Q,
            _QKNORM_GROUP_MAIN_Q,
        )

        rows = 8
        run = self._run(stored=True, rows=rows)
        attn = run.attn
        ((args, kwargs),) = run.calls
        layer, loc, qkv, groups, cache, positions, eps = args
        self.assertIs(layer, attn.attn)
        self.assertIs(loc, run.fb.out_cache_loc)
        self.assertEqual(tuple(qkv.shape), (rows, QKV_WIDTH))
        self.assertEqual(
            [(id(w), off, cnt) for w, off, cnt in groups],
            [(id(w), off, cnt) for w, off, cnt in attn._qknorm_groups()],
        )
        self.assertIs(cache, attn.rotary_emb.cos_sin_cache)
        self.assertIs(positions, run.positions)
        self.assertEqual(eps, attn.q_norm.variance_epsilon)
        self.assertEqual(
            (kwargs["group_a"], kwargs["group_b"]),
            (_QKNORM_GROUP_MAIN_Q, _QKNORM_GROUP_INDEX_Q),
        )
        self.assertEqual(
            (kwargs["k_group"], kwargs["index_k_group"]),
            (m31_qk_store.GROUP_MAIN_K, m31_qk_store.GROUP_INDEX_K),
        )
        e4m3 = torch.float8_e4m3fn
        for name, shape in (
            ("fp8_a", (rows, attn.num_heads, attn.head_dim)),
            ("fp8_b", (rows, attn.num_idx_heads, attn.idx_head_dim)),
        ):
            self.assertEqual(
                (tuple(kwargs[name].shape), kwargs[name].dtype), (shape, e4m3)
            )
        # the K / V / index-K views RadixAttention.forward hands the backend's store,
        # in place in the q|k|v|index rows
        radix = attn.attn
        size = qkv.element_size()
        for name, offset, heads, head_dim in (
            ("cache_k", attn.q_size, radix.tp_k_head_num, radix.qk_head_dim),
            (
                "cache_v",
                attn.q_size + attn.kv_size,
                radix.tp_v_head_num,
                radix.v_head_dim,
            ),
            (
                "cache_idx_k",
                attn._fused_main_size + attn.num_idx_heads * attn.idx_head_dim,
                1,
                attn.idx_head_dim,
            ),
        ):
            view = kwargs[name]
            self.assertEqual(tuple(view.shape), (rows, heads, head_dim), name)
            self.assertEqual(view.data_ptr(), qkv.data_ptr() + offset * size, name)
        self.assertEqual(
            (kwargs["k_scale"], kwargs["v_scale"], kwargs["idx_k_scale"]),
            (radix.k_scale_float, radix.v_scale_float, radix.idx_k_scale_float),
        )
        # the backend ran without its store, and SGLang's q/k norm did not run
        names = run.trace.names()
        self.assertEqual(names.count("indexer"), 1)
        self.assertNotIn("store", names)
        self.assertNotIn("qknorm", names)

    def test_a_declined_fused_store_runs_sglangs_norm_and_the_backend_stores(self):
        run = self._run(stored=False)
        self.assertEqual(len(run.calls), 1)
        names = run.trace.names()
        self.assertEqual(names.count("qknorm"), 1)
        self.assertEqual(names.count("store"), 1)
        self.assertNotIn("fused", names)
        self.assertLess(names.index("qknorm"), names.index("store"))


class TestFusedKernelSource(CustomTestCase):
    """The fused kernel runs the q/k norm kernel's code first (its PDL wait inside)
    and reads the store's inputs only after it: slot attn_ll's GEMM lets it start
    before the GEMM's own predecessor finishes (attention.py). It reuses both stock
    kernels' device code rather than rewriting the math."""

    def test_order(self):
        import sglang

        src = (
            Path(sglang.__file__).parent
            / "kernels/jit/csrc/minimax/fused_qknorm_rope_store.cuh"
        ).read_text()
        self.assertIn('#include "fused_gemma_qknorm_rope.cuh"', src)
        self.assertIn('#include "nvfp4_store_kv_index.cuh"', src)
        self.assertIn("minimax_nvfp4::quantize_block(", src)
        self.assertNotIn("PDLWaitPrimary", src)
        self.assertNotIn("PDLTriggerSecondary", src)
        forward = src[src.index("SGL_DEVICE static void forward(") :]
        forward = forward[: forward.index("\n  }\n")]
        norm = forward.index("Norm::forward(params.norm);")
        self.assertNotIn("params.", forward[:norm].split("{", 1)[1])
        sync = forward.index("__syncwarp();")
        self.assertLess(norm, sync)
        self.assertLess(sync, forward.index("store_block<T>("))
        # the K / index-K rows this kernel wrote are read back with coherent loads
        store = src[src.index("store_block(const Nvfp4StoreParams& params") :]
        self.assertIn("volatile", store[: store.index("quantize_block(")])


if __name__ == "__main__":
    unittest.main()
