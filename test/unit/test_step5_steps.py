# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Step-5 stock step functions on CPU, against SGLang's own forward on stubbed
modules: the same ops in the same order, on the same streams, with the same
arguments.

SGLang's real ``Step5Attention.forward``, ``Step5MoE.forward`` +
``Step5DecoderLayer._forward_ffn``, ``Step5DecoderLayer.forward`` and
``Step5Model.forward``'s tail run on objects built with ``__new__`` whose submodules,
ops and backend log what they are called with; the plugin's step functions run on the
same objects (the attention through SGLang's Step-5 view of the same backend); the
two logs must be equal. On an SGLang without the Step-5 view the attention cases skip,
naming it.
"""

import contextlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as fi_trtllm
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import step5 as sgl_step5

try:
    from sglang.srt.kernel_plugins.megakernel import binding as sgl_binding
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels import binding as sgl_binding

from sgl_nvidia_mega.step5 import attention, boundary, ffn, runtime
from sgl_nvidia_mega.step5.plan import Slots
from sgl_nvidia_mega.step5.runtime.executor import Step5Verify

Step5View = getattr(sgl_binding, "Step5View", None)

# ------------------------------------------------------------------ the op log


class Log(list):
    stream = "main"

    def op(self, name, out=None):
        def run(*args, **kwargs):
            kw = tuple(sorted((k, tag(v)) for k, v in kwargs.items()))
            self.append((name, self.stream, tuple(map(tag, args)), kw))
            return out(*args, **kwargs) if callable(out) else out

        return run


class T:
    """A tensor stand-in: a name, a shape, and what the forwards call on tensors."""

    def __init__(self, log, name, shape=(4, 4096), dtype=torch.bfloat16):
        self.log, self.name, self.shape, self.dtype = log, name, tuple(shape), dtype
        self.is_cuda = True

    def record_stream(self, stream):
        self.log.append(("record", self.name, stream.name))

    def _derived(self, what, shape=None):
        return T(self.log, f"{self.name}.{what}", shape or self.shape, self.dtype)

    def flatten(self, dim):
        return self._derived("flat")

    def reshape(self, *shape):
        return self._derived("reshape")

    def unsqueeze(self, dim):
        return self._derived("unsqueeze")

    def expand(self, *shape):
        return self._derived("expand")

    def __add__(self, other):
        self.log.append(("add", self.log.stream, self.name, tag(other)))
        return T(self.log, f"({self.name}+{tag(other)})", self.shape, self.dtype)


def tag(v):
    if isinstance(v, T):
        return v.name
    if isinstance(v, Stream):
        return f"stream:{v.name}"
    if callable(v) and getattr(v, "__name__", "") in ("run_side_work", "<lambda>"):
        return f"callable:{v.__name__}"
    if isinstance(v, (str, bool, int, float, type(None), torch.dtype)):
        return v
    if isinstance(v, tuple):
        return tuple(tag(x) for x in v)
    return type(v).__name__


class Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait", self.name, other.name))


@contextlib.contextmanager
def streams(log, capture):
    main = Stream("main", log)

    @contextlib.contextmanager
    def on(stream):
        prev, log.stream = log.stream, stream.name
        try:
            yield
        finally:
            log.stream = prev

    with (
        patch.object(torch.cuda, "current_stream", return_value=main),
        patch.object(torch.cuda, "stream", side_effect=on),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=capture),
    ):
        yield main


def _same(logs):
    assert logs[0] == logs[1], (
        "\n".join(map(str, logs[0])) + "\n----\n" + "\n".join(map(str, logs[1]))
    )


class _Callable:
    def __init__(self, fn, **attrs):
        self._fn = fn
        self.__dict__.update(attrs)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


# ------------------------------------------------------------------- attention


def _attention(log, *, dsa, rows, kv_target, dsa_gate):
    """A Step5Attention (fused projection) whose ops log, and its backend."""
    a = sgl_step5.Step5Attention.__new__(sgl_step5.Step5Attention)
    alt = Stream("alt", log)
    qkv = T(log, "qkv", (rows, 3472))
    q, k, v, gate = (T(log, n, (rows, 16)) for n in ("q", "k", "v", "gate"))
    selection = (T(log, "packed"), T(log, "counts"), T(log, "lengths"))

    def attn(qq, kk, vv, fb, **kw):
        # what the backend does with the call: the DSA store, the deferred indexer,
        # the join, the head gate getter, the attention core
        call_kw = tuple(sorted((x, tag(y)) for x, y in kw.items()))
        log.append(("attn", log.stream, (tag(qq), tag(kk), tag(vv)), call_kw))
        sel = kw.get("index_selection")
        if callable(sel):
            sel = sel()
        if kw.get("index_stream") is not None:
            torch.cuda.current_stream().wait_stream(kw["index_stream"])
        if kw.get("head_gate_getter") is not None:
            log.append(("gate_getter", tag(kw["head_gate_getter"]())))
        log.append(("core", log.stream, tag(sel)))
        return T(log, "attn_out", (rows, 16, 192))

    a.__dict__.update(
        is_dsa=dsa,
        alt_stream=alt,
        dsa_overlap_max_rows=256,
        qkv_gate_weight=object(),
        _qkv_gate_layer=SimpleNamespace(),
        qkv_proj=SimpleNamespace(
            quant_method=SimpleNamespace(apply=log.op("qkvg", qkv))
        ),
        q_norm=SimpleNamespace(weight=T(log, "q_norm.w", (192,))),
        k_norm=SimpleNamespace(weight=T(log, "k_norm.w", (192,))),
        rotary_emb=SimpleNamespace(cos_sin_cache=T(log, "cos_sin", (8, 192))),
        eps=1e-5,
        num_heads=16,
        head_dim=192,
        indexer=log.op("indexer", selection),
        o_proj=log.op("o_proj", (T(log, "o_out", (rows, 4096)), None)),
    )
    a.attn = _Callable(attn, layer_id=3 if dsa else 4)
    loc = T(log, "swa_loc", (rows,), torch.int64)
    backend = SimpleNamespace(
        supports_swa_head_gate=lambda: True,
        supports_dsa_head_gate=lambda: dsa_gate,
        swa_kv_write_target=lambda layer: (
            (T(log, "k_cache", (99, 192)), T(log, "v_cache", (99, 192)), loc)
            if kv_target
            else None
        ),
        get_indexer_metadata=lambda layer_id, fb: "indexer_md",
    )
    return a, backend, (q, k, v, gate)


@pytest.mark.skipif(
    Step5View is None,
    reason="this SGLang has no megakernel Step-5 view (ForwardView.step5)",
)
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize(
    "dsa, rows, kv_target, dsa_gate",
    [
        (False, 4, True, False),  # SWA: fused SWA K/V store, FA4 gate epilogue
        (False, 8, False, False),  # SWA without a write target: the backend stores
        (True, 4, False, True),  # DSA, 4 rows: the head gate in the merge
        (True, 8, False, False),  # DSA above 4 rows: fused_sigmoid_mul
    ],
)
def test_the_attention_step_issues_what_sglang_issues(
    capture, dsa, rows, kv_target, dsa_gate
):
    logs = []
    for run in ("sglang", "plugin"):
        log = Log()
        a, backend, qkvg_out = _attention(
            log, dsa=dsa, rows=rows, kv_target=kv_target, dsa_gate=dsa_gate
        )
        x = T(log, "x", (rows, 4096))
        pos = T(log, "pos", (rows,), torch.int64)
        fb = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        qknorm = log.op("qknorm", qkvg_out)
        exec_ns = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False)
        )
        with (
            streams(log, capture),
            patch.object(sgl_step5, "get_exec", return_value=exec_ns),
            patch("sglang.srt.runtime_context.get_exec", return_value=exec_ns),
            patch.object(sgl_step5, "get_attn_backend", return_value=backend),
            patch.object(sgl_step5, "is_in_breakable_cuda_graph", return_value=False),
            patch.object(sgl_step5, "step5_qknorm_rope_gated", side_effect=qknorm),
            patch(
                "sglang.kernels.ops.attention.step5.qknorm_rope.step5_qknorm_rope_gated",
                side_effect=qknorm,
            ),
            patch.object(
                sgl_step5,
                "fused_sigmoid_mul",
                side_effect=log.op("sigmoid_mul", T(log, "gated")),
            ),
        ):
            if run == "sglang":
                out = a.forward(positions=pos, hidden_states=x, forward_batch=fb)
            else:
                layer = SimpleNamespace(self_attn=a)
                st = SimpleNamespace(
                    fb=fb, positions=pos, step5=Step5View(backend), slots=Slots()
                )
                out = attention.step(layer, x, st)
        log.append(("return", tag(out)))
        logs.append(list(log))
    _same(logs)


def test_the_attention_switch_runs_sglangs_attention_module_whole(monkeypatch):
    """Test only (SGLANG_NVIDIA_MEGAKERNEL_STEP5_ATTN=module): the layer's
    Step5Attention is called as a module, with the forward's own arguments."""
    monkeypatch.setattr(runtime, "ATTN_MODE", "module")
    calls = []
    layer = SimpleNamespace(self_attn=lambda **kw: calls.append(kw) or "part")
    st = SimpleNamespace(fb="fb", positions="pos", step5=None)
    assert attention.step(layer, "x", st) == "part"
    assert calls == [dict(positions="pos", hidden_states="x", forward_batch="fb")]


# ------------------------------------------------------------------------- FFN


def _layer_ffn(log, *, kind, rows, can_fuse):
    """A Step5DecoderLayer with its Step5MoE / Step5MLP whose ops log."""
    layer = sgl_step5.Step5DecoderLayer.__new__(sgl_step5.Step5DecoderLayer)
    routed = (
        fi_trtllm.FlashInferTrtllmDeferredFinalizeOutput(
            gemm2_out=T(log, "gemm2_out"),
            expert_weights=T(log, "ew"),
            expanded_idx_to_permuted_idx=T(log, "perm"),
            top_k=8,
        )
        if kind == "fp8"
        else T(log, "routed")
    )
    moe = sgl_step5.Step5MoE.__new__(sgl_step5.Step5MoE)
    # _forward_ffn calls self.moe(...) through nn.Module.__call__
    torch.nn.Module.__init__(moe)
    experts = _Callable(
        log.op("experts", routed),
        forward_deferred_finalize=log.op("experts_deferred", routed),
    )
    moe.__dict__.update(
        attn_dp_size=1,
        alt_stream=Stream("alt", log),
        overlap_max_rows=256,
        share_expert=log.op("shared", T(log, "shared_out")),
        router=log.op("router", "topk"),
        use_mega_moe=False,
        fuse_shared_finalize=kind == "fp8",
        experts=experts,
    )
    layer.__dict__.update(
        use_moe=kind != "dense",
        use_dp_attention_reduce=False,
        moe=moe,
        mlp=log.op("mlp", T(log, "mlp_out")),
        boundary_fusions=SimpleNamespace(can_fuse_moe=lambda s, r: can_fuse),
    )
    return layer


@pytest.mark.parametrize(
    "kind, defer, can_fuse",
    [
        ("fp8", True, True),  # the Step5DeferredMoE handoff
        ("fp8", True, False),  # finalize + shared add in the FFN
        ("fp8", False, True),  # no multicast communicator: finalize in the FFN
        ("bf16", True, True),  # layers 88-90: routed + shared
        ("dense", False, False),
    ],
)
def test_the_ffn_step_issues_what_sglang_issues(kind, defer, can_fuse):
    logs = []
    for run in ("sglang", "plugin"):
        log = Log()
        layer = _layer_ffn(log, kind=kind, rows=8, can_fuse=can_fuse)
        x = T(log, "x", (8, 4096))
        fb = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY, num_token_non_padded="ntnp"
        )
        finalize = log.op("finalize", T(log, "combined"))
        with (
            streams(log, capture=True),
            patch.object(
                sgl_step5,
                "get_forward",
                return_value=SimpleNamespace(defer_moe_finalize=defer),
            ),
            patch.object(
                sgl_step5,
                "get_moe_runner_backend",
                return_value=SimpleNamespace(value="flashinfer_trtllm"),
            ),
            patch.object(
                sgl_step5,
                "finalize_flashinfer_trtllm_deferred_output",
                side_effect=finalize,
            ),
            patch.object(
                fi_trtllm,
                "finalize_flashinfer_trtllm_deferred_output",
                side_effect=finalize,
            ),
        ):
            if run == "sglang":
                out = layer._forward_ffn(hidden_states=x, forward_batch=fb)
            else:
                st = SimpleNamespace(
                    fb=fb, bound={"ffn.runner_defers": True}, slots=Slots()
                )
                out = ffn.step(layer, x, st, defer=defer)
        log.append(("return", type(out).__name__, tag(getattr(out, "shared", out))))
        logs.append(list(log))
    _same(logs)


# ---------------------------------------------------------------- the layer


class _Ctx:
    def __init__(self, log, exit_):
        self.log, self.exit_ = log, exit_

    def __enter__(self):
        self.log.append(("ffn_exit.enter",))
        return self.exit_

    def __exit__(self, *a):
        self.log.append(("ffn_exit.exit",))


def test_a_layer_runs_what_sglangs_decoder_layer_runs():
    """Step5DecoderLayer.forward's TP arm against the executor's layer, with the
    sublayers stubbed: the same boundary calls with the same arguments."""
    logs = []
    for run in ("sglang", "plugin"):
        log = Log()
        layer = sgl_step5.Step5DecoderLayer.__new__(sgl_step5.Step5DecoderLayer)
        exit_ = SimpleNamespace(
            defer_moe_finalize=True, finish=log.op("ffn_exit.finish", "h_next")
        )
        layer.__dict__.update(
            use_dp_attention_reduce=False,
            attn_boundary=SimpleNamespace(
                prepare=log.op("attn.prepare", "x_attn"),
                finish=log.op("attn.finish", "h_attn"),
            ),
            ffn_boundary=SimpleNamespace(
                prepare=log.op("ffn.prepare", "x_ffn"),
                exit=lambda fb: _Ctx(log, exit_),
            ),
        )
        layer.__dict__["self_attn"] = log.op("self_attn", "part")
        layer.__dict__["_forward_ffn"] = log.op("ffn", "out")
        fb = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        if run == "sglang":
            out = sgl_step5.Step5DecoderLayer.forward(
                layer, positions="pos", hidden_states="h", forward_batch=fb
            )
        else:
            st = SimpleNamespace(fb=fb, positions="pos")

            def attention_step(lay, x, s):
                return lay.self_attn(
                    positions=s.positions, hidden_states=x, forward_batch=s.fb
                )

            def ffn_step(lay, x, s, defer):
                log.append(("defer", defer))
                return lay._forward_ffn(hidden_states=x, forward_batch=s.fb)

            with (
                patch.object(attention, "step", side_effect=attention_step),
                patch.object(ffn, "step", side_effect=ffn_step),
            ):
                out = Step5Verify._layer(layer, "h", st)
        log.append(("return", out))
        logs.append([e for e in log if e[0] != "defer"])
    _same(logs)


# ----------------------------------------------------------------- the tail


class _RecordingBatch(SimpleNamespace):
    """A ForwardBatch stand-in that records who assigns its attributes."""

    def __setattr__(self, name, value):
        writers = self.__dict__.setdefault("_writers", [])
        writers.append((name, sys._getframe(1).f_globals.get("__name__")))
        super().__setattr__(name, value)


def test_the_tail_is_step5_models_and_writes_no_forward_batch_attribute():
    """close_stream: the same BF16 sum and the same TORCH-backend norm input as
    Step5Model.forward's tail (export, reset, add), with the stream dropped by SGLang's
    residual_batch, never by a plugin assignment."""
    torch.manual_seed(0)
    out = torch.randn(8, 4096).bfloat16()
    residual = torch.randn(8, 4096).bfloat16()
    results = []
    for run in ("sglang", "plugin"):
        fb = _RecordingBatch()
        residual_batch.start(fb)
        stream = residual_batch.stream_of(fb)
        stream.residual = residual.clone()
        handle = stream.record(out.clone(), PLAIN_ADD)
        if run == "sglang":
            h, r = residual_batch.stream_of(fb).export(handle)
            fb.residual_stream = None
            summed = h + r
        else:
            normed = []
            model = SimpleNamespace(norm=lambda x, backend: normed.append(backend) or x)
            _, before = boundary.close_stream(
                SimpleNamespace(fb=fb, model=model), handle
            )
            summed = before.bfloat16()
            assert [b.value for b in normed] == ["torch"]
            writers = [w for w in fb._writers if w[0] == "residual_stream"]
            assert all(
                not (m or "").startswith("sgl_nvidia_mega") for _, m in writers
            ), writers
        assert fb.residual_stream is None
        results.append(summed)
    assert torch.equal(results[0].view(torch.int16), results[1].view(torch.int16))
