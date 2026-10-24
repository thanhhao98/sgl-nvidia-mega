# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/step5.py (Step5Attention.forward).
"""The attention of a covered target-verify step, over the layer's loaded weights,
SGLang's Step5AttentionBackend and its KV pools.

Stock: the CUDA arm of ``Step5Attention.forward`` with the fused q/k/v/gate GEMM
(``qkv_gate_weight``), outside breakable graphs, on a non-idle verify batch, with
SGLang's own ops in its order. The branches this envelope excludes (unfused
projection, breakable graphs, idle batches) are refused at load or by ``plan``. The
backend decisions SGLang's forward asks of ``get_attn_backend()`` come from the Step-5
view of SGLang's megakernel binding (``st.step5``: the same backend's answers).

SWA layers (69): the fused GEMM; ``step5_qknorm_rope_gated`` with the paged SWA K/V
store fused in (``kv_write`` = the backend's write target, so the attention does not
store again); FA4 through RadixAttention with the sigmoid head gate in its epilogue;
``o_proj``. DSA layers (23): fork the alternate stream before the GEMM; the CSA indexer
(index-weight GEMM, q/k/z projection, norm/RoPE/E4M3 pack, ring append + R8
compression, scoring, top-512, pack) on it, recorded just before the join when
capturing (``defer_side_work``) and at once otherwise; the q/k norm + RoPE on the main
stream; the backend stores the full-pool K/V at ``out_cache_loc``, runs the deferred
indexer, joins, runs the sparse decode (the head gate fused into the merge at 4 rows,
else ``fused_sigmoid_mul``); ``o_proj``. Returns the TP partial, whose all-reduce the
boundary runs.

Seams (slots replace one each): ``qkvg``, ``qknorm``, ``indexer``, ``core`` (the
backend call), ``head_gate``, ``o_proj``. Test only: with
``SGLANG_NVIDIA_MEGAKERNEL_STEP5_ATTN=module`` (runtime.ATTN_ENV) ``step`` runs
SGLang's attention module whole instead, to bisect a parity failure of this code.
"""

from __future__ import annotations

import torch

from sgl_nvidia_mega.step5 import runtime

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/step5.py": (
        # the module alias step5.py imports the frozen options under
        "step5_options",
        "self_attn",
        "forward(positions)",
        "forward(hidden_states)",
        "forward(forward_batch)",
        "is_dsa",
        "alt_stream",
        "dsa_overlap_max_rows",
        "qkv_proj",
        "_qkv_gate_layer",
        "q_norm",
        "k_norm",
        "rotary_emb",
        "eps",
        "indexer",
        "attn",
        "o_proj",
        "_apply_head_gate",
    ),
    "srt/layers/linear.py": ("quant_method",),
    "srt/layers/quantization/unquant.py": ("apply",),
    "srt/layers/rotary_embedding/*.py": ("cos_sin_cache",),
    "srt/layers/radix_attention.py": ("layer_id",),
    "srt/layers/attention/step5/indexer.py": ("forward(weights_first)",),
    "srt/layers/attention/step5_backend.py": (
        "_forward_dsa(index_selection)",
        "_forward_dsa(index_stream)",
        "_forward_dsa(head_gate_getter)",
    ),
    "kernels/ops/attention/step5/qknorm_rope.py": (
        "step5_qknorm_rope_gated",
        "step5_qknorm_rope_gated(kv_write)",
    ),
    "kernels/ops/attention/step5/options.py": (
        "SWA_FA4_GATE_EPILOGUE",
        "HEAD_GATE",
        "SWA_FUSED_QKNORM_STORE",
        "INDEXER_WEIGHT_FIRST",
    ),
    "srt/runtime_context.py": ("get_exec",),
    "srt/arg_groups/fields/*.py": ("enable_deterministic_inference",),
    # the Step-5 view (SGLang's Step5View on ForwardView.step5)
    "srt/kernel_plugins/megakernel/binding.py|srt/megakernels/binding.py": (
        "step5",
        "swa_head_gate",
        "dsa_head_gate",
        "swa_kv_write_target",
        "indexer_metadata",
    ),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/step5.py:Step5Attention.forward": "0c0d101c086d050d",
}


def step(layer, hidden_states, st):
    """``layer.self_attn.forward`` on the normed input [rows, 4096]: the TP-partial
    output [rows, 4096] (Step5Attention.forward, CUDA fused-projection arm)."""
    if runtime.ATTN_MODE == "module":
        # test only: SGLang's attention module whole (it asks its backend itself)
        return layer.self_attn(
            positions=st.positions, hidden_states=hidden_states, forward_batch=st.fb
        )
    from sglang.kernels.ops.attention.step5 import options as step5_options
    from sglang.srt.runtime_context import get_exec

    a = layer.self_attn
    fb, s5 = st.fb, st.step5
    rows = hidden_states.shape[0]
    # fused projection: the alternate stream carries the DSA indexer only; the
    # forward mode is target verify (plan), never idle
    use_alt_stream = (
        a.is_dsa
        and a.alt_stream is not None
        and hidden_states.is_cuda
        and 0 < rows <= a.dsa_overlap_max_rows
    )
    use_indexer_alt_stream = use_alt_stream
    defer_side_work = use_alt_stream and torch.cuda.is_current_stream_capturing()
    gate = None
    selection = None
    if a.is_dsa:
        indexer_metadata = s5.indexer_metadata(a.attn.layer_id, fb)
    if use_alt_stream:
        current_stream = torch.cuda.current_stream()
        a.alt_stream.wait_stream(current_stream)
        hidden_states.record_stream(a.alt_stream)

    def run_side_work():
        with torch.cuda.stream(a.alt_stream):
            return indexer(
                layer,
                hidden_states,
                indexer_metadata,
                st,
                weights_first=step5_options.INDEXER_WEIGHT_FIRST,
            )

    qkv = qkvg(layer, hidden_states, st)
    if use_alt_stream:
        # SGLang keeps eager submission order; under capture it records the side
        # branch just before its join so replay keeps the main lane (the fork stays)
        selection = run_side_work if defer_side_work else run_side_work()
    use_gate_epilogue = (
        not a.is_dsa and step5_options.SWA_FA4_GATE_EPILOGUE and s5.swa_head_gate
    )
    use_dsa_gate_epilogue = (
        a.is_dsa
        and not torch.compiler.is_compiling()
        and not get_exec().deterministic.enable_deterministic_inference
        and step5_options.HEAD_GATE
        and rows == 4
        and hidden_states.dtype == torch.bfloat16
        and s5.dsa_head_gate
    )
    kv_write = None
    if not a.is_dsa and step5_options.SWA_FUSED_QKNORM_STORE:
        candidate = s5.swa_kv_write_target(a.attn)
        if (
            candidate is not None
            and candidate[2].shape == (qkv.shape[0],)
            and candidate[2].dtype == torch.int64
        ):
            kv_write = candidate
    save_kv_cache = kv_write is None
    q, k, v, gate = qknorm(layer, qkv, kv_write, st)
    if a.is_dsa:
        if not use_indexer_alt_stream:
            selection = indexer(layer, hidden_states, indexer_metadata, st)
        gate_kwargs = (
            {"head_gate_getter": lambda: gate} if use_dsa_gate_epilogue else {}
        )
        output = core(
            layer,
            q,
            k,
            v,
            st,
            save_kv_cache=save_kv_cache,
            index_selection=selection,
            index_stream=a.alt_stream if use_indexer_alt_stream else None,
            **gate_kwargs,
        )
    elif st.slots.swa_attn:
        from sgl_nvidia_mega.step5.attention import swa_attn

        # slot swa_attn: the page-split kernels in place of FA4 (the gate applied)
        output = swa_attn.attend(
            layer, q, gate, kv_write, st, gate_epilogue=use_gate_epilogue
        )
    elif use_gate_epilogue:
        output = core(layer, q, k, v, st, head_gate=gate, save_kv_cache=save_kv_cache)
    else:
        output = core(layer, q, k, v, st, save_kv_cache=save_kv_cache)
    if not (use_gate_epilogue or use_dsa_gate_epilogue):
        output = head_gate(layer, output, gate, st)
    return o_proj(layer, output, st)


# --- seams: each is SGLang's own call in the stock path ---------------------------


def qkvg(layer, hidden_states, st):
    """The fused q/k/v/gate GEMM [rows, 3472] (TGV up to 64 rows, else cuBLAS)."""
    a = layer.self_attn
    return a.qkv_proj.quant_method.apply(a._qkv_gate_layer, hidden_states)


def qknorm(layer, qkv, kv_write, st):
    """q/k RMSNorm(1 + w), NeoX RoPE (192 dims SWA, 64 DSA), the gate split; with
    ``kv_write`` the paged SWA K/V store too (bit-identical rows)."""
    from sglang.kernels.ops.attention.step5.qknorm_rope import step5_qknorm_rope_gated

    a = layer.self_attn
    return step5_qknorm_rope_gated(
        qkv,
        a.q_norm.weight,
        a.k_norm.weight,
        st.positions,
        a.rotary_emb.cos_sin_cache,
        eps=a.eps,
        kv_write=kv_write,
    )


def indexer(layer, hidden_states, metadata, st, **kwargs):
    """The CSA indexer: (packed regions [rows, 513] i64, counts, lengths). It also
    appends this step's index keys to the layer's ring and publishes a completed
    region's compressed key (Step5Indexer.forward)."""
    return layer.self_attn.indexer(hidden_states, st.positions, metadata, **kwargs)


def core(layer, q, k, v, st, **kwargs):
    """The attention core through RadixAttention -> Step5AttentionBackend.forward_extend
    (TARGET_VERIFY is an extend mode): FA4 over the paged SWA cache, or the DSA KV
    store + deferred indexer + join + sparse decode."""
    return layer.self_attn.attn(q, k, v, st.fb, **kwargs)


def head_gate(layer, output, gate, st):
    """The sigmoid head gate where no epilogue applied it (DSA above 4 rows)."""
    return layer.self_attn._apply_head_gate(output, gate)


def o_proj(layer, output, st):
    """o_proj [rows, 3072] -> the TP partial [rows, 4096] (reduce_results=False)."""
    out, _ = layer.self_attn.o_proj(output.flatten(1))
    return out
