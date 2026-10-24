# SGLang-NVIDIA-Mega

## Introduction

**SGLang-NVIDIA-Mega** holds the NVIDIA megakernels of the [SGLang](https://github.com/sgl-project/sglang) framework: for each model it covers, the code of its small-batch decode step and the kernels that only make sense inside such a step, which hand their outputs to each other through mailboxes and programmatic dependent launch instead of through per-op boundaries.

The package loads as an SGLang [general plugin](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/plugin.mdx) (the `sglang.srt.plugins` entry point) and is turned on with SGLang's `--enable-nvidia-megakernel` server flag. The package registers each model's spec with SGLang's megakernels framework (`sglang.kernels.plugin_kernels.nvidia`, `sglang.srt.megakernels` before sgl-project/sglang#45389). With the flag, SGLang checks the setup at argument resolution, before the weights load. After the model runner has prepared a covered model (before the KV cache is sized), SGLang builds the megakernel and replaces that model instance's `forward`, which runs each covered step itself and leaves every other step to the model's own forward. Every rank then logs `NVIDIA megakernel: active (…)`. Without the flag, SGLang runs untouched.

> **Experimental:** the kernels, the models and setups they cover, and the switch may change.

## Features

### Supported Models

| Model | GPU | Tensor parallel | Coverage |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | GB300 (SM103) | 4 | DSpark target verify steps of 1 to 384 rows (batch 1 to 64 at 6 rows a request) |
| Kimi K3 (packed MXFP4) | GB300 (SM103) | 8 | One-token decode steps of 1–8, 16, 32 and 64 rows |
| Step-5-Preview (FP8) | GB300, B300 (SM103) | 4 | EAGLE target verify steps of 4 to 32 rows (batch 1 to 8 at 4 rows a request) |

For DeepSeek-V4.1-Flash, larger verify steps, prefill and extend run SGLang's
standard path. The DSpark draft step runs on the plugin's draft spec. For Kimi
K3, the plugin covers the 93-layer language model (24 MLA and 69 KDA layers) at
the listed widths; every other call runs the standard path. For Step-5-Preview,
larger verify steps, prefill, extend and the multi-layer MTP draft steps run
SGLang's standard path.

The Kimi K3 adapter is experimental. On the pinned GB300/TP8 configuration
below, every listed CUDA-graph width and padding boundary passed dummy-weight
capture/replay, B1 and B16 passed stock-versus-plugin real-weight canaries, and a
specialized-route/captured-replay canary passed, and the GSM8K quality protocol
matched its frozen stock-path reference score. This is bounded functional and
quality evidence; it does not claim performance, unlisted configurations, production
readiness, or publication readiness.

### What a covered DeepSeek-V4.1-Flash step runs

The step is the plugin's own code over the weights SGLang loaded (`deepseek_v41/runtime`): the embedding and the Engram hashes, then per layer the mHC boundary (`boundary/`), the attention (`attention/`), the MoE (`moe/`) and Engram (`engram/`), then the head and the logits (`logits/`). Each package has a stock implementation that reproduces SGLang's decode step bit for bit, and puts this package's kernels in where the slot plan (`plan.py`) gives them a width:

| Slot | Rows | Kernel |
| --- | --- | --- |
| `wo_b` + attention all-reduce + mHC post + next norm (`hop_a`, `hop_a_batch2`) | 1–16 | `wo_b` GEMM whose epilogue stores its rows into a Lamport mailbox on every rank, and a Lamport consumer that reduces them and runs the mHC post, combine and RMSNorm |
| MoE all-reduce + mHC post + next norm (`hop_b`, `hop_b_wide`) | 1–48 | Hop B: the top-k finalize and the shared-expert add on a Lamport mailbox, whose consumer runs the mHC post, combine and RMSNorm; past 8 rows it writes the next input's MXFP8 scales per 8-row tile, so the next layer's `wqkv_a` runs on the low-latency GEMM. Hop A's widths use hop A's mailbox, the wider ones a second one |
| Routed experts (`moe`) | 1–48 | A route kernel and a pair of block-scaled tcgen05 GEMMs with 8-row tiles; an expert chosen by more than 8 rows takes several tiles |
| Router top-k (`gate`, `gate_wide`) | 1–48 | One CTA per row, bit for bit SGLang's fused gate (including the vision-token routing) |
| Shared expert gate_up (`shared_ll`) | 1–48 | FlashInfer's low-latency MXFP8 GEMM with 8-row tiles on the alternate stream from the layer's fork, after a one-pass MXFP8 quant into its row-tile layout (bit for bit the shared linear's own quant); SGLang's clamped SwiGLU and `down_proj` follow |
| Replicated embedding (`embed`) | 1–48 | SGLang's fused lookup on the whole table, all-gathered once at build; no all-reduce |
| Indexer top-k (`attn_indexer`) | 30–48 (batch 5–8) | The decode indexer's top-k in the plugin's forward, reading SGLang's paged indexer metadata and writing its selection into SGLang's buffer; the index query and head weights run on the layer's second stream, under the compressor. At 6–24 rows, chunked metadata or without paged metadata SGLang's indexer runs |
| Engram prefetch (`engram`) | 1–8 (batch 1) | Every Engram layer's lookup and WKV quarter on a side stream; their two all-reduces on the main stream (SGLang's TP group) |
| Attention and MoE all-reduce + mHC post + next norm (`two_shot`) | 49–384 | Two-shot exchange: each rank sends row t only to its owner rank (t mod 4), which sums the four partials in rank order and sends the row back; the MoE post also folds the top-k finalize and the shared-expert add, and quantizes the next input to MXFP8 where SGLang's fused post does |

With the megakernel on, the default 39 slots run: the ones above, the attention chain, hop B's triggers, and the DSpark draft step's 16 slots. `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (comma-separated) overrides that list for tests and bisecting; empty runs every step function's stock code. The opt-in slots run only when listed there: `target_lm_head` (the target's LM head on the low-M kernel at 1-8 rows), `lm_head_pinned_tactic` (both low-M LM heads on one fixed tactic per row count, untuned), `hop_b_quad_trigger` (hop B's finalize-publish launches its consumer right after its own wait, batch 1 only), `hop_b_cap16` and `draft_embed`.

### What a covered Step-5-Preview step runs

The step is the plugin's own code over the modules SGLang loaded (`step5/runtime`): the embedding, then per layer the residual stream between the sublayers (`boundary/`), the attention (`attention/`: sliding-window layers and the sparse DSA layers with their CSA indexer) and the FFN (`ffn/`: the dense MLP of layers 0–2 and 91, the MoE of layers 3–90), then the logits (`logits/`). It copies the bodies of SGLang's Step-5 forward functions (the VL model's forward and its embedding routine, `Step5ForCausalLM.forward`, `Step5Model.forward`, `Step5DecoderLayer.forward` and `_forward_ffn`, `Step5Attention.forward` and `Step5MoE.forward`, each pinned by the digest of its code in `test/unit/test_contract.py`) and calls SGLang's own submodules, kernels and stage boundaries in SGLang's order, so it reproduces SGLang's verify step bit for bit. The attention reads the backend's decisions through the Step-5 view of SGLang's megakernel binding (`ForwardView.step5`).

With every slot off (the default: `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` unset or empty) every op of a covered step is SGLang's and the build allocates nothing. `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (comma-separated) names the slots to run; both are opt-in until their gates pass:

| Slot | Seam | Numerics | What it changes |
|---|---|---|---|
| `moe_quant_alt` | `ffn.experts` | bitwise | An FP8 MoE layer (3–87) quantizes the experts' input on the layer's alternate stream, beside the router GEMM: the same `per_token_group_quant_fp8` call on the same tensor, handed to the experts through `forward_deferred_finalize(pre_quant_input=)` (SGLang's DeepSeekFp8 runner must take a pre-quantized pair) |
| `swa_attn` | `attention.core` | tolerance | The 69 sliding-window layers run a page-split attention kernel and its merge (`step5/sm103/attention/swa_split.py`, Triton) instead of FA4, which runs one CTA per request: up to 10 splits a request, 16 or 32 query rows a CTA, PDL with the pre-wait reads of the K / V written by earlier steps; bounds in `runtime.TOLERANCES["swa_attn"]`, asserted by `test/kernels/test_step5_swa_attn.py` |

A requested slot that cannot run (an SGLang without the pre-quantized input, other expert quantization, deterministic inference, a layer shape the kernels do not take) stops the build with its reasons; a step whose SGLang path is not the one a slot replaces stops instead of running the slot on other inputs. Up to 32 rows SGLang's Step-5 path is one tier when its Step-5 multicast all-reduce communicator exists (TP4 and EP1 with the `trtllm` all-reduce fusion backend, on one node with multicast): both boundaries of every layer run its fused all-reduce kernel, and an FP8 MoE layer hands its finalize to the next layer's input kernel. Without the communicator SGLang runs FlashInfer's fusion at every boundary and finalizes the MoE inside the FFN. The stock step follows either, and the startup line names the tier: `boundaries Step-5 multicast at <= 32 rows`, or `FlashInfer boundaries, no multicast communicator (finalize in the FFN)`.

### Design

- **Built at load.** Buffers, kernels and tactics for every covered width are allocated and compiled when the model is loaded, before the KV cache is sized. Nothing is compiled or allocated inside a step.
- **Graph-safe coverage.** A step is covered by its forward mode and row count alone, so every rank decides the same and a captured graph replays the decision made at capture.
- **Fail-stop synchronization.** DeepSeek-V4.1-Flash waits on another GPU have
  a deadline below SGLang's watchdog. A rank that misses it faults at address
  0x0, so no token from a failed step is returned. Kimi K3 validates SGLang's
  finite build deadline and compiles it into all six reachable TP8 remote-wait
  paths; expiry likewise stores through address 0x0 and traps.
- **No silent fallback.** With `--enable-nvidia-megakernel`, an unsupported setup stops the server at startup, before the weights load, listing every reason; a model the loaded weights do not fit stops every rank after the load.

## Quick Start

### Requirements

- An SGLang with the megakernels framework (`sglang.kernels.plugin_kernels.nvidia`, or `sglang.srt.megakernels` before sgl-project/sglang#45389; `--enable-nvidia-megakernel`)
- SGLang's CUDA 13 image: FlashInfer (its low-latency block-scaled GEMM as in 0.7.0 to 0.7.2), CuTe DSL (`nvidia-cutlass-dsl`) and the CUDA Python bindings
- DeepSeek-V4.1-Flash: 4 x GB300
- Kimi K3: 8 x GB300 (SM103), BF16, context length 8192, TP8 and DP1,
  FP8 E4M3 KV cache with page size 64, `trtllm_mla` for prefill and decode,
  `flashinfer_mxfp4` with default precision, and full decode CUDA graphs
- Step-5-Preview: 4 x GB300 or 4 x B300 (SM103), the FP8 checkpoint, and an SGLang
  with the Step-5 model (sgl-project/sglang#44356) whose megakernel binding has the
  Step-5 view (`ForwardView.step5`, `Step5View`); without the view the server
  stops at startup, naming it. Full decode CUDA graphs, BF16 KV cache, page size
  64, `fa4` attention, `flashinfer_trtllm` MoE, EP1, the `trtllm` all-reduce fusion
  backend and EAGLE with 3 steps, top-1 and 4 draft tokens (multi-layer MTP)

### Installation

```bash
git clone https://github.com/sgl-project/sgl-nvidia-mega.git
cd sgl-nvidia-mega
pip install -e .
```

### Usage

DeepSeek-V4.1-Flash:

```bash
SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 \
  --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --dsv4-attn-backend flashmla \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --enable-nvidia-megakernel
```

The three environment variables are required by the DeepSeek-V4.1 spec; without them the server stops at startup, naming each.

Step-5-Preview (FP8) on 4 x GB300 or 4 x B300:

```bash
python -m sglang.launch_server \
  --model-path stepfun-ai/Step-5-Preview \
  --tp 4 \
  --ep-size 1 \
  --attention-backend fa4 \
  --page-size 64 \
  --moe-runner-backend flashinfer_trtllm \
  --flashinfer-allreduce-fusion-backend trtllm \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-multi-layer-eagle \
  --max-running-requests 64 \
  --swa-full-tokens-ratio 0.1 \
  --enable-nvidia-megakernel
```

The megakernel runs the decode graphs of 1 to 8 requests; the graphs of 9 to 64 requests run SGLang's forward. Every rank logs `NVIDIA megakernel: active (Step-5: slots none (stock); verify rows 4-32 (batch 1-8); …)` with the boundary tier, the capture check logs `Step-5 runs the decode graphs of batch sizes [1, 2, 3, 4, 5, 6, 7, 8]`, and the MTP draft's runners log that it keeps its own forward.

## Repository Structure

```text
sgl-nvidia-mega/
├── sgl_nvidia_mega/
│   ├── plugin.py          # registers the specs with SGLang's megakernels framework
│   ├── common/            # NVRTC builds and PDL launches; CuTe DSL sync primitives
│   ├── deepseek_v41/
│   │   ├── spec.py          # the spec: the setups it serves and its build
│   │   ├── plan.py          # which kernel fills each slot, per width
│   │   ├── runtime/         # the build, the executor and its per-step state, the embedding
│   │   ├── boundary/        # the mHC between sublayers
│   │   ├── attention/       # the attention step
│   │   ├── moe/             # the MoE step
│   │   ├── engram/          # the Engram layers
│   │   ├── logits/          # the head and the logits
│   │   └── sm103/           # kernels and the build-time runner; no SGLang imports
│   ├── kimi_k3/              # the Kimi K3 spec, adapter, executor and CuTe DSL kernels
│   └── step5/             # Step-5-Preview: spec, plan, runtime, boundary, attention, ffn, logits, sm103 (kernels)
├── benchmark/             # the static decode table behind the speed numbers
└── test/
    ├── unit/              # CPU: the plugin, the setup checks, the step loop and the SGLang contract
    ├── kernels/           # GPU: each kernel against SGLang's or FlashInfer's op, fail-stop
    └── e2e/               # GPU: serving accuracy, KL, passkey and fail-stop, switch on and off
```

## Testing

```bash
pytest test/unit               # CPU only
pytest test/kernels            # DeepSeek-V4.1-Flash: 4 x GB300
pytest test/e2e                # DeepSeek-V4.1-Flash: 4 x GB300, with SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH set
pytest test/e2e -k step5       # Step-5-Preview: 4 x GB300 or B300, with SGLANG_NVIDIA_MEGAKERNEL_STEP5_E2E_MODEL_PATH set
```

Kimi K3 GPU and end-to-end suites are not yet part of these commands.

On an SGLang tree that lacks other models, run the Step-5 CPU tests alone: `pytest test/unit/test_step5_spec.py test/unit/test_step5_runtime.py test/unit/test_step5_steps.py test/unit/test_step5_slots.py` (GPU: `pytest test/kernels/test_step5_swa_attn.py`; the chain microbenchmark `python3 test/kernels/bench_step5_swa_attn.py`) and `pytest test/unit/test_contract.py -k "step5 or binding_callable or TOLERATED_WRITES or TOLERATED_READS"`. The Step-5 end-to-end servers serve at most 8 requests, so every covered width has its decode graph; the serving gates (parity, served token identity, profiles and accuracy) run at the deployed `--max-running-requests 64`, since at 8 SGLang builds no MNNVL verify workspace for the steps past 32 rows.

## Contributing

Open a pull request against `main`. The PR title follows `[Type] scope: title` (see `.github/pull_request_template.md`), and `pre-commit run --all-files` must pass.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
