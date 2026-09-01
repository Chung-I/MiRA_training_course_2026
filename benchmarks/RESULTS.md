# Measured serving benchmarks

All numbers produced by `bench_serving.py` on the same machine, the same model, and the
same workload, so they are directly comparable.

- **GPU:** NVIDIA RTX 5090 (Blackwell, sm_120, 32 GB), driver 580.173.02
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`, fp16, 0.494 B parameters
- **Workload:** identical prompt, exactly 128 new tokens per request, greedy decoding
- **Date:** 2026-09-01

## Throughput: HuggingFace vs vLLM vs SGLang

Total tokens per second across all concurrent requests.

| batch | HF `generate()` | vLLM 0.27.1 | SGLang 0.5.18 | SGLang, default graph range |
|---:|---:|---:|---:|---:|
| 1 | 165 | 726 | 701 | 700 |
| 2 | 321 | 1,310 | 971 | 1,026 |
| 4 | 639 | 2,698 | 2,772 | 2,541 |
| 8 | 1,275 | 5,155 | 4,977 | 4,992 |
| 16 | 2,548 | 9,880 | 9,625 | 9,680 |
| 32 | 4,964 | 19,889 | 18,491 | 5,657 *(eager)* |
| 64 | 9,782 | 37,265 | 32,523 | 11,152 *(eager)* |

vLLM ran natively; SGLang ran in the `lmsysorg/sglang:latest` container, because this
machine has no CUDA toolkit (see the SGLang section below). The last column is SGLang at
its stock settings, kept deliberately, and explained under "the cliff".

### What the numbers say

**A serving engine is worth about 4x, and the two engines are close to each other.** The
gap over HuggingFace is not a batching artifact: `generate()` batches perfectly well and
all three scale close to linearly. vLLM and SGLang are simply about four times faster per
token through CUDA graphs instead of per-step Python dispatch, a paged KV cache instead of
padded contiguous allocation, and fused kernels. Between the two engines the spread is
small, and on a 0.5 B model it is mostly dispatch overhead rather than anything
architectural. Do not read a winner into it.

**Against the bandwidth ceiling.** The notebook predicts a single-stream roof of
**1,812 tok/s** for this model on this card (1.79 TB/s / 0.92 GB of fp16 weights):

| | batch-1 tok/s | share of the roof |
|---|---:|---:|
| HF `generate()` | 165 | **9%** |
| SGLang | 701 | **39%** |
| vLLM | 726 | **40%** |

Same GPU, same model, same tokens. The hardware was never the problem.

**Batching beats the single-stream roof entirely.** vLLM at batch 64 reaches 37,265 tok/s,
about **20x the single-stream ceiling**, because the ceiling is per stream. The weights are
read once for the whole batch, so every extra concurrent request is nearly free. This is
the economic argument for continuous batching, measured.

### The cliff: a default that costs half your throughput

The last column is the same SGLang with nothing configured. It tracks vLLM to batch 16 and
then collapses: 5,657 tok/s at batch 32 against 18,491 once configured.

SGLang captures decode CUDA graphs for a fixed set of batch sizes, and by default that set
stops at 24:

```
Capturing batches (bs=1 2 4 8 12 16 24)
```

Any batch beyond 24 finds no graph and falls back to eager execution, which is most of the
4x. Raising `cuda_graph_max_bs_decode` to 64 extends the capture set to
`1 2 4 8 12 16 24 32 40 48 56 64` and the cliff disappears.

Two things worth taking from this:

1. **Benchmark at the concurrency you will actually serve.** Had this been measured only up
   to batch 16, the default would have looked fine. Had it been measured only at 32 and 64,
   SGLang would have looked like the slower engine. Neither reading is true.
2. **A wrong number is more dangerous than a crash.** The missing CUDA toolkit announced
   itself with a stack trace. This one produced a plausible, quotable, and completely
   misleading throughput curve.

## SGLang: the pip install cannot serve here, the container can

SGLang 0.5.9 installs cleanly from pip (11 GB, torch 2.9.1+cu128) and then cannot serve on
this machine. Running the official container instead worked on the first attempt, which is
why there are SGLang numbers above.

**This machine has no CUDA toolkit.** SGLang JIT-compiles attention kernels for the exact
architecture it finds (`sm_120a`) at startup, through FlashInfer, which shells out to
`nvcc`. There is no `nvcc` on this box, and the pip wheel that looks like it supplies one,
`nvidia-cuda-nvcc-cu12` 12.8.93, installs exactly one binary: `ptxas`. Setting `CUDA_HOME`
only moves the error from an assertion to an honest `nvcc: not found`.

Switching to `attention_backend="triton"` gets past FlashInfer but fails the same way during
CUDA graph capture (`ninja exited with status 127`).

**There is no pip wheel that fixes this.** `nvidia-cuda-nvcc-cu12` ships exactly one
binary, `ptxas`, at every 12.x version, and `cuda-toolkit[all]==12.8.2` only depends on that
same wheel. The package that does ship a real `nvcc` is `nvidia-cuda-nvcc`, which exists
only for 13.x, and SGLang 0.5.9 pins torch cu128.

**The fix used here: the official container**, which ships a complete CUDA toolkit.

```bash
docker run --gpus all --rm --user "$(id -u):$(id -g)" --shm-size 16g \
  -e HOME=/tmp -e HF_HOME=/hf \
  -v ~/.cache/huggingface:/hf -v "$PWD":/bench \
  lmsysorg/sglang:latest \
  python3 /bench/bench_serving.py --backend sglang --out /bench/results_sglang.json
```

It worked first try. The image also carries a newer stack than pip resolved locally,
SGLang 0.5.18 on torch 2.13+cu130. Running as your own UID with `HOME` and `HF_HOME`
pointed at mounts keeps the output file writable and reuses the existing model cache.

The alternative, if you want to stay native, is a CUDA 12.8 toolkit installed into your home
directory with NVIDIA's runfile (`--toolkit --toolkitpath=$HOME/cuda-12.8`, no root needed),
which would also fix vLLM's DeepGEMM path. Do not use `apt install nvidia-cuda-toolkit`: the
Ubuntu candidate is CUDA 12.0, and a 12.0 compiler against 12.8 headers reproduces the same
CCCL version assert that broke vLLM.

### The pattern

Every environment failure here was CUDA toolchain plumbing. None was about models, GPUs, or
any framework's serving logic. Both engines wanted a compiler for FP8 and JIT paths that an
fp16 0.5 B benchmark never executes. "Just use vLLM" is one line in a README and rather more
than one line on a machine that has not been prepared for it, which is the argument for
running these things in their official containers.

## Reproducing

```bash
python bench_serving.py --backend hf     --out results_hf.json
python bench_serving.py --backend vllm   --out results_vllm.json
python bench_serving.py --backend sglang --out results_sglang.json
python bench_serving.py --backend trtllm --out results_trtllm.json
```

## Environment notes, learned the hard way

Getting vLLM to start on a 3-month-old GPU took three separate fixes. None of them were
about Blackwell; all of them were toolchain plumbing. This is the honest version of
"just use vLLM":

1. **`Could not find nvcc`** — vLLM's DeepGEMM path wants a CUDA toolkit. One ships
   inside the venv at `site-packages/nvidia/cu13`; point `CUDA_HOME` at it, or set
   `VLLM_USE_DEEP_GEMM=0`.
2. **`FileNotFoundError: 'ninja'`** — ninja *was* installed, at `<venv>/bin/ninja`.
   Invoking `<venv>/bin/python` by absolute path does **not** put the venv's `bin` on
   `PATH`, so FlashInfer's `subprocess` call could not find it. Put the venv `bin` on
   `PATH` explicitly. This one is worth remembering generally: it fails as a missing
   dependency when it is really a `PATH` problem.
3. **`CUDA compiler and CUDA toolkit headers are incompatible`** — this venv ships
   nvcc 13.3 against CUDA 13.0 headers, so CCCL's version assert rejects any FlashInfer
   JIT build. Worked around with `VLLM_USE_FLASHINFER_SAMPLER=0`, which avoids the JIT
   entirely; vLLM falls back to its native sampler, which is irrelevant to throughput
   under greedy decoding.

The full launch environment is in `../../bench-logs/run_vllm4.sh` style:

```bash
export CUDA_HOME=<venv>/lib/python3.12/site-packages/nvidia/cu13
export PATH="<venv>/bin:$CUDA_HOME/bin:$PATH"
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0
```
