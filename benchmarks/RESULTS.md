# Measured serving benchmarks

All numbers produced by `bench_serving.py` on the same machine, the same model, and the
same workload, so they are directly comparable.

- **GPU:** NVIDIA RTX 5090 (Blackwell, sm_120, 32 GB), driver 580.173.02
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`, fp16, 0.494 B parameters
- **Workload:** identical prompt, exactly 128 new tokens per request, greedy decoding
- **Date:** 2026-09-01

## Throughput: HuggingFace `generate()` vs vLLM

| batch | HF `generate()` | vLLM 0.27.1 | speedup |
|---:|---:|---:|---:|
| 1 | 165 | 726 | 4.4x |
| 2 | 321 | 1,310 | 4.1x |
| 4 | 639 | 2,698 | 4.2x |
| 8 | 1,275 | 5,155 | 4.0x |
| 16 | 2,548 | 9,880 | 3.9x |
| 32 | 4,964 | 19,889 | 4.0x |
| 64 | 9,782 | 37,265 | 3.8x |

Total tokens per second across all concurrent requests.

### What the numbers say

**A serving engine is worth about 4x, at every batch size.** The gap is not a batching
artifact — HuggingFace batches perfectly well, and both scale close to linearly here.
vLLM is simply about four times faster per token: CUDA graphs instead of per-step Python
dispatch, a paged KV cache instead of padded contiguous allocation, and fused kernels.

**Against the bandwidth ceiling.** The notebook predicts a single-stream roof of
**1,812 tok/s** for this model on this card (1.79 TB/s ÷ 0.92 GB of fp16 weights):

| | batch-1 tok/s | share of the roof |
|---|---:|---:|
| HF `generate()` | 165 | **9%** |
| vLLM | 726 | **40%** |

Same GPU, same model, same tokens. The hardware was never the problem.

**Batching beats the single-stream roof entirely.** vLLM at batch 64 reaches 37,265
tok/s — about **20x the single-stream ceiling** — because the ceiling is per stream. The
weights are read once for the whole batch, so every extra concurrent request is nearly
free. This is the economic argument for continuous batching, measured.

## SGLang: installed, could not serve on this machine

SGLang 0.5.9 installs cleanly (11 GB, torch 2.9.1+cu128). It cannot serve here, and the
reason is worth more than the missing table row.

**This machine has no CUDA toolkit.** SGLang JIT-compiles attention kernels for the exact
architecture it finds (`sm_120a`) at startup, through FlashInfer, which shells out to
`nvcc`. There is no `nvcc` on this box, and the pip wheel that looks like it supplies one,
`nvidia-cuda-nvcc-cu12` 12.8.93, installs exactly one binary: `ptxas`. Setting `CUDA_HOME`
only moves the error from an assertion to an honest `nvcc: not found`.

Switching to `attention_backend="triton"` gets past FlashInfer but fails the same way during
CUDA graph capture (`ninja exited with status 127`).

**Why there is no number here.** The remaining option is `disable_cuda_graph=True`. That
would produce a figure, but SGLang without CUDA graphs against vLLM with them is not a
comparison, it is a misattribution, and a reader would take it as evidence that SGLang is
slower. A missing row is more honest than a misleading one.

**The real fix**, for whoever picks this up: install a CUDA toolkit (`apt install
nvidia-cuda-toolkit`, needs root, or `conda install -c nvidia cuda-nvcc` matching the
venv's torch CUDA version), or run SGLang's official Docker image, which ships one.

### The pattern across all three

Every failure in this exercise was CUDA toolchain plumbing. None was about models, GPUs,
or any framework's serving logic. Both engines demanded a compiler for FP8 and JIT paths
that an fp16 0.5B benchmark never executes. "Just use vLLM" is one line in a README and
rather more than one line on a machine that has not been prepared for it.

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
