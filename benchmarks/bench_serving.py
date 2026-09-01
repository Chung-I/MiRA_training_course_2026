#!/usr/bin/env python3
"""Compare HuggingFace generate(), vLLM and SGLang on the same model and workload.

One script, three backends, identical prompts and identical token counts, so the
numbers are comparable. Used to produce the table in the MiRA 2026 serving talk.

    python bench_serving.py --backend hf     --out results_hf.json
    python bench_serving.py --backend vllm   --out results_vllm.json
    python bench_serving.py --backend sglang --out results_sglang.json
"""
import argparse, json, statistics, time

PROMPT = 'Explain what a robot is, in detail.'


def bench_hf(model_id, batch_sizes, n_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16).cuda().eval()
    rows = []
    for bs in batch_sizes:
        ids = tok([PROMPT]*bs, return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():                                   # warm up
            model.generate(**ids, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**ids, max_new_tokens=n_tokens, do_sample=False,
                           min_new_tokens=n_tokens)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        rows.append({'batch': bs, 'seconds': dt, 'total_tok_s': bs*n_tokens/dt,
                     'per_user_tok_s': n_tokens/dt})
        print(f'  batch {bs:3d}: {bs*n_tokens/dt:8.1f} tok/s total')
    return rows


def bench_vllm(model_id, batch_sizes, n_tokens, util):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_id, dtype='float16', gpu_memory_utilization=util,
              max_model_len=2048, enforce_eager=False)
    sp = SamplingParams(temperature=0.0, max_tokens=n_tokens, min_tokens=n_tokens,
                        ignore_eos=True)
    llm.generate([PROMPT]*2, sp)                                # warm up
    rows = []
    for bs in batch_sizes:
        t0 = time.perf_counter()
        outs = llm.generate([PROMPT]*bs, sp)
        dt = time.perf_counter() - t0
        got = sum(len(o.outputs[0].token_ids) for o in outs)
        rows.append({'batch': bs, 'seconds': dt, 'total_tok_s': got/dt,
                     'per_user_tok_s': got/dt/bs})
        print(f'  batch {bs:3d}: {got/dt:8.1f} tok/s total ({got} tokens)')
    return rows


def bench_sglang(model_id, batch_sizes, n_tokens, util, attention_backend='auto'):
    import sglang as sgl
    # 'auto' lets SGLang pick its own default (FlashInfer), which is the fair
    # comparison against vLLM. Only force 'triton' where no CUDA toolkit exists,
    # since FlashInfer JIT-compiles kernels with nvcc at startup.
    kw = {} if attention_backend == 'auto' else {'attention_backend': attention_backend}
    # SGLang captures decode CUDA graphs only up to bs=24 by default, so batches
    # above that silently fall back to eager and the throughput curve collapses.
    # Raise the cap to the largest batch measured, otherwise the comparison is
    # graphs-vs-no-graphs rather than framework-vs-framework.
    llm = sgl.Engine(model_path=model_id, dtype='float16',
                     mem_fraction_static=util, context_length=2048,
                     cuda_graph_max_bs_decode=max(batch_sizes), **kw)
    sp = {'temperature': 0.0, 'max_new_tokens': n_tokens,
          'min_new_tokens': n_tokens, 'ignore_eos': True}
    llm.generate([PROMPT]*2, sp)                                # warm up
    rows = []
    for bs in batch_sizes:
        t0 = time.perf_counter()
        outs = llm.generate([PROMPT]*bs, sp)
        dt = time.perf_counter() - t0
        got = sum(o['meta_info']['completion_tokens'] for o in outs)
        rows.append({'batch': bs, 'seconds': dt, 'total_tok_s': got/dt,
                     'per_user_tok_s': got/dt/bs})
        print(f'  batch {bs:3d}: {got/dt:8.1f} tok/s total ({got} tokens)')
    llm.shutdown()
    return rows


def bench_trtllm(model_id, batch_sizes, n_tokens, util):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig, CudaGraphConfig
    # CudaGraphConfig.max_batch_size defaults to 0, i.e. no CUDA graphs at all.
    # Leaving it flattens the throughput curve, the same trap as SGLang's bs=24 cap.
    llm = LLM(model=model_id, dtype='float16',
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=util),
              cuda_graph_config=CudaGraphConfig(max_batch_size=max(batch_sizes),
                                                enable_padding=True))
    sp = SamplingParams(temperature=0.0, max_tokens=n_tokens, min_tokens=n_tokens)
    llm.generate([PROMPT]*2, sp)                                # warm up / engine build
    rows = []
    for bs in batch_sizes:
        t0 = time.perf_counter()
        outs = llm.generate([PROMPT]*bs, sp)
        dt = time.perf_counter() - t0
        got = sum(len(o.outputs[0].token_ids) for o in outs)
        rows.append({'batch': bs, 'seconds': dt, 'total_tok_s': got/dt,
                     'per_user_tok_s': got/dt/bs})
        print(f'  batch {bs:3d}: {got/dt:8.1f} tok/s total ({got} tokens)')
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True, choices=['hf', 'vllm', 'sglang', 'trtllm'])
    ap.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--batch-sizes', default='1,2,4,8,16,32,64')
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--gpu-util', type=float, default=0.45)
    ap.add_argument('--attention-backend', default='auto')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    bs_list = [int(x) for x in a.batch_sizes.split(',')]

    print(f'=== {a.backend} | {a.model} | {a.tokens} new tokens ===')
    t = time.time()
    if a.backend == 'hf':
        rows = bench_hf(a.model, bs_list, a.tokens)
    elif a.backend == 'vllm':
        rows = bench_vllm(a.model, bs_list, a.tokens, a.gpu_util)
    elif a.backend == 'trtllm':
        rows = bench_trtllm(a.model, bs_list, a.tokens, a.gpu_util)
    else:
        rows = bench_sglang(a.model, bs_list, a.tokens, a.gpu_util, a.attention_backend)

    result = {'backend': a.backend, 'model': a.model, 'tokens': a.tokens,
              'rows': rows, 'wall_seconds': time.time()-t}
    try:
        import torch
        result['gpu'] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if a.out:
        json.dump(result, open(a.out, 'w'), indent=2)
        print('wrote', a.out)
