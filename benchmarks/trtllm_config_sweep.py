"""Try TensorRT-LLM configurations against the flat throughput curve."""
import json, time
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig

PROMPT = 'Explain what a robot is, in detail.'
BATCHES = [16, 64]
NTOK = 128

CONFIGS = [
    ('baseline',            dict()),
    ('flashinfer_attn',     dict(attn_backend='FLASHINFER')),
    ('max_num_tokens_32k',  dict(max_num_tokens=32768)),
    ('chunked_prefill',     dict(enable_chunked_prefill=True, max_num_tokens=32768)),
    ('kv_0.7_tokens_32k',   dict(max_num_tokens=32768, _kv=0.70)),
]

def run(name, kw):
    kv = kw.pop('_kv', 0.40)
    llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct', dtype='float16',
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=kv), **kw)
    sp = SamplingParams(temperature=0.0, max_tokens=NTOK, min_tokens=NTOK)
    llm.generate([PROMPT] * 4, sp)                      # warm up
    out = {}
    for bs in BATCHES:
        t0 = time.perf_counter()
        res = llm.generate([PROMPT] * bs, sp)
        dt = time.perf_counter() - t0
        got = sum(len(o.outputs[0].token_ids) for o in res)
        out[bs] = got / dt
        print(f'  {name:20s} batch {bs:3d}: {got/dt:9.1f} tok/s', flush=True)
    try:
        llm.shutdown()
    except Exception:
        pass
    return out

def main():
    results = {}
    for name, kw in CONFIGS:
        try:
            results[name] = run(name, dict(kw))
        except Exception as e:
            print(f'  {name:20s} FAILED: {type(e).__name__}: {str(e)[:120]}', flush=True)
            results[name] = {'error': f'{type(e).__name__}: {str(e)[:200]}'}
        json.dump(results, open('/out/sweep.json', 'w'), indent=2)

if __name__ == '__main__':
    main()
