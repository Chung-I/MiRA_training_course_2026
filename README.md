# MiRA Training Course 2026 — GNN & LLM Serving

Materials for two talks at the MiRA Training Course 2026 (9/3–9/4, R546), for
incoming master's and PhD students.

| Talk | Date | Notebook |
|---|---|---|
| **Graph Neural Networks** | 9/3, 17:20–17:50 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chung-I/MiRA_training_course_2026/blob/main/notebooks/gnn_demo.ipynb) |
| **Fast, efficient, VRAM-friendly LLM serving** | 9/4, 10:20–11:00 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chung-I/MiRA_training_course_2026/blob/main/notebooks/llm_serving_demo.ipynb) |

Click a badge to run the notebook in Google Colab. No setup, no local install.

---

## 1. Graph Neural Networks

**Runs on a free CPU runtime.** PyTorch and NumPy only — both pre-installed on Colab.
Message passing is written out by hand so you can read every line; PyTorch Geometric
appears once at the end, as an optional cell.

The one idea: *every node updates itself by summarizing its neighbors. The hard part is
never the layer — it is deciding what the nodes and the edges are.*

| Demo | What it shows |
|---|---|
| 1 | An MLP's answer changes when you renumber the nodes. A GNN's does not. |
| 2 | A GCN layer in 15 lines: normalized neighbor average, then a linear map. |
| 3 | L layers = an L-hop receptive field. |
| 4 | Node features are pure noise: the MLP gets 51%, the 2-layer GCN gets 91%. The graph *is* the signal. |
| 5 | Building a k-NN graph on a point cloud, at k = 3, 8, 20. Same points, three different models. |
| 6 | Depth stops paying and then falls off a cliff, as the embeddings collapse together (over-smoothing). |
| 7 | Absolute coordinates break under translation; relative positions on edges do not. |
| 8 | The same model in PyTorch Geometric. |

Robotics systems referenced in the talk, all message passing underneath:

| System | Nodes | Edges | Paper |
|---|---|---|---|
| RoboBallet — 8-arm motion planning | arms, tasks, obstacles | possible collisions | [arXiv:2509.05397](https://arxiv.org/abs/2509.05397) |
| DPI-Net — manipulating deformables and fluids | particles | contacts | [arXiv:1810.01566](https://arxiv.org/abs/1810.01566) |
| GNS — learned physics simulator | particles | within a radius | [arXiv:2002.09405](https://arxiv.org/abs/2002.09405) |
| MeshGraphNets | mesh vertices | mesh edges | [arXiv:2010.03409](https://arxiv.org/abs/2010.03409) |
| DGCNN / EdgeConv — point clouds | points | k-NN in feature space | [arXiv:1801.07829](https://arxiv.org/abs/1801.07829) |
| Grasp-the-Graph 2.0 — grasp detection | cloud points | local neighborhoods | [arXiv:2505.02664](https://arxiv.org/abs/2505.02664) |
| Decentralized multi-robot planning | robots | communication range | [arXiv:1912.06095](https://arxiv.org/abs/1912.06095) |
| 3D Dynamic Scene Graphs | places, objects, humans | containment, adjacency | [arXiv:2002.06289](https://arxiv.org/abs/2002.06289) |

Further reading: Stanford CS224W · the PyTorch Geometric examples · Distill,
*A Gentle Introduction to Graph Neural Networks* · Battaglia et al.,
[arXiv:1806.01261](https://arxiv.org/abs/1806.01261).

---

## 2. Fast, efficient, VRAM-friendly LLM serving

**Parts 1–5 need nothing at all** — no GPU, no installs, not even NumPy. That is the
point of the talk: you can answer *"does it fit, and how fast will it be"* with
arithmetic, before you touch a GPU.

**Parts 6–8 need a GPU.** On Colab: *Runtime → Change runtime type → T4 GPU* (free
tier). `torch` and `transformers` are already there.

The one idea: *serving cost is decided by memory, not FLOPs.*

| Part | What it shows | GPU? |
|---|---|---|
| 1 | Weight memory = parameters × bytes. 70B in fp16 is 140 GB. | no |
| 2 | The KV cache formula. Llama-3-8B costs 128 KiB per token; Llama-2-7B costs 512 KiB, because of GQA. | no |
| 3 | Weights are a fixed cost, the KV cache is per user. On a 24 GB card: 7 users at fp16, 36 at int4 weights + fp8 cache. | no |
| 4 | tokens/sec ≲ memory bandwidth ÷ bytes of weights. This is why quantization speeds up decoding. | no |
| 5 | Dollars per million tokens, and when to just use an API. | no |
| 6 | Measured weight memory vs. predicted — they agree to within 1%. | yes |
| 7 | Measured decode speed vs. the bandwidth ceiling. | yes |
| 8 | Batching is nearly free: measured on an RTX 5090, batch 1 → 16 gives 15× the total throughput for 4% slower per-user speed. | yes |

In real life you do not serve models this way. Use [vLLM](https://docs.vllm.ai) or
[SGLang](https://docs.sglang.ai), which do continuous batching and PagedAttention for
you — then re-run the arithmetic from Parts 1–5 against what the server reports.

---

## Running locally instead

The notebooks need only `torch` (both) and `transformers` (LLM Parts 6–8);
`matplotlib` is optional and the GNN notebook prints numbers without it.

```bash
pip install torch matplotlib transformers
jupyter lab notebooks/
```

## Layout

```
notebooks/   the two Colab demos
slides/      talk slides
```
