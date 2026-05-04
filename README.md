# flash-moe-rocm-demo

> Run **MoE language models that don't fit in VRAM** on a single AMD GPU, using a 3-tier (VRAM LRU → OS page cache → SSD) expert weight cache integrated into [llama.cpp](https://github.com/ggml-org/llama.cpp).

This repo is the AMD Developer Hackathon 2026 submission. It contains:

- **The agent demo** — a small Python harness that exercises a local `llama-server` running our MoE-streaming branch with three real tools (file IO, shell, HTTP fetch).
- **Benchmarks** — `docs/bench-*.md` with measurements on MI300X (gfx942) for a model that fits VRAM (Qwen3-30B-A3B Q4_K_M) and one that doesn't (DeepSeek-V3 671B Q4_K_M, ~377 GB).
- **Build scripts** — `scripts/build_llamacpp.sh` clones llama.cpp, applies our patch series, and builds for ROCm gfx942.
- **Pointers to the upstream PR** — the actual engine work lives in a [llama.cpp PR](#) (link added when filed); this repo is the demo on top.

## Why

Large MoE checkpoints (DeepSeek-V3 671B, Qwen3-235B-A22B, future >1T models) don't fit a single consumer or single-card datacenter GPU. Today the workarounds are CPU offload (slow), tensor-parallel multi-GPU (expensive), or downcasting quantization (lossy). Streaming individual MoE expert weights from SSD on demand exploits the structural sparsity of MoE — only K-of-N experts activate per token — to keep the active-set in VRAM and amortize disk reads via OS page cache + a frequency-weighted LRU.

This is the same technique [Flash-MoE](https://github.com/danveloper/flash-moe) demonstrated for Apple Silicon and CUDA in standalone engines; this project ports it as a **buffer-type extension to mainline llama.cpp**, so any backend that registers a buffer-type provider can plug in.

## Hardware

Tested on:

- **AMD Instinct MI300X** (gfx942, 192 GB HBM3, ROCm 7.0+)
- **Storage**: NVMe SSD with ≥ 5 TB free (scratch volume)

The codepath is HIP-guarded for gfx942-specific tunings but the buffer-type layer is backend-agnostic — patches welcome for CDNA1/2 (gfx90a), RDNA, NVIDIA.

## Headline numbers (MI300X, ROCm 7.0)

| Model                          | Size    | Fits VRAM? | Cache  | tg/s  |
| ------------------------------ | ------: | ---------- | -----: | ----: |
| Qwen3-30B-A3B Q4_K_M (baseline)|  17 GiB | yes        |   —    | 169.7 |
| Qwen3-30B-A3B Q4_K_M (slab)    |  17 GiB | yes        | 8 GiB  |  29.7 |
| Qwen3-30B-A3B Q4_K_M (fused)   |  17 GiB | yes        | 8 GiB  |  34.0 |
| DeepSeek-V3 671B Q4_K_M        | 377 GiB | **no**     | 128 GiB| TBD   |

Streaming costs ~5× on a model that fits VRAM (the dispatch overhead dominates when no SSD reads happen). The actual win is **enabling models that don't fit at all** — see DeepSeek-V3 row.

## Prior art

- [vLLM RFC #38256](https://github.com/vllm-project/vllm/issues/38256) — `ExpertWeightProvider` abstraction (Apache 2.0). Conceptual relative; we land in llama.cpp instead.
- [tinyserve](https://github.com/e1n00r/tinyserve) by `e1n00r` — independent reference impl on RTX 2000.
- [ktransformers](https://github.com/kvcache-ai/ktransformers) — earlier CPU-offload MoE inference.
- [llama.cpp `--n-cpu-moe`](https://github.com/ggml-org/llama.cpp) (PR #11397, MIT) — same `tensor_buft_overrides` shape we mirror.
- [AITER](https://github.com/ROCm/aiter) (MIT) — CDNA3 fused-kernel reference.

## Quick start

```bash
# 1. Build llama.cpp on the streaming branch
./scripts/build_llamacpp.sh        # ~10 min on MI300X

# 2. Pack a model's MoE expert tensors into per-layer SSD files (one-time)
./scripts/pack_model.sh /path/to/DeepSeek-V3-Q4_K_M-00001-of-00009.gguf /mnt/scratch/moe-stream-dsv3/

# 3. Start the streaming server
./scripts/run_server.sh /path/to/DeepSeek-V3-Q4_K_M-00001-of-00009.gguf /mnt/scratch/moe-stream-dsv3/ 131072 8080

# 4. Run the agent demo
pip install -e .
agent-demo --task "What is the size in bytes of /etc/hostname?" --base-url http://localhost:8080/v1
```

## License

MIT. See [LICENSE](LICENSE).
