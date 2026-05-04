# Benchmarks

All numbers from a single AMD Instinct MI300X VF (192 GB HBM3, ROCm 7.0) on a DigitalOcean GPU droplet.

## Qwen3-30B-A3B Q4_K_M (17 GiB — fits VRAM)

`llama-bench -m model.gguf -ngl 99 -t 8 -p 256 -n 64`

| Configuration            | pp256 (t/s)      | tg64 (t/s)     | Notes                                     |
| ------------------------ | ---------------: | -------------: | ----------------------------------------- |
| baseline (stock)         | 4549.60 ± 21.60  | 169.66 ± 1.19  | All experts in VRAM, no streaming overhead |
| `--moe-stream-dir` slab  |  102.06 ± 13.84  |  29.71 ± 2.36  | 8 GiB cache, slab D2D per dispatch         |
| `--moe-stream-dir` fused |  132.86 ±  4.05  |  34.04 ± 1.56  | 8 GiB cache, MOE_STREAM_FUSED=1            |

**Reading**: when the model fits VRAM, streaming is **5× slower** than baseline. The cache adds dispatch overhead; the SSD never gets touched. **This is expected** — 30B is an architectural smoke test, not the use case.

The fused scattered-pointer kernel is **+14.6 % over slab** (34.0 vs 29.7), validating that the per-dispatch slab D2D copy is real cost.

## DeepSeek-V3 671B Q4_K_M (377 GiB — does NOT fit VRAM)

`llama-bench -m DeepSeek-V3-Q4_K_M-00001-of-00009.gguf -ngl 99 --moe-stream-dir … --moe-cache-mb 131072 -p 64 -n 32`

| Configuration            | pp64 (t/s)       | tg32 (t/s)     | Cache hit rate | Notes                            |
| ------------------------ | ---------------: | -------------: | -------------: | -------------------------------- |
| baseline (stock)         | OOM              | OOM            | n/a            | 377 GiB > 192 GiB VRAM           |
| `--moe-stream-dir` slab  | TBD              | TBD            | TBD            | 128 GiB cache                    |
| `--moe-stream-dir` fused | TBD              | TBD            | TBD            | 128 GiB cache, MOE_STREAM_FUSED=1|

**Reading**: this is the headline. DeepSeek-V3 simply cannot run on a single MI300X (or any single accessible GPU below the MI325X 256 GiB tier) without streaming. Numbers added when the run completes.

## Methodology

- llama.cpp branch `feature/moe-expert-gpu-cache` at the bench commit (`d611ba5`).
- `cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx942 -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-bench`.
- Each bench is one llama-bench invocation (`-r 1` for the DSV3 row to bound runtime; the 30B rows are llama-bench's default `-r 5`).
- pp/tg are llama-bench's standard prompt-processing / token-generation steady-state numbers.
- DSV3 cache size 128 GiB chosen so the cache fits with ~10 GiB non-expert weights + ~5 GiB compute buffers, leaving ~50 GiB headroom for kernel scratch and async copies.

## Where each tier helps

| Workload                                              | Helps?           |
| ----------------------------------------------------- | ---------------- |
| Model >> VRAM (DSV3, future 1T+)                      | Streaming enabled (no other option) |
| Model fits VRAM, low concurrency                      | Streaming hurts (dispatch overhead) — use baseline |
| Model fits VRAM, very high concurrency / batched      | Slab fusion potentially helps — TODO measure       |
| Model partially fits VRAM (e.g. Qwen3-235B in 192GiB) | Streaming wins because the alternative is CPU offload at GB/s, not HBM at TB/s |
