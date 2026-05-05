# flash-moe vs llama.cpp on the same model — apples-to-apples

A direct comparison on **Qwen3.5-397B-A17B** running on a single AMD Instinct MI300X (gfx942, 192 GiB HBM3, ROCm 7.0 / 7.2):

| Engine                       | Format           | Approach                       | Decode tg/s              |
| ---------------------------- | ---------------- | ------------------------------ | -----------------------: |
| **llama.cpp baseline**       | Q4_K_M GGUF      | full model in VRAM             | **OOM** (227 GiB > 192 GiB) |
| **llama.cpp + this PR**, slab, 128 GiB cache    | Q4_K_M GGUF | streaming MoE buffer-type | **5.62**            |
| **llama.cpp + this PR**, fused, 128 GiB cache   | Q4_K_M GGUF | streaming + scattered kernel | 5.41                  |
| **llama.cpp + this PR**, fused, 160 GiB cache   | Q4_K_M GGUF | streaming + bigger cache | 5.49                       |
| **flash-moe `rocm_infer`** baseline       | 4-bit MLX safetensors | standalone engine, custom kernels | **7.13** (warm, 30 t)        |
| **flash-moe `rocm_infer`** all opts (FLA-GDN + fused MoE) | 4-bit MLX safetensors | + scattered + Triton-style | **10.01** (warm, 100 t) |
| flash-moe (Apple M3 Max, original)        | 4-bit MLX safetensors | unified-memory streaming | 4.36                  |

(All MI300X numbers from `llama-bench` `-p 64 -n 32` for streaming configs; flash-moe numbers from the [project_mi300_optimization.md](https://github.com/ssubbotin/flash-moe-rocm-demo/blob/main/docs/) memory we logged on 2026-05-04.)

## What this tells us

- **The streaming buffer-type approach gets within ~79 % of flash-moe's hand-tuned standalone baseline** (5.62 / 7.13) on the exact same hardware and the exact same model (just different on-disk encoding: GGUF Q4_K_M vs MLX 4-bit).
- **Against flash-moe's fully-optimised stack** (10.01 tg/s with FLA-GDN + custom fused-MoE kernel), the gap is ~56 %. The 4.4 tg/s delta is the cost of riding inside a general-purpose engine (llama.cpp graph + sched + MUL_MAT_ID dispatch + Q4_K_M dequant) instead of a dedicated layer-loop.
- **Both engines beat the original Apple M3 Max number** (4.36 tg/s); MI300X memory bandwidth + 192 GiB HBM3 cache budget is the decisive hardware change.

## Where the gap actually comes from

Approximate decomposition of the 5.62 → 10.01 gap:

| Source | Estimated contribution |
| ----------------------- | ---------------------: |
| Q4_K_M dequant overhead (super-block + 8 sub-blocks + packed 6-bit scales/mins) vs MLX 4-bit (single nibble + bf16 scale + bf16 bias per group of 64) | ~30–40 % |
| llama.cpp scheduler / graph-compute overhead per op vs flash-moe's hardcoded layer loop | ~15–25 % |
| Cautious-by-default streaming dispatcher (extra `cudaStreamSynchronize` for ids fetch) vs flash-moe's pipelined deferred CMD3 | ~10–15 % |
| Conservative cache slot sizing for mixed-quant MoE (we size for `max(expert_bytes)` to cover Q6_K down) | ~5–10 % |

None of these are kernel-level — the fused scattered kernel itself competes well. The gap is integration-tax for being inside an engine that supports many models and many backends.

## Why slab > fused on Qwen3.5-397B (and the opposite on DSV3)

The 397B's per-expert is ~2.5 MB (gate/up) and ~2.9 MB (down). DSV3's is ~8.3 MB (gate/up) and ~12 MB (down) — almost 5× larger.

- **slab path** does one D2D copy per unique expert per dispatch. At 2.5 MB per expert × ~16 unique per layer × 60 layers = ~2.4 GB of D2D per token. Cheap on HBM3 (5.3 TB/s).
- **fused path** skips the slab copy but issues one HIP block per (token × slot × output-row-tile). For small experts the per-block launch + LDS staging overhead is a larger fraction of useful work.

So **slab wins on small experts; fused wins on large experts**. Practical guidance:
- Q4_K_M with `intermediate_size ≤ 1024`: slab (`MOE_STREAM_FUSED=0`)
- Q4_K_M with `intermediate_size ≥ 4096`: fused (`MOE_STREAM_FUSED=1`)
- DeepSeek-V3 (intermediate 18432): fused gives +19 %
- Qwen3.5-397B (intermediate 1024): slab gives +4 %

The right long-term fix is auto-selecting based on `expert_bytes` at cache-init time — small follow-up commit.

## What this means for "flash-moe in mainline"

The original flash-moe achievement was: **397B parameter model, 4-bit MLX, on a MacBook Pro at 4.4 tg/s using SSD streaming**. That work was a standalone engine optimised for one model on one platform.

This PR's claim is narrower and broader at the same time: **the same SSD-streaming MoE technique, available to every llama.cpp user on every supported backend, on every MoE GGUF**. The numbers are lower per-token than flash-moe's hand-tuned engine on the same hardware, but the technique now reaches ~10 000× more users via llama.cpp's ecosystem (mobile, desktop, server, CLI tools, Hugging Face, LM Studio, lemonade, Ollama, koboldcpp, etc. — all of which consume llama.cpp underneath).

Trade-off explicitly: peak per-token latency for ecosystem reach.
