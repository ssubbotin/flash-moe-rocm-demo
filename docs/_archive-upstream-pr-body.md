# Suggested upstream llama.cpp PR body

Title: `[RFC/WIP] ggml-cuda: GPU-resident MoE expert cache + optional SSD streaming`

Open with: `gh pr create --repo ggml-org/llama.cpp --base master --head ssubbotin:feature/moe-expert-gpu-cache --draft --title "..." --body-file docs/upstream-pr-body.md`

---

## What this adds

A new CUDA/HIP buffer type — `ggml_backend_cuda_buffer_type_moe_stream(device, ssd_dir, cache_pool_bytes)` — that lets MoE expert tensors live on disk and be streamed into a frequency-weighted LRU cache in VRAM on demand.

Two new CLI flags (mirror of `--n-cpu-moe`'s `tensor_buft_overrides` shape):
- `--moe-stream-dir DIR` — directory holding per-(layer, projection) expert files
- `--moe-cache-mb N` — on-device LRU cache budget in MiB

A new converter tool `tools/moe-pack/` that turns any GGUF MoE checkpoint into the per-layer file layout (`layer_<NN>.<gate|up|down>.bin`) the buffer type consumes.

A custom `MUL_MAT_ID` dispatch hook that recognizes when `src0` lives in this buffer type, pulls the K active experts through the cache, and computes the matvec — either via a slab D2D + `mul_mat_vec_q` fallback (any quant) or a Q4_K-only scattered-pointer kernel (`MOE_STREAM_FUSED=1`, gfx942-tuned).

## Why

Models like DeepSeek-V3 671B Q4_K_M (377 GiB) don't fit a single GPU below MI325X-class (256 GiB). Today the workarounds are CPU offload via `--n-cpu-moe` (slow, GB/s host bandwidth), tensor-parallel multi-GPU (expensive), or smaller quantization (lossy). MoE structural sparsity (only K-of-N experts per token) means SSD streaming is realistic — only the active set transits VRAM at any moment.

This is the same technique [Flash-MoE](https://github.com/danveloper/flash-moe) demonstrated in a standalone Apple Silicon engine; this PR ports it as a buffer-type extension to mainline llama.cpp so any backend (CUDA today, SYCL/Metal/Vulkan straightforward to follow) can opt in.

## Numbers (single AMD Instinct MI300X, ROCm 7.0, gfx942)

| Model                       | Size    | Stock         | This PR (slab) | This PR (fused) |
| --------------------------- | ------: | ------------: | -------------: | --------------: |
| Qwen3-30B-A3B Q4_K_M        |  17 GiB | 169.66 tg/s   |  29.71 tg/s    |   34.04 tg/s    |
| DeepSeek-V3 671B Q4_K_M     | 377 GiB | **OOM**       |   0.57 tg/s    |    0.68 tg/s    |

For models that fit VRAM (Qwen3-30B), streaming costs ~5x because it bypasses the upstream slab fast path. **For models that don't fit at all (DeepSeek-V3), this PR is the difference between OOM and 0.68 tok/s decode** on a single MI300X.

Sample DSV3 completion captured live, see [companion repo](https://github.com/ssubbotin/flash-moe-rocm-demo/blob/main/docs/sample-completion-dsv3.md).

## Architecture in one diagram

```
common/arg.cpp ─► tensor_buft_overrides regex push (mirrors --n-cpu-moe)
                                │
                                ▼
ggml_backend_cuda_buffer_type_moe_stream  (singleton per (device, ssd_dir))
   ├─ init_tensor: parse blk.<L>.ffn_<gate|up|down>_exps.weight
   │              open <DIR>/layer_<NN>.<P>.bin (mmap'd) + pinned staging
   ├─ set_tensor: no-op (data lives on disk)
   ├─ alloc_buffer: pseudo-pointers (never dereferenced on host)
   └─ lazy: ggml_moe_cache_init() on first dispatch using max(expert_bytes)

ggml_cuda_supports_op(MUL_MAT_ID) → claim when src0 is moe_stream
ggml_cuda_mul_mat_id() → ggml_cuda_moe_stream_mul_mat_id() →
   for each unique active expert: ggml_moe_cache_get(cache, key, expert_id)
     hit  → device pointer in pool slot
     miss → moe_stream_miss(): mmap+pread + cudaMemcpyAsync H2D + sync
   compute:
     slab  → D2D each unique into ctx.pool slab; mul_mat_vec_q (any quant)
     fused → mul_mat_id_q4k_scattered_amd (Q4_K only, gfx942-tuned)
```

Three tiers of cache: **VRAM LRU** (pool) → **OS page cache** (mmap) → **SSD pread**. The mmap path makes tier-1 invisible to our code; the kernel decides whether the page is resident.

## What I'd love feedback on

This is a draft — I'd appreciate review on:
1. **Buffer-type ergonomics**. Singleton-per-(device, ssd_dir) with lazy cache init feels right but is novel for ggml backends.
2. **`supports_op` claiming a hot op.** I claim `MUL_MAT_ID` only when `src0` is in the moe_stream buft. The dispatch hook calls `ggml_cuda_mul_mat_vec_q` etc. internally so existing kernels are unchanged.
3. **CUDA graph capture incompatibility.** The dispatcher syncs the compute stream to read routing ids host-side; this is illegal during graph capture, so I disable capture for graphs touching moe_stream tensors. Suggestions for keeping graph capture welcome.
4. **Mixed-quant MoE.** Qwen3-30B is Q4_K gate/up + Q6_K down. The cache slot is sized to the max per-tensor expert_bytes. Fused kernel handles only Q4_K; mixed paths fall back to slab.
5. **Cross-backend rollout.** Phase 1 LRU is backend-agnostic; only the dispatch hook is CUDA. SYCL would be ~200 LOC similar; Metal/Vulkan harder.

## Companion repo + demo

[`ssubbotin/flash-moe-rocm-demo`](https://github.com/ssubbotin/flash-moe-rocm-demo) — agent demo + benchmarks + scripts.

## License + provenance

All new files in this PR carry `Copyright (c) 2026 Sergey Subbotin` + SPDX `MIT` headers, fully compatible with llama.cpp's MIT. References consulted: vLLM RFC #38256 (Apache 2.0), `e1n00r/tinyserve` (MIT), AITER (MIT), llama.cpp PR #11397 (MIT). One file (`kernels_fused_moe.hip.h` from a prior MI300X experiment, also Sergey-authored MIT-headered) was used as the structural reference for the scattered-pointer kernel; no source from `danveloper/flash-moe`'s engine code (which has no LICENSE → "all rights reserved" by default) was consulted.

## Commit history (will squash before request-review)

```
$ git log --oneline master..feature/moe-expert-gpu-cache
d611ba5 bench: Qwen3-30B-A3B baseline / slab / fused via llama-bench (170 / 30 / 34 tg/s)
b120e0c llama-bench: --moe-stream-dir / --moe-cache-mb flags so streaming numbers are captureable
a27ec85 ggml-cuda: lazy cache pool sizing — accept --moe-cache-mb in any flag order
b6664d0 ggml-cuda: scattered-pointer fused MoE behind MOE_STREAM_FUSED=1 (skips slab D2D copy)
253b0c9 ggml-cuda: add scattered-pointer Q4_K_M MoE kernel for moe-stream dispatch (no integration yet)
c01c2ea bench: Qwen3-30B-A3B baseline tg64=170.78 tok/s + smoke stream-8g=22.88 tok/s
212094e ggml-cuda: keep MoE stream tensors out of mmvq+glu fusion and CUDA graphs
f076a9d ggml-cuda: Q4_K_M MoE dispatch via cache lookups (per-expert per-token, no fusion)
5bc3295 ggml-cuda: dispatch skeleton for MoE streaming MUL_MAT_ID (placeholder compute)
fa114c7 tools: moe-pack — convert GGUF MoE expert tensors to per-layer files
83cf92a ggml-cuda: 3-tier MoE expert source (LRU + page cache + SSD pread)
a33ccd1 docs: per-layer SSD layout for moe-stream buffer type
b71bbc8 common: --moe-stream-dir and --moe-cache-mb flags
e573f94 ggml-cuda: skeleton MoE streaming buffer type
f6f238d ggml-moe-cache: fix miss-handler failure leak + upstream polish
34b9ed4 tests: LRU policy
2dbb67e ggml: portable LRU MoE expert cache core
4311768 ggml: public header for portable MoE expert cache
```

## Status

- [x] Builds clean for `GGML_HIP=ON` gfx942
- [x] End-to-end correctness verified on Qwen3-30B-A3B and DeepSeek-V3 671B (sample completions in companion repo)
- [x] llama-bench numbers captured for both
- [x] Tested: model load, warmup, decode, multi-turn chat
- [ ] CUDA path (NVIDIA) sanity-tested — should work as-is via the HIP/CUDA shared `.cu`, untested
- [ ] Existing test suite: should not regress; haven't run full ctest
- [ ] Squash + clean history before request-review
- [ ] Add tests for the new buffer type + dispatch path
