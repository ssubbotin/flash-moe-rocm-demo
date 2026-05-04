# Architecture

The streaming path is implemented as a CUDA/HIP **buffer-type extension** to ggml, plus a **custom MUL_MAT_ID dispatch hook** in the CUDA backend. Everything else in llama.cpp is unchanged.

## Component map

```
                 ┌───────────────────────────────────────┐
common/arg.cpp ──┤ --moe-stream-dir DIR --moe-cache-mb N │
                 └────────────────────┬──────────────────┘
                                      │ tensor_buft_overrides push
                                      ▼
            ┌───────────────────────────────────────────────────┐
            │  ggml_backend_cuda_buffer_type_moe_stream         │
            │  (singleton per (device, ssd_dir))                │
            │                                                   │
            │  init_tensor() parses 'blk.<L>.ffn_<gate|up|down> │
            │    _exps.weight'; opens /<DIR>/layer_<NN>.<P>.bin │
            │    and stores meta { layer_id, projection_id,     │
            │    num_experts, expert_bytes, fd }.               │
            │  set_tensor() is a no-op (data lives on disk).    │
            │  alloc_buffer() hands out pseudo-pointers (never  │
            │    dereferenced on host).                         │
            │                                                   │
            │  Lazy: ggml_moe_cache_init() on first dispatch    │
            │    using max(expert_bytes) across all init_tensor │
            │    calls (handles mixed Q4_K + Q6_K MoEs).        │
            └────────────────────┬──────────────────────────────┘
                                 │
                                 │ MUL_MAT_ID on src0 in this buft
                                 ▼
            ┌───────────────────────────────────────────────────┐
            │  ggml_cuda_supports_op(MUL_MAT_ID) → claim it     │
            │  ggml_cuda_mul_mat_id() → dispatcher hook         │
            │                                                   │
            │  ggml_cuda_moe_stream_mul_mat_id(...)             │
            │    1. memcpy ids host-side (small)                │
            │    2. unique-experts set                          │
            │    3. for each unique expert:                     │
            │         ggml_moe_cache_get(cache, key, expert_id) │
            │       cache hit  → device pointer in pool slot    │
            │       cache miss → dispatching_miss():            │
            │           moe_stream_miss():                      │
            │             memcpy(pinned_staging,                │
            │                    mmap_base + e * stride, ...);  │
            │             cudaMemcpyAsync H2D + sync.           │
            │           OS page cache absorbs warm reads;       │
            │           cold reads pread from SSD.              │
            │    4. compute path:                               │
            │       slab    → D2D copy each unique to ctx.pool  │
            │                 slab; call mul_mat_vec_q          │
            │       fused   → mul_mat_id_q4k_scattered_amd      │
            │                 reads weights direct from cache   │
            │                 slots via device pointer table    │
            └───────────────────────────────────────────────────┘
```

## Three tiers in numbers (Qwen3-30B-A3B Q4_K_M, MI300X)

| Tier            | Storage         | Latency / 8 MiB expert | Bandwidth target    |
| --------------- | --------------- | ---------------------- | ------------------- |
| 0. VRAM LRU     | HBM3            | < 100 µs (D2D copy)    | 5300 GB/s peak HBM3 |
| 1. Page cache   | DDR             | ~ 1 ms                 | ~ 50 GB/s DDR       |
| 2. SSD pread    | NVMe            | ~ 2 ms                 | ~ 4 GB/s NVMe       |

A "miss" walks tier 0 → 1 → 2; tier 1 vs 2 is invisible to our code (it's `mmap` + `memcpy`, the kernel decides whether the page is resident).

## Why a custom buffer type and not a graph-level rewrite

Three options were considered:

1. **Hook in `mmid.cu` inner kernels** — would need to rewrite four kernels (`mmvq`, `mmq`, `mmvf`, `mmf`) to accept K scattered pointers instead of a single slab. Touches the hottest code in the backend; high risk; doesn't fit upstream review velocity.
2. **Slab-staging buffer type** — keep upstream kernels, present a single contiguous slab built from the LRU pool. Doesn't solve the "model bigger than VRAM" case (the slab still has to hold all experts).
3. **Buffer-type + custom dispatch** (this design) — buffer type is the integration surface; CUDA backend's `supports_op` claims `MUL_MAT_ID` for tensors in this buffer; the dispatch hook does cache lookup + compute. Kernels stay unchanged.

(3) is what landed. It mirrors the existing `--n-cpu-moe` shape (also a `tensor_buft_overrides` regex push), so upstream reviewers have a precedent.

## Why the slab fallback path

The fused (scattered-pointer) kernel handles Q4_K only. Mixed-quant MoEs (e.g. Qwen3 is Q4_K gate/up + Q6_K down) need the slab path for the Q6_K projection. The slab path also exists for fall-through correctness when the fused path's narrow shape constraints (`n_tokens ≤ MMVQ_MAX_BATCH_SIZE`, `n_dim_in % QK_K == 0`) aren't met.

## Known limitations

- **CUDA/HIP only.** SYCL / Metal / Vulkan would each need their own buffer-type provider. The Phase 1 LRU core is backend-agnostic, so the lift is just the dispatch hook.
- **Single-stream sync inside the dispatcher.** `cudaStreamSynchronize` is illegal during HIP/CUDA graph capture; we mark moe-stream graphs as ineligible for capture (commit `212094e`). On long-context decode where capture would otherwise help, this costs ~1–2 % overhead.
- **Cache key 32-bit `(layer<<2 | projection)`.** Up to ~250 MoE layers — fine for everything published as of 2026.
- **No prefetch.** A natural follow-up: hint the cache during routing softmax (which knows the upcoming expert ids) to start cold reads in the background. The `ggml_moe_cache_params::prefetch_lookahead` field was reserved for this in Phase 1, dropped in housekeeping; reintroduce when we wire it.
