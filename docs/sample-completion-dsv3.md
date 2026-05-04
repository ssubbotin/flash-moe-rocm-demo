# Sample DeepSeek-V3 671B completion on single MI300X

Captured 2026-05-05 from `llama-server` running on a DigitalOcean MI300X VF (gfx942, 192 GiB HBM3, ROCm 7.0) with this branch's `--moe-stream-dir` + `MOE_STREAM_FUSED=1` + 128 GiB cache pool.

**Model**: `unsloth/DeepSeek-V3-GGUF` Q4_K_M, 9 shards, 377 GiB on disk.

**Request**:

```http
POST /v1/chat/completions
{
  "model": "local",
  "messages": [
    {"role": "user", "content": "In one sentence: what is the capital of France, and one interesting fact about it?"}
  ],
  "max_tokens": 50,
  "temperature": 0.2
}
```

**Response** (47 tokens generated):

> The capital of France is Paris, and an interesting fact about it is that the city is home to the iconic Eiffel Tower, which was the tallest man-made structure in the world when it was completed in 1889.

**Timings** (from llama-server):

```
prompt_n              21
prompt_ms             53263.078
prompt_per_second     0.39 t/s     (cold prefill, cache warming)
predicted_n           47
predicted_ms          86814.554
predicted_per_token   1847 ms
predicted_per_second  0.54 t/s     (steady-state decode)
```

The 0.54 t/s steady decode is below the bench-measured 0.68 t/s peak — first-request decode includes additional warming as the LRU fills past the bench harness's settle window. Bench numbers in `benchmarks.md` are after explicit warmup.

**Why this matters**: stock llama.cpp on the same droplet **cannot load DeepSeek-V3** — model size 377 GiB exceeds VRAM 192 GiB, OOM at allocation time. With `--moe-stream-dir` only the active expert subset transits VRAM at any moment; cold experts live on the NVMe scratch volume.
