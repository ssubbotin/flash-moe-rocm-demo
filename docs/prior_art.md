# Prior art and credits

This work would not exist without:

- **vLLM RFC #38256** — *Incremental MoE Expert Offloading — GPU Cache + Async Pipeline* (Apache 2.0). The `ExpertWeightProvider` abstraction inspired the buffer-type-as-provider design here.
- **e1n00r/tinyserve** (MIT) — production-grade reference implementation of CPU-offloaded MoE expert caching for vLLM. Showed that 97-100 % cache hit rates are achievable with a frequency-weighted LRU on real workloads.
- **kvcache-ai/ktransformers** (Apache 2.0) — earlier ROCm/CPU MoE expert offload work. Established `--n-cpu-moe`-style override pattern that mainline llama.cpp adopted.
- **llama.cpp PR #11397** (MIT) — `--override-tensor` mechanism that we mirror via `tensor_buft_overrides`. Without this, integrating a new buffer type would have required core scheduler changes.
- **AITER** by ROCm (MIT) — gfx942 fused-kernel patterns that informed the scattered-pointer kernel here.
- **danveloper/flash-moe** — pioneering Apple Silicon + CUDA MoE-on-SSD inference engine. Demonstrated the basic feasibility on a MacBook M3 Max running Qwen3.5-397B-A17B at 4.4 tok/s. This llama.cpp PR is an independent re-derivation of those techniques (no source-code reuse from that repo's MIT-unspecified files); the only file reused is one with an explicit `Sergey Subbotin` MIT-able copyright header.
- **HuggingFace `transformers`** — Apache 2.0 reference implementation for Qwen3 and DeepSeek-V3 architectures. The model-side semantics in llama.cpp ultimately trace back to these implementations.

## Differences from the cited prior art

- vLLM's `ExpertWeightProvider` is Python-side; this work integrates at the C++/HIP backend level so it benefits any llama.cpp consumer, including embedded users (mobile, server-side, CLI tools).
- tinyserve targets CPU-offload; this work targets SSD-streaming, which is a strict superset (CPU pinned memory is just the OS page cache from our perspective).
- ktransformers ships in Python with a custom dispatch loop; we ship in C++ as a backend extension that doesn't touch the model graph.
- AITER's fused kernels are written for AITER's own scheduling assumptions; the kernel here adapts the same shape primitives to llama.cpp's `MUL_MAT_ID` interface.
- Flash-MoE is a standalone engine; this work is an extension to a mainline engine.
