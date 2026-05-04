# Demo video storyboard (≤ 5 min for hackathon submission)

Total: 4:30. Cut for 30 s of headroom.

## 0:00 – 0:20 — Title card

Static slide:

> **Flash-MoE-ROCm-Demo**
> Run **DeepSeek-V3 671B** on a **single MI300X**
> via streaming MoE expert weights from SSD
> AMD Developer Hackathon 2026

Voiceover: *"This is DeepSeek-V3, a 671 billion parameter mixture-of-experts model, decoding live on a single AMD MI300X — a card with 192 gigabytes of HBM, less than half the model's 377 gigabyte size. Stock llama.cpp can't load it. Here's how we made it run."*

## 0:20 – 1:00 — The problem + the idea

Cut to terminal showing failed stock load:

```bash
$ build/bin/llama-server -m DeepSeek-V3-Q4_K_M-00001-of-00009.gguf -ngl 99
ggml_backend_alloc_ctx_tensors_from_buft: failed to allocate ROCm0 buffer of
size 187543... (OOM at 192 GiB VRAM, model needs 377 GiB)
```

Voiceover: *"MoE models activate only K-of-N experts per token — DeepSeek-V3 uses 8 of 256 experts per layer. The other 248 sit idle. So we leave the cold experts on NVMe SSD and stream them into a small VRAM cache only when they're needed."*

Cut to a simple architecture diagram (the one in `docs/architecture.md`).

## 1:00 – 2:00 — How it works (in code, briefly)

Cut to the moe-pack tool running:

```bash
$ ./scripts/pack_model.sh DeepSeek-V3-Q4_K_M-00001-of-00009.gguf /mnt/scratch/moe-stream-dsv3/
[pack] detected sharded GGUF prefix 'DeepSeek-V3-Q4_K_M', running on all shards
... (parallel xargs runs over 9 shards) ...
[pack] done — 174 files in /mnt/scratch/moe-stream-dsv3, 367G total
```

Voiceover: *"First we convert the GGUF expert tensors into per-layer files for fast random-access reads. This is a one-time setup."*

Cut to llama-server starting with the new flag:

```bash
$ MOE_STREAM_FUSED=1 build/bin/llama-server \
    -m DeepSeek-V3-Q4_K_M-00001-of-00009.gguf -ngl 99 \
    --moe-stream-dir /mnt/scratch/moe-stream-dsv3/ \
    --moe-cache-mb 131072 --port 8080
...
moe-stream: LRU cache initialized (pool=131072 MiB, slot_bytes=12042240, device=0)
main: server is listening on http://127.0.0.1:8080
```

Voiceover: *"At runtime, llama.cpp routes the expert tensors through our custom buffer type. The cache is a 128 GiB on-device LRU; cache misses fall through to the OS page cache and ultimately to SSD pread."*

## 2:00 – 3:30 — Live inference

Split screen: terminal on the left, `rocm-smi --showmeminfo vram` running every second on the right.

Curl the chat completions endpoint:

```bash
$ curl -s http://localhost:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"In one sentence: capital of France and one fact about it?"}],"max_tokens":50}'
```

Show the response stream in (or replay the captured one):

> The capital of France is Paris, and an interesting fact about it is that the city is home to the iconic Eiffel Tower, which was the tallest man-made structure in the world when it was completed in 1889.

VRAM panel shows ~150 GB used, climbing as more experts cache, then settling.

Voiceover: *"Live inference on a model that wouldn't even load before. Decode runs at 0.68 tokens per second — slow by datacenter standards, but unblocks workloads that were physically impossible on this hardware tier."*

## 3:30 – 4:15 — Bench numbers + the agent

Cut to `llama-bench` table on Qwen3-30B (fast model, demonstrates the agent loop is interactive):

```
| baseline (stock) |  170.78 tg/s |
| stream slab      |   29.71 tg/s |
| stream fused     |   34.04 tg/s |
```

Cut to agent demo:

```bash
$ agent-demo --task "Run uptime and tell me how long the system has been up"
The system has been up for 1 day, 5 hours, and 47 minutes...
```

Voiceover: *"On a model that fits VRAM, streaming costs about 5x — the cache adds dispatch overhead. But the OpenAI-compatible chat endpoint and tool-calling protocol both work end-to-end. Here's a small Python agent calling shell commands through our streaming server."*

## 4:15 – 4:30 — Close

Static slide:

> Code: github.com/ssubbotin/flash-moe-rocm-demo
> Upstream PR (draft): github.com/ggml-org/llama.cpp/pull/<NUMBER>
> MIT license. AMD Developer Hackathon 2026.

Voiceover: *"Code is MIT-licensed. The llama.cpp upstream PR is draft and looking for review on the buffer-type ergonomics. Thanks to the AMD Developer Cloud team for the MI300X access."*

---

## Capture commands

For the moe-pack timing shot:
```bash
ssh mi300 "screen -ls; cat /tmp/pack-dsv3.log"
```

For VRAM panel:
```bash
ssh mi300 "watch -n 1 'rocm-smi --showmeminfo vram | head -5'"
```

For inference shot:
```bash
ssh mi300 "tail -f /tmp/dsv3-server.log"
```

(Pre-record each terminal pane separately; composite in OBS or an editor.)
