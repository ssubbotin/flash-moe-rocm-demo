# Wakeup summary — 2026-05-05 ~01:35 CEST

You went to sleep with Phase 4 sub-commit 3 (fused kernel) verified at +12% over slab on warm Qwen3-30B; we'd identified path 1 (housekeeping) and path 2 (DSV3 demo) as next.

## What landed while you slept

**Path 1 (housekeeping) — DONE.** Two commits on `feature/moe-expert-gpu-cache`:
- `a27ec85` ggml-cuda: lazy cache pool sizing — `--moe-cache-mb` and `--moe-stream-dir` accept any flag order now
- `b120e0c` llama-bench: `--moe-stream-dir` / `--moe-cache-mb` flag plumbing so we can capture clean streaming numbers
- `d611ba5` bench commit: Qwen3-30B-A3B baseline 169.66 / slab 29.71 / fused 34.04 tg/s via llama-bench

**Path 2 (DSV3 demo) — DONE.** Headline number captured:
- Downloaded `unsloth/DeepSeek-V3-GGUF Q4_K_M` (377 GiB, 9 shards) in ~3 min
- Packed via moe-pack to per-layer files (367 GiB, 174 files) in ~1m40s
- Bench (`d611ba5`):
  - **slab pp64=0.80, tg32=0.57 t/s**
  - **fused pp64=0.79, tg32=0.68 t/s** (+19% over slab)
  - **baseline = OOM** (377 GiB > 192 GiB VRAM)
- Live chat completion captured: *"The capital of France is Paris, and an interesting fact about it is that the city is home to the iconic Eiffel Tower, which was the tallest man-made structure in the world when it was completed in 1889."* (47 tokens, 0.54 tok/s steady, captured at 2026-05-05 ~01:13)

**Phase 5 (companion repo + agent + submission) — mostly done.** New public repo:
- `https://github.com/ssubbotin/flash-moe-rocm-demo` (MIT, 6 commits)
- README.md with the headline numbers
- src/agent_demo/ (Python, OpenAI SDK, 3 tools: read_file/run_shell/http_get)
- scripts/{build_llamacpp,pack_model,run_server}.sh
- docs/{architecture,benchmarks,prior_art,sample-completion-dsv3,agent-demo-transcript,upstream-pr-body,demo-video-storyboard}.md
- Agent verified end-to-end against Qwen3-30B server: direct answer ("7×8=56") + tool call ("uptime → 1d 5h 47m")

**Upstream branch — pushed to fork.** `https://github.com/ssubbotin/llama.cpp/tree/feature/moe-expert-gpu-cache`. PR not yet opened to ggml-org (that's a user-shared-state action; left for your confirmation).

## What needs you specifically

1. **Open the upstream draft PR** — body fully drafted at `flash-moe-rocm-demo/docs/upstream-pr-body.md`. One command:
   ```bash
   cd ~/llamacpp-moe-cache && gh pr create --repo ggml-org/llama.cpp --base master \
     --head ssubbotin:feature/moe-expert-gpu-cache --draft \
     --title "[RFC/WIP] ggml-cuda: GPU-resident MoE expert cache + optional SSD streaming" \
     --body-file /home/sergey/flash-moe-rocm-demo/docs/upstream-pr-body.md
   ```

2. **Record the demo video** — storyboard at `flash-moe-rocm-demo/docs/demo-video-storyboard.md`. ~4:30 of capture, OBS or similar. Captures needed from droplet (model load, bench output, chat completion, agent run).

3. **Submit on lablab.ai** — fields needed:
   - Project: flash-moe-rocm-demo
   - Repo URL: https://github.com/ssubbotin/flash-moe-rocm-demo
   - Upstream PR URL (after step 1)
   - Demo video URL (after step 2)
   - License: MIT
   - Description: 1 paragraph from README

## Droplet state

- ssh mi300 alive (165.245.140.116)
- VRAM clean (~300 MB), no llama-server running
- /mnt/scratch 1.2 TB used (DSV3 GGUF + DSV3 pack + Qwen3-30B GGUF + Qwen3-30B pack + small test models). 3.6 TB free.
- Build artifacts in `/root/llamacpp-moe-cache/build/` — survive shutdown only if VM persists
- AMD credits expire 2026-05-05 (today). Check console for status; switch to DigitalOcean ($1.99/hr) if needed.

## Numbers banked

| Workload | tg/s |
|---|---|
| Qwen3-30B baseline | 169.66 |
| Qwen3-30B stream slab | 29.71 |
| Qwen3-30B stream fused | 34.04 |
| **DSV3 671B baseline** | **OOM** |
| **DSV3 671B stream slab** | **0.57** |
| **DSV3 671B stream fused** | **0.68** |

The DSV3 row is the hackathon's "you can't do this otherwise" headline.
