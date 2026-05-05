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

**Update 2026-05-05 ~08:00 CEST** — read the lablab.ai rules + llama.cpp `AGENTS.md`. Two policy findings drive the revised plan:

- **lablab.ai is OK with AI-assisted work** but requires disclosure (Code of Conduct: "We also disclose when reliant on any third party AI tools, such as ChatGPT or others"). README updated with the disclosure section (commit `585ed82`). Existing entries openly tag Claude / Claude Code / Codex.
- **llama.cpp upstream is NOT OK with AI-generated PRs** (`AGENTS.md`: "predominantly AI-generated" PRs rejected; AI-written PR descriptions are immediate-closure triggers). **Private forks are explicitly exempt.** So we ship the fork branch as a public reference and DON'T open an upstream PR.

→ The original "open upstream PR" step is **dropped** (PR body archived to `docs/_archive-upstream-pr-body.md`). The fork URL `https://github.com/ssubbotin/llama.cpp/tree/feature/moe-expert-gpu-cache` is what we cite from the submission instead.

### Remaining items needing you:

1. **Record the demo video** (≤ 5 min) — storyboard at `docs/demo-video-storyboard.md`. OBS or similar. Captures needed: model load, bench output, DSV3 chat completion, agent run.

2. **Cover image + slide deck** — required submission fields per lablab `delivering-your-hackathon-solution`. A title slide + 5–6 content slides matching the storyboard sections.

3. **(Optional, extra prize tier) Hugging Face Space** — `flash-moe-rocm-demo` could be published as a Space in the AMD Developer Hackathon HF org. Wins the "most likes" Hugging Face Special Prize ($500 credits + 6mo Pro). Streamlit-style demo wrapping `agent-demo`.

4. **Submit on lablab.ai**:
   - Track: **AI Agents & Agentic Workflows** (best fit — we have a working agent demo + the streaming engine as the "high-performance AI app" angle)
   - Optional secondary: **Build in Public** (post 2 technical updates tagging @lablab, @AIatAMD; we already have open-source + walkthrough)
   - Fields:
     - Project: `flash-moe-rocm-demo`
     - Repo: https://github.com/ssubbotin/flash-moe-rocm-demo
     - Llama.cpp fork (cite, don't PR): https://github.com/ssubbotin/llama.cpp/tree/feature/moe-expert-gpu-cache
     - Demo Application Platform: any (we're a CLI/server tool, the "demo" is the recorded video + repo)
     - License: MIT (verifiable)
     - **AI tool tags**: `Anthropic Claude`, `Claude Code` (matches existing-entry pattern)
     - Description: 1 paragraph from README focusing on "DeepSeek-V3 671B on a single MI300X via streaming MoE expert weights from SSD"

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
