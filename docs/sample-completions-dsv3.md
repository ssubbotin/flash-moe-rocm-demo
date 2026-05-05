# Sample DeepSeek-V3 671B completions on a single MI300X

Captured 2026-05-05 from `llama-server` running on a DigitalOcean MI300X VF (gfx942, 192 GiB HBM3, ROCm 7.0) with this branch's `--moe-stream-dir` + `MOE_STREAM_FUSED=1` + 128 GiB cache pool + the per-(layer, projection) pack files this fork's `moe-pack` produced.

**Model**: `unsloth/DeepSeek-V3-GGUF` Q4_K_M, 9 shards, 377 GiB on disk.

**Server flags**: `-m DeepSeek-V3-Q4_K_M-00001-of-00009.gguf -ngl 99 --moe-stream-dir /mnt/scratch/moe-stream-dsv3 --moe-cache-mb 131072 -c 1024 --no-warmup` with `MOE_STREAM_FUSED=1`. `--no-warmup` is required for DSV3 in this build because the upstream warmup graph triggers a CUDA-graph-capture path that `ggml_cuda_graph_check_compability` doesn't fully cover for our buffer type (TODO follow-up). Skipping warmup costs nothing for correctness — the cache fills naturally on first inference instead.

Prefill is slow (cold cache + 2-stage fill); decode steady-state runs at ~0.6 tok/s, varying with prompt-specific expert-routing locality.

---

## 1. Code generation — Python "longest palindromic substring"

**Prompt**: *"Write a Python function find_longest_palindrome(s) that returns the longest palindromic substring. Just the function, no explanation."*

**Response** (149 tokens, 0.55 tok/s):

```python
def find_longest_palindrome(s):
    def expand_around_center(s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    longest = ""
    for i in range(len(s)):
        odd = expand_around_center(s, i, i)
        even = expand_around_center(s, i, i + 1)
        if len(odd) > len(longest):
            longest = odd
        if len(even) > len(longest):
            longest = even
    return longest
```

**Verdict**: Correct expand-around-centers approach. Handles odd and even-length palindromes. O(n²) time, O(1) extra space. Ships as-is.

---

## 2. Math reasoning — clock angle at 3:15

**Prompt**: *"If a clock shows exactly 3:15, what is the angle (in degrees) between the hour hand and the minute hand? Give a one-sentence answer."*

**Response** (22 tokens, 0.68 tok/s):

> At 3:15, the angle between the hour and minute hands is 7.5 degrees.

**Verdict**: Correct. At 3:15 the minute hand is at 90° (pointing at 3), and the hour hand has advanced ¼ of the way from 3 to 4, so it's at 97.5°. Difference = 7.5°. The model gets the subtle "hour hand also moves" detail that simpler models miss (which would say "0 degrees" because both hands look at 3).

---

## 3. Logical reasoning — three-premise syllogism

**Prompt**: *"Premise 1: All birds have wings. Premise 2: Penguins are birds. Premise 3: Penguins cannot fly. Question: Do penguins have wings? Answer in one sentence."*

**Response** (18 tokens, 0.72 tok/s):

> Yes, penguins have wings because all birds have wings and penguins are birds.

**Verdict**: Correct. The model distinguishes "having wings" from "being able to fly" — premise 3 is a distractor that some models fail.

---

## 4. Multi-turn — context retention across cache evictions

**Conversation**:

```
user:      My name is Sergey and I am working on the AMD Developer Hackathon.
assistant: Hello Sergey! How is the hackathon going for you?
user:      What is my name and what event am I working on?
```

**Response** (41 tokens, 0.66 tok/s):

> Your name is **Sergey**, and you're working on the **AMD Developer Hackathon**. Best of luck with your project! Let me know if you need help with anything. 😊

**Verdict**: Correct attribution across turns. Important: between the first and last message, the LRU cache has been actively evicting and refetching expert weights from SSD; the model's context window content is held in KV cache (VRAM-resident, separate from the streaming expert cache), so this isn't surprising on its own — but it's a useful end-to-end correctness check that the streaming dispatcher doesn't corrupt KV state.

---

## Observed end-to-end behaviour

- **Server load time**: 210 s (3.5 min) for the full 377 GiB model + 128 GiB cache pool init.
- **Per-completion latency** (decode-only, this batch): consistently 0.55–0.72 tok/s, in line with the bench-measured 0.68 tok/s steady-state on this configuration.
- **Quality**: all four prompts produced sane, correct outputs. No tokenizer artefacts, no garbled characters, no truncation surprises. The streaming dispatcher's correctness guarantees hold under realistic generation lengths (18–149 tokens here, 47-token sample in [sample-completion-dsv3.md](sample-completion-dsv3.md)).
- **Storage tier behaviour**: vmtouch had pre-warmed the pack files to 62 % page-cache residency before this run. Slab-path benches subsequently churned that down to 20 %; fused-path was gentler (24 % residency after). The completions above ran on the fused path and used a mix of warm-cache (page-cache hits) and cold (SSD pread) expert reads.
