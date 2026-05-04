# Agent demo transcripts

Captured 2026-05-05 against a `llama-server` running **Qwen3-30B-A3B Q4_K_M** on the MI300X droplet (this branch's binary, port 18080 via SSH tunnel). Qwen3-30B is used here instead of DeepSeek-V3 only because DSV3 at ~0.6 tok/s would make multi-turn tool flows take minutes per response — the streaming path is fully validated by the DSV3 bench in [benchmarks.md](benchmarks.md) and [sample-completion-dsv3.md](sample-completion-dsv3.md).

The same `agent-demo` Python harness (3 tools: `read_file`, `run_shell`, `http_get`) drives any OpenAI-compatible server on the local llama.cpp branch.

## 1. Direct knowledge (no tool call)

```bash
$ agent-demo --task "What is 7 times 8?" --base-url http://localhost:18080/v1 --max-steps 3
7 times 8 equals 56.
```

The agent recognised it didn't need a tool, answered directly. (System prompt asks the model to prefer direct answers when possible.)

## 2. Single tool call (`run_shell` for live host info)

```bash
$ agent-demo --task "Run the shell command 'uptime' and tell me how long the system has been up." \
             --base-url http://localhost:18080/v1 --max-steps 5

The system has been up for **1 day, 5 hours, and 47 minutes**.

The output from the `uptime` command is:
`01:25:43 up 1 day, 5:47, 1 user, load average: 0,90, 0,78, 0,75`

The relevant portion is `up 1 day, 5:47`.
```

The model emitted a `tool_calls` block with `{"name": "run_shell", "arguments": {"cmd": "uptime"}}`, the harness ran it on the local box (the agent host, not the droplet — the droplet runs the model), captured stdout, fed it back into the next chat turn, and the model summarised the result.

## What this demonstrates

- The streaming branch's `llama-server` correctly negotiates the OpenAI tool-calling protocol.
- The `--jinja` template (Qwen3 chat template) properly emits and parses `<tool_call>` XML.
- A standard `openai` Python SDK client works against this server with no special wrapper.
- An MI300X-served LLM can drive a tool-using agent loop end-to-end at usable interactive speeds (~30 tok/s on Qwen3-30B with the streaming buffer type active).

## Repro

```bash
# 1. Start the server on a machine with MI300X (or any ROCm/CUDA card)
./scripts/run_server.sh /path/to/model.gguf /path/to/moe-stream-dir/ 8192 8080

# 2. Tunnel back if running remotely
ssh -fN -L 18080:localhost:8080 your-droplet

# 3. Install + run
pip install -e .
agent-demo --task "your question" --base-url http://localhost:18080/v1
```
