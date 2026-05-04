#!/usr/bin/env bash
# Start llama-server with MoE streaming enabled.
#
# Usage:
#   run_server.sh <model.gguf> <stream_dir> [cache_mb] [port] [extra args...]
set -euo pipefail

MODEL="${1:?usage: run_server.sh <model.gguf> <stream_dir> [cache_mb] [port]}"
STREAM_DIR="${2:?usage: run_server.sh <model.gguf> <stream_dir> [cache_mb] [port]}"
CACHE_MB="${3:-8192}"
PORT="${4:-8080}"
shift 4 || true
BIN="${LLAMA_SERVER:-${HOME}/llamacpp-moe-cache/build/bin/llama-server}"

# MOE_STREAM_FUSED=1 enables the scattered-pointer fused dispatch (~+15% over
# the slab fallback on Q4_K_M).
exec env MOE_STREAM_FUSED=1 "${BIN}" \
    -m "${MODEL}" \
    --moe-stream-dir "${STREAM_DIR}" \
    --moe-cache-mb "${CACHE_MB}" \
    -ngl 99 \
    --port "${PORT}" \
    "$@"
