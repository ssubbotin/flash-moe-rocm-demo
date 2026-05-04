#!/usr/bin/env bash
# Pack a GGUF MoE checkpoint's expert tensors into per-(layer, projection)
# files for use with --moe-stream-dir. Multi-shard GGUFs: pass any one shard;
# rerun this script on each shard (it skips non-matching tensors).
#
# Usage:
#   pack_model.sh <input.gguf-or-shard1> <output_dir> [moe-pack-binary]
set -euo pipefail

INPUT="${1:?usage: pack_model.sh <input.gguf> <output_dir> [moe-pack-binary]}"
OUT="${2:?usage: pack_model.sh <input.gguf> <output_dir> [moe-pack-binary]}"
BIN="${3:-${HOME}/llamacpp-moe-cache/build/bin/moe-pack}"

if [[ ! -x "${BIN}" ]]; then
    echo "moe-pack not found at ${BIN}; run scripts/build_llamacpp.sh first" >&2
    exit 1
fi

mkdir -p "${OUT}"

# Detect multi-shard pattern (NNNNN-of-NNNNN). If matched, run on each shard
# in parallel; otherwise just run on the single file.
DIR=$(dirname "${INPUT}")
NAME=$(basename "${INPUT}")
PREFIX="${NAME%-*-of-*.gguf}"

if [[ "${NAME}" =~ -[0-9]{5}-of-[0-9]{5}\.gguf$ ]]; then
    echo "[pack] detected sharded GGUF prefix '${PREFIX}', running on all shards"
    ls "${DIR}/${PREFIX}-"*-of-*.gguf | xargs -P 8 -I{} "${BIN}" {} "${OUT}/"
else
    "${BIN}" "${INPUT}" "${OUT}/"
fi

echo "[pack] done — $(ls "${OUT}" | wc -l) files in ${OUT}, $(du -sh "${OUT}" | cut -f1) total"
