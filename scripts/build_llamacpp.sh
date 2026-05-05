#!/usr/bin/env bash
# Clone llama.cpp + check out the moe-stream branch + build for ROCm gfx942.
#
# Assumes ROCm 7.0+ is installed, hipcc + cmake on PATH, and the GPU is gfx942.
# Adjust GGML_HIP_ARCH for other CDNA/RDNA targets.
set -euo pipefail

DEST="${1:-${HOME}/llamacpp-moe-cache}"
BRANCH="${BRANCH:-feature/moe-expert-gpu-cache}"
REMOTE_URL="${REMOTE_URL:-https://github.com/ssubbotin/llama.cpp.git}"
ARCH="${ARCH:-gfx942}"

if [[ ! -d "${DEST}/.git" ]]; then
    echo "[build] cloning ssubbotin/flash-moe (forked llama.cpp branch)"
    git clone --branch "${BRANCH}" "${REMOTE_URL}" "${DEST}"
else
    echo "[build] ${DEST} already cloned — fetching"
    git -C "${DEST}" fetch origin "${BRANCH}"
    git -C "${DEST}" checkout "${BRANCH}"
fi

cd "${DEST}"
[[ -d build ]] || cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS="${ARCH}" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-bench moe-pack
echo "[build] done — binaries in ${DEST}/build/bin/"
