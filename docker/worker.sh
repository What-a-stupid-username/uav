#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: $0 INPUT_DIRECTORY OUTPUT_DIRECTORY [UAV options...]" >&2
  exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
input_dir="$(realpath -e -- "$1")"
if [[ ! -d "$input_dir" ]]; then
  echo "Input must be a directory" >&2
  exit 2
fi
mkdir -p -- "$2" "$repo_dir/cache" "$repo_dir/tmp"
output_dir="$(realpath -e -- "$2")"
shift 2
exec docker run --rm -i --device "${UAV_GPU_DEVICE:-nvidia.com/gpu=all}" --ipc=host \
  --entrypoint python --workdir /workspace \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -e PYTHONPATH=/workspace:/workspace/third_party/sam3 \
  -e TORCH_EXTENSIONS_DIR=/workspace/tmp/extensions -e UAV_MODEL_DIR=/models -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$repo_dir:/workspace:ro" \
  -v "$repo_dir/models:/models:ro" \
  -v "$input_dir:/input:ro" \
  -v "$output_dir:/output" \
  -v "$repo_dir/cache:/cache" \
  -v "$repo_dir/tmp:/workspace/tmp" \
  "${UAV_IMAGE:-uav:local}" -u -m uav.cli --worker "$@"
