#!/usr/bin/env bash
set -euo pipefail
if (( $# < 3 )); then
  echo "Usage: $0 OUTPUT_DIRECTORY INPUT_VIDEO INPUT_VIDEO... [-- UAV options...]" >&2
  exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p -- "$1" "$repo_dir/cache" "$repo_dir/tmp"
output_dir="$(realpath -e -- "$1")"
shift
mounts=(); sources=(); index=0
while (( $# )) && [[ "$1" != -- ]]; do
  source_path="$(realpath -e -- "$1")"
  [[ -f "$source_path" ]] || { echo "Input must be a regular file: $1" >&2; exit 2; }
  index=$((index+1))
  target="/input/$index-$(basename -- "$source_path")"
  mounts+=(-v "$source_path:$target:ro")
  sources+=(--source "$target")
  shift
done
(( index > 0 )) || { echo 'No videos supplied' >&2; exit 2; }
if (( $# )); then shift; fi
exec docker run --rm --device "${UAV_GPU_DEVICE:-nvidia.com/gpu=all}" --ipc=host \
  --entrypoint python --workdir /workspace \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -e PYTHONPATH=/workspace:/workspace/third_party/sam3 \
  -e TORCH_EXTENSIONS_DIR=/workspace/tmp/extensions -e UAV_MODEL_DIR=/models -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$repo_dir:/workspace:ro" \
  -v "$repo_dir/models:/models:ro" \
  -v "$output_dir:/output" \
  -v "$repo_dir/cache:/cache" \
  -v "$repo_dir/tmp:/workspace/tmp" \
  "${mounts[@]}" "${UAV_IMAGE:-uav:local}" -m uav.cli \
  "${sources[@]}" --output /output "$@"
