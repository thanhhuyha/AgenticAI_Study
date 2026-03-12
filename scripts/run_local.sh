#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <image_path> [run_id]"
  exit 1
fi

IMAGE_PATH="$1"
RUN_ID="${2:-$(date +%Y%m%d_%H%M%S)}"

therapy-agent run "$IMAGE_PATH" --run-id "$RUN_ID"
