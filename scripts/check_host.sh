#!/usr/bin/env bash
set -euo pipefail

echo "== Docker =="
docker version --format 'Client={{.Client.Version}} Server={{.Server.Version}}'

echo
echo "== GPU in Docker =="
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo
echo "Host checks passed."
