#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/lowcmd_roundtrip_probe [probe options...]" >&2
  exit 2
fi

probe="$1"
shift
if [[ ! -x "$probe" ]]; then
  echo "probe is missing or not executable: $probe" >&2
  exit 2
fi

exec "$probe" --domain "${ISSUE9_DOMAIN_ID:-73}" \
  --interface "${ISSUE9_DDS_INTERFACE:-lo}" "$@"
