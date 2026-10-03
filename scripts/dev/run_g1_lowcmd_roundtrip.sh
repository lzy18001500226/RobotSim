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

if [[ "${ISSUE9_DOMAIN_ID-73}" != 73 || "${ISSUE9_DDS_INTERFACE-lo}" != lo ]]; then
  echo "Issue #9 probe only supports DDS domain 73 on loopback interface lo" >&2
  exit 2
fi
if [[ ${CYCLONEDDS_URI+x} ]]; then
  echo "Issue #9 probe does not accept CYCLONEDDS_URI overrides" >&2
  exit 2
fi
if [[ "${CYCLONEDDS_DOMAIN_ID-73}" != 73 || "${ROS_DOMAIN_ID-73}" != 73 ]]; then
  echo "Issue #9 probe only supports DDS domain 73" >&2
  exit 2
fi

# This Issue #9 probe is deliberately limited to the simulator's local DDS
# segment. The executable independently validates the domain, environment,
# and active OS loopback interface before opening DDS publishers.
exec "$probe" --domain 73 --interface lo "$@"
