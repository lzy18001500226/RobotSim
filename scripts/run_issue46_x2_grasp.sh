#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_pin="575cc6b988f976c23550e0db85aa1e5475d3652d"
vendor_root="${AGIBOT_X2_VENDOR_ROOT:-/tmp/robotsim-issue46-agibot-x2-urdf-${source_pin}}"
venv_root="${ISSUE46_X2_VENV_ROOT:-/tmp/robotsim-issue46-x2-prototype-575cc6b9/.venv}"
evidence_dir="${ISSUE46_EVIDENCE_DIR:-/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/scene-frame-correction/$(date +%Y%m%d-%H%M%S)}"
finger_close_fraction="${ISSUE46_FINGER_CLOSE_FRACTION:-0.45}"

mkdir -p "$evidence_dir"
exec > >(tee "$evidence_dir/run.log") 2>&1

printf 'RobotSim Issue #46 X2 single-hand physical grasp reproduction\n'
printf 'RobotSim HEAD: %s\n' "$(git -C "$repo_root" rev-parse HEAD)"
printf 'Upstream: https://github.com/AgibotTech/agibot_x2_urdf\n'
printf 'Upstream pin: %s\n' "$source_pin"
printf 'Vendor checkout: %s\n' "$vendor_root"
printf 'Evidence directory: %s\n' "$evidence_dir"
printf 'Finger close fraction: %s\n' "$finger_close_fraction"

if [[ ! -d "$vendor_root/.git" ]]; then
  git clone --no-checkout https://github.com/AgibotTech/agibot_x2_urdf.git "$vendor_root"
fi
git -C "$vendor_root" fetch --no-tags origin "$source_pin"
git -C "$vendor_root" checkout --detach "$source_pin"
actual_pin="$(git -C "$vendor_root" rev-parse HEAD)"
[[ "$actual_pin" == "$source_pin" ]]
[[ -z "$(git -C "$vendor_root" status --porcelain)" ]]

if [[ ! -x "$venv_root/bin/python" ]]; then
  python3.10 -m venv "$venv_root"
fi
"$venv_root/bin/python" -m pip install --disable-pip-version-check \
  'mujoco==3.3.6' 'numpy==1.26.4' 'scipy==1.15.3' 'imageio==2.37.0' 'imageio-ffmpeg==0.6.0'

cat > "$evidence_dir/experiment_commands.txt" <<EOF
git clone --no-checkout https://github.com/AgibotTech/agibot_x2_urdf.git $vendor_root
git -C $vendor_root fetch --no-tags origin $source_pin
git -C $vendor_root checkout --detach $source_pin
python3.10 -m venv $venv_root
$venv_root/bin/python -m pip install mujoco==3.3.6 numpy==1.26.4 scipy==1.15.3 imageio==2.37.0 imageio-ffmpeg==0.6.0
AGIBOT_X2_VENDOR_ROOT=$vendor_root ISSUE46_EVIDENCE_DIR=$evidence_dir ISSUE46_FINGER_CLOSE_FRACTION=$finger_close_fraction MUJOCO_GL=egl $venv_root/bin/python $repo_root/scripts/research/issue46_x2_grasp.py
EOF

set +e
AGIBOT_X2_VENDOR_ROOT="$vendor_root" \
ISSUE46_EVIDENCE_DIR="$evidence_dir" \
ISSUE46_FINGER_CLOSE_FRACTION="$finger_close_fraction" \
MUJOCO_GL=egl \
  "$venv_root/bin/python" "$repo_root/scripts/research/issue46_x2_grasp.py"
status=$?
set -e
printf 'Experiment exit status: %s\n' "$status"
exit "$status"
