# Robot Asset Registry

This directory identifies exact robot/end-effector assemblies without storing vendor meshes. `registry.json` is the project-owned identity and review-status record; `source_hashes.json` is the generated, text-only SHA-256 inventory of pinned source files and mesh bundles; `compiled_model_signatures.json` stores the exact MuJoCo 3.3.6 model counts and names. Vendor checkouts remain external and must be reconstructed from their pinned source repositories.

## Current State

| Asset family | Registry IDs | Current state |
|---|---|---|
| Unitree G1 stock | `g1_unitree_mujoco_stock_23dof`, `g1_unitree_mujoco_stock_29dof`, `g1_rev1_stock_urdf_5994d4f` | Source/compile validated; pending human review |
| G1 Inspire | `g1_rev1_inspire_dfq_bilateral_5994d4f`, `g1_rev1_inspire_ftp_bilateral_5994d4f` | Distinct official assemblies; pending human review |
| G1 Dex1.1 | `g1_rev1_dex1_1_bilateral_5994d4f` | Source/compile validated; pending human review |
| G1 Dex3 | `g1_rev1_dex3_historical_5994d4f` | Rejected historical three-finger appearance; no new gallery generated |
| X2 base, OmniHand, OmniPicker | `x2_ultra_base_mjcf_v1_4`, `x2_ultra_omnihand_v1_4`, `x2_ultra_omnipicker_v1_4` | Source/compile validated; each pending human review |

No candidate is initially `APPROVED`. Compilation validates structural identity only. It does not establish visual acceptability, hardware control semantics, grasp capability, or physics fidelity. Historical acceptance images may be linked as context but do not change the status in this registry.

OmniHand and OmniPicker are separate assemblies. OmniPicker must contain the claw drivers and must not contain OmniHand finger joints. Inspire DFQ and FTP also have different roots, geometry, collision structure, and mount transforms.

## Reconstruct Source Checkouts

Use the exact commits in `registry.json` and `third_party/LOCK.md`:

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco /tmp/unitree_mujoco
git -C /tmp/unitree_mujoco checkout 1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d
git clone https://github.com/AgibotTech/agibot_x2_urdf /tmp/agibot_x2_urdf
git -C /tmp/agibot_x2_urdf checkout 575cc6b988f976c23550e0db85aa1e5475d3652d
git clone https://github.com/unitreerobotics/unitree_ros /tmp/unitree_ros
git -C /tmp/unitree_ros checkout 5994d4faef0a9cadd3287f8de0199a67eeb2a259
```

The unitree_ros evidence in this change was captured from files served at the exact pinned revision because a full checkout was unavailable. For that raw snapshot, the checker requires explicit `--source-revision 5994d4faef0a9cadd3287f8de0199a67eeb2a259`; normal checkouts are checked using `git rev-parse HEAD`.

## Opt-In Identity Check

The identity lock comprises `registry.json`, `source_hashes.json`, and `compiled_model_signatures.json`. Compute the expected hash:

```bash
python scripts/check_robot_asset_registry.py manifest-hash
```

Validate a compiled model before a new opt-in runner starts:

```bash
python scripts/check_robot_asset_registry.py validate \
  --asset-id x2_ultra_omnipicker_v1_4 \
  --manifest-sha256 <hash-from-manifest-hash> \
  --source-root /tmp/agibot_x2_urdf \
  --model /tmp/agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf
```

`validate` checks the asset ID, combined manifest hash, pinned source revision or explicit raw-snapshot revision, every recorded source-file/mesh hash, exact root-model hash, MuJoCo structural signature, and family-specific required/forbidden names. Existing research runners are not changed by this registry task.

Use one full-assembly ID for a runtime model. Do not combine a base ID with a separately named hand unless a future registry row defines that exact derived assembly and its own model/hash profile.

## Human Review Gallery

The candidate gallery lives outside source at `C:\Users\HP\Desktop\RobotSim\reviews\asset-registry-review-20261012`. It records exact model identity, image hashes, limitations, and `physics_steps=0`. Open-like/closed-like frames are static source-joint configurations, not actuator trajectories. No video is produced for assemblies whose source lacks MuJoCo actuation semantics.

Interactive MJCF inspection:

```bash
source /path/to/mujoco-3.3.6-venv/bin/activate
python -m mujoco.viewer --mjcf /tmp/unitree_mujoco/unitree_robots/g1/g1_29dof.xml
```

The built-in viewer's `--mjcf` mode accepts MJCF, not URDF. For an exact URDF candidate, use the registry renderer or compile it to an explicitly identified MJCF and record the derived file hash before opening it. A conversion is a derived artifact, never the upstream source model.

## Human Review Checklist

For each `asset_id`, inspect its own gallery and record a separate decision:

- Correct robot body and source revision.
- Correct end-effector family and left/right side; no mixed/duplicated end effectors.
- Plausible wrist mounting and handedness.
- Correct articulation; static joint poses do not prove an actuator interface.
- Visual and collision representations align and include expected contact geometry.
- Simulation-only changes are explicit.
- Asset ID, combined manifest hash, source-file and model hashes match.

Only an explicit human action may change a row to `APPROVED` or `REJECTED`. Record reviewer, date, exact asset ID, manifest hash, decision scope and evidence path. Do not infer approval from compile or test success.

## Future Task Contract

New tasks and opt-in runners should declare:

```text
asset_id: <exact registry ID>
registry_manifest_sha256: <combined hash>
model_path: <exact expected source or derived path>
identity_validation: PASS required before simulation
```

Do not retrofit existing research runners without separate authorization. Missing, stale, modified, structurally mismatched, or mutually exclusive model identity must fail and report the differing file/name/count.
