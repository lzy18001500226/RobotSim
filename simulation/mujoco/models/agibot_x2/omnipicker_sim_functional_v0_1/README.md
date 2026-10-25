# X2 OmniPicker Simulation Functional v0.1

Classification: `SIMULATION_ONLY_UNVERIFIED_EXPERIMENTAL_GEOMETRY_SELECTED_COUPLING`.

Registry consumers must request the versioned ID explicitly; there is no implicit default.

This package is the uniquely selectable, hash-pinned port of PR #90 Candidate B. Its only qualification is an isolated, unloaded, single-actuator 2%-100%-2% open/close cycle in MuJoCo 3.3.6. It is not manufacturer transmission fidelity and is not qualified for a bottle grasp, load transfer, lift, full-X2 integration, or hardware control. The tested cycle did not include the dynamic 0% endpoint.

The canonical source is `model.xml`. Its MJCF asset paths are repository-relative and it must be loaded through `simulation/mujoco/models/agibot_x2/loader.py`. Do not load an older Candidate B file by path, infer a default, or fall back to an official fixed-link URDF. The loader requires the exact model ID and verifies every model/source/collision hash before compiling.

## Provisioning

From a fresh RobotSim checkout, provision the ignored vendor source at the governed path and exact revision:

```bash
mkdir -p reference/AgibotTech
git clone --no-checkout https://github.com/AgibotTech/agibot_x2_urdf.git reference/AgibotTech/agibot_x2_urdf
git -C reference/AgibotTech/agibot_x2_urdf fetch --depth=1 origin 575cc6b988f976c23550e0db85aa1e5475d3652d
git -C reference/AgibotTech/agibot_x2_urdf checkout --detach 575cc6b988f976c23550e0db85aa1e5475d3652d
git -C reference/AgibotTech/agibot_x2_urdf status --porcelain
```

The last command must print nothing. Then install the exact packages in an isolated environment and generate the ignored, source-derived proxies:

```bash
python3 -m venv /tmp/robotsim-omnipicker-venv
/tmp/robotsim-omnipicker-venv/bin/python -m pip install -r simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1/requirements.txt
/tmp/robotsim-omnipicker-venv/bin/python simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1/provision_collision.py
```

The provisioner does not alter vendor assets or run MuJoCo dynamics. It verifies the exact source, installed dependency versions, and all 161 generated proxy hashes before reuse or completion.

## Consumer interface

Consumers must select the exact model ID and keep its experimental classification visible:

```python
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker

loaded = load_x2_omnipicker("X2-OmniPicker-Sim-Functional-v0.1")
model = loaded.model
```

There is intentionally no default ID. Unknown, missing, modified, or obsolete selections raise `X2OmniPickerModelError`; consumers must not catch that error to choose an older Candidate B or substitute model.

The model uses the pinned official visual meshes and hash-locked CoACD collision proxies. The proxies are generated locally and are not redistributed in this repository. The local STEP file was not needed to load the model; its provenance and license remain separate unresolved metadata.

## Mesh-frame status and comparison reference

The four local mesh-frame rotations for narrow3, wide3, narrow_loop, and wide_loop are classified `CAD_SUPPORTED_EXPERIMENTAL`. The pinned v1.4.0 URDF specifies zero mesh-origin rotations for these links. The registered offsets derive from fitting the source STL bore centers to CAD-derived D-site directions; they are used for both visual geometry and collision proxies. None is `SOURCE_VERIFIED`, and this package does not establish exact STEP-to-URDF link identity or exact source-surface collision fidelity.

`X2-OmniPicker-Sim-Functional-v0.1` remains the selected experimental functional baseline. Official zero-frame geometry is an unregistered static comparison reference only; it has no dynamics qualification and cannot be selected through the loader. The two frame choices have distinct identities and acceptance boundaries.

Historical qualification artifacts are not required to load or test this package. The frozen PR #90 source model SHA-256 is `32353b81b9893fb049da55962b72a65bbae865189e1d404579502140515d6c45`; its original 17,500-step dynamic record SHA-256 is `c1c6362987067736f9ab29446ad3a601b41a21b1e8d5e20c59514d26da192e7a`. That evidence qualifies the original PR #90 model identity only. The portable model hash above has static-equivalence coverage; portable artifact dynamics were not rerun. These historical files live outside the source checkout and a fresh clone does not depend on their local `reviews/` paths.

## Historical variants

The existing [`docs/research/issue46_omnipicker_model_integrity_cleanup_manifest.md`](../../../../../docs/research/issue46_omnipicker_model_integrity_cleanup_manifest.md) remains the source of truth for the prior 52-file / 24-hash historical inventory, per-path references, and proposed deprecation/cleanup dispositions. It is review-only: no historical variant, evidence, worktree, or source asset is removed by this model package.

## Validation

The package regression performs static `mj_forward` checks only. It checks compiled structure against the frozen PR #90 signature and replays 101 saved static joint configurations to compare joint limits, quartic/mimic residuals, D-point gaps, source-STL gripping-face aperture, and collision-proxy contacts. No `mj_step` is run by this suite. The original 17,500-step run remains historical evidence for the original MJCF hash; path normalization produces a distinct portable model hash and does not itself claim a new dynamic pass.

Run focused tests from the repository root:

```bash
/tmp/robotsim-omnipicker-venv/bin/python -m unittest discover -s simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1/tests -v
```
