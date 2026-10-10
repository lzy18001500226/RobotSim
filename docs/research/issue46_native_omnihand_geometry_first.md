# Issue #46 Native OmniHand Geometry-First Contact

**Result: BLOCKED at persistent bilateral contact.** The geometry scan found two source-valid thumb/index target poses, and a bounded same-pose probe produced real MuJoCo thumb and index contacts. Contact did not meet the existing 80% final-100-ms persistence gate, and the hand carried no upward bottle load. The middle-finger, load-transfer, and lift phases were not run.

## Runtime and candidates

- Upstream X2 URDF pin: `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- Source URDF SHA-256: `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`.
- Runtime: MuJoCo 3.3.6, Python 3.10.12, 2 ms timestep, `implicitfast`.
- Bottle: canonical 0.57 kg geometry; no bottle or hand collision geometry edits. No bottle qpos writes after initialization.
- `prior_fixed_palm` failed the static bilateral target check (thumb surface gap 73.90 mm).
- `offset_50x65mm` at `[0.250, 0.159, 0.9175]` m passed static geometry: thumb/index target gaps 0.043/0.053 mm; middle/ring/pinky gaps 7.946/7.512/7.799 mm; palm-to-bottle gap 31.635 mm; hand-to-table gap 47.884 mm.
- `offset_55x70mm` at `[0.245, 0.154, 0.9175]` m passed the static geometry check (thumb/index gaps 0.109/0.056 mm), but its bounded dynamic attempt did not reach contact.

Both candidate orientations were WXYZ `[0.70710678, 0.70710678, 0, 0]`. Source joint ranges were enforced. The dynamic fixture used simulation-only direct follower position actuators; this is not evidence of hardware controller semantics.

## Dynamic result

The best candidate was allowed to settle at the static target, then received a bounded target refinement of `+0.005` thumb alpha and `+0.003` index alpha. The trace records collision-enabled contacts between `R_thumb_dip` and the bottle and between `R_index_dip` and the bottle.

- Probe phase: 12/125 frames simultaneous contact (9.6%), from 2.824 s through 2.850 s.
- 0.5 s hold: 212/250 frames simultaneous contact (84.8%), from 2.852 s through 3.350 s; the longest uninterrupted episode was 74 ms.
- Final 100 ms: thumb contact 74%, index contact 100%; the bilateral gate requires each to be at least 80%, and the longest simultaneous-contact episode was 20 ms.
- Bottle weight: 5.5917 N. Mean hand vertical force on the bottle over the hold was `-0.02235 N`; in the final 100 ms it was `-0.00513 N` (signed downward), corresponding to approximately `-0.40%` and `-0.092%` of weight. The table remained the supporting contact.
- During the hold the bottle moved approximately `[+0.0156, -0.0745, +0.0018]` mm and rotated 0.00372 rad. Maximum observed penetration was 0.194 mm; no safety abort occurred.
- Contact-driven middle-finger closure, load transfer, 1 mm lift, and 5 mm lift were not run. No grasp or pickup is claimed.

## Evidence and reproduction

Raw traces, JSON results, CSV contact envelope, and MP4 are stored outside Git in the local `geometry-first-20261010` evidence directory. The final probe artifacts are `contact-probe-verification-01/offset_50x65mm/dynamic/dynamic_trace.jsonl`, `contact_analysis.json`, `dynamic_result.json`, and `dynamic_contact.mp4`. The captured dynamic runner SHA-256 is `9c56f8466685c533f78a6ea9edd0b1dce5e0c00acadf73b7f7e38b7a1fa366f2`; the updated runner that adds offline bilateral-contact timing metrics is `63cb48ca4486288445e18868e1a5df4cc41b9d03df6e46fd9786818ff7d2bb32`. Fixture SHA-256 is `1052fe9b8398661bc7c342b8573fdd031e9aadcd4d15da96798c051b6b785233`.

From the experimental worktree, rerun one bounded candidate with:

```bash
AGIBOT_X2_VENDOR_ROOT=/tmp/robotsim-issue46-native-omnihand-575cc6b988f976c23550e0db85aa1e5475d3652 \
ISSUE46_GEOMETRY_EVIDENCE_DIR=/tmp/issue46-geometry-first-rerun \
python \
scripts/research/issue46_native_omnihand_geometry_first.py --candidate offset_50x65mm
```

The rerun writes a fresh bounded fixture and does not change source geometry. No further pose search or lift is justified by this result; the next step requires maintainer review of the contact/controller model.
