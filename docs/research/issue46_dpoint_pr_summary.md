# Issue #46 — D-Point Review Package

This Draft PR publishes a reproducible coordinate audit and a proposed, unapplied MJCF anchor mapping. It does not change official model files or run physics.

- Reproduced D-anchor separation: `6.641573 mm` Euclidean; MuJoCo connect maximum row: `6.616062 mm` (same delta vector, different norm).
- The previous `±0.202251 rad / 0.737 mm` result rotated only the loop site toward the current URDF jaw site. It did not target the source CAD D point.
- Re-expressing the same CAD D point in both official link frames gives the proposed local site coordinates in the report and a source-sample kinematic D gap below `0.00061 mm` across 33 samples per side.
- CAD-to-URDF root/passive coordinates invert sign while the official axes remain unchanged. This makes candidate close endpoint `q_CAD=-0.32` exceed the original root limits; no dynamic rollout was attempted.
- Source geometry indicates candidate `q_CAD` decreasing closes the jaw surfaces; exact outer/middle B-rep overlap is present at `q=-0.19`. Official STL convex-hull contact pairs remain surface-unverified.

Full calculations, frames, fit residuals, aperture samples, limitations and reproduction commands: [`issue46_omnipicker_dpoint_alignment.md`](issue46_omnipicker_dpoint_alignment.md).

Proposed MJCF-only changes remain unapproved and unapplied: [`issue46_dpoint_coordinate_proposal.diff`](issue46_dpoint_coordinate_proposal.diff).
