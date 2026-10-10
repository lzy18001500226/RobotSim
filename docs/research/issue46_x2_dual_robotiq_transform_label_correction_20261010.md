# Issue #46 Dual-Robotiq Transform Label Correction

The raw candidate-B result and executed runner call the operation a local-Y “roll.” That word is inaccurate; the raw evidence is preserved unchanged.

For the compiled adapter, measured frame alignment is:

- local `+Z`: insertion axis;
- local `-X`: jaw-opening axis;
- local `Y`: perpendicular to both.

Candidate B applied `R_new = R_source * Rotation.from_euler("y", pi)` at the unchanged source-mount translation. It is a 180-degree half-turn about local Y, reversing both insertion and jaw-opening axes. It is not a roll about the insertion axis. The checked-in runner uses the corrected name and emits `candidate_b_right_local_y_halfturn_180_simulation_only.xml`; its XML hash is byte-identical to the executed file whose historical filename contains `roll`.

This correction changes terminology only. It does not alter or reinterpret the captured dynamic traces, candidate transform, or FAIL verdict.
