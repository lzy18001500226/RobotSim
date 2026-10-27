# AgiBot X2 OmniPicker model selection

The registry at `registry.json` is the single selection point for simulator
consumers. It exposes exactly two public identities:

- `X2-OmniPicker-CAD`: preserved historical CAD-supported experimental model.
- `X2-OmniPicker-STL`: table-driven source-STL kinematic review candidate.

Call `load_x2_omnipicker()` to load the registry default or pass one of the
two identities explicitly. Unknown and historical aliases fail closed. The
registry default on this branch is a proposed selection pending full-X2 human
visual review; it is not a statement of manufacturer equivalence or physical
grasp qualification.
