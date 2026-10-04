# Technical note: MuJoCo

Focus areas for this project:
- MJCF composition and `<include>`
- body / joint / actuator definitions
- sites and sensor mounting frames
- contact and collision geometry
- camera RGB / depth rendering
- ray casting / `mj_multiRay`
- deterministic stepping
- offscreen / headless operation
- ground-truth state access for evaluation only

Rule: simulation truth must never silently replace a sensor estimate in final experiments.
