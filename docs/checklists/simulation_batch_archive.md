# Simulation batch archive checklist

Use at batch completion or retirement. The producer classifies the runs;
maintenance verifies the manifest under the
[storage workflow](../workflows/storage_maintenance.md). An open checkbox or
unknown reference keeps the affected item protected.

- [ ] Batch processes have ended or their owner explicitly excluded active runs;
  logs/snapshots are stable. Record owner, source fingerprint and actual status.
- [ ] The accepted stage has a pushed checkpoint or complete verified
  [source handoff](../workflows/stage_checkpoint.md). Preserve new/untracked
  helpers, configuration, executable modes and immutable dependency pins.
- [ ] The current truth 10/10 package is pinned: all ten case identities, same
  relevant frozen version, actual acceptance records, measured traces needed
  for independent checks, exact source snapshots and reproduction commands.
- [ ] Radar resume inputs and the first unresolved timing/control disagreement
  are pinned, including source, map/state, input capture and alignment metadata.
- [ ] Resolve source/config/doc/Issue/PR links, generated pointers, symlinks,
  parent-run dependencies and required model/SDK/container references. Retain
  required failed runs and golden baselines as well as successful cases.
- [ ] Classify each leaf: keep online, archive with restore path, superseded but
  referenced, or deletion candidate. Do not classify by age or failure alone.
- [ ] List every exact candidate path, owner, type, size and rationale. Show
  nested Git checkouts and uncommitted work. Verify no candidate ancestor
  contains protected descendants.
- [ ] Record archive target free space, manifest/hash format, mode/path metadata
  and restore command. Demonstrate integrity and a realistic restore sample
  before removing originals; a tar/zip command returning zero is insufficient.
- [ ] Obtain explicit approval for this exact destructive batch and executor.
  Coordinate a single writer; a previous approval does not cover unseen paths.
- [ ] Executor rechecks protection/consumers, hashes and resolved paths before
  approved removal, then verifies retained consumers and reports failures.
- [ ] Record Windows D: free bytes separately from WSL root free bytes, before
  and after. Report VHDX logical length separately. Schedule offline compaction
  and Docker/WSL recovery only through its separately authorized service owner.
- [ ] Publish the small disposition/restore index with the stage handoff or
  existing Issue/PR; keep bulk archives outside tracked source. Mark any missing
  source, acceptance, restore or consumer validation as unresolved.
