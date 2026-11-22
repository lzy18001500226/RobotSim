# Read-only storage inspection and maintenance handoff

The infrastructure maintainer owns periodic inspection and candidate manifests.
The named execution owner performs separately authorized archive/removal and
service maintenance. A scan or candidate list grants no deletion permission.
Use the [batch archive checklist](../checklists/simulation_batch_archive.md)
after a batch ends and before retiring its evidence.

## Inspect the two storage layers

Record timestamp, inspection owner and maintenance state before measuring.
Prefer a weekly read-only inventory and another inventory after a completed
simulation batch. These are procedure triggers, not an installed scheduler.
During an active archive or shutdown window, the execution owner supplies WSL
measurements; do not launch WSL to inspect a distribution that was shut down
for offline compaction. Resume Linux inspection after the owner confirms recovery.

| Observation | Meaning and bounded command |
| --- | --- |
| Windows D: total/free bytes | Authoritative host headroom: `Get-Volume -DriveLetter D` (or `Get-PSDrive -Name D`). Save raw bytes and GiB. |
| WSL root total/used/available | Filesystem headroom inside Ubuntu: `df -B1 /`, only while the distro is already running. |
| `/mnt/d` available bytes | A Windows-backed mount; compare with host D: rather than counting it as additional space. |
| Distribution and Docker VHDX locations | Read the Lxss registration/configured Docker storage location. Inspect exact registered paths. |
| VHDX file length | Logical host file length from `Get-Item -LiteralPath`; it is not NTFS allocation, reclaimable bytes or current D: free space. |
| Batch output estimate and archive staging space | Use sizes from a comparable completed batch and the chosen archive destination; measure destination headroom separately. |

Deletion inside ext4 can increase WSL availability while the VHDX stays large.
Only a subsequent host measurement establishes D: recovery. Do not promise
that an internal-byte total equals reclaimable host bytes. Archive destination,
temporary working copies and restore-test space all need their own capacity
budget. Alert when the next approved batch plus archive/restore working space
exceeds measured headroom; record the estimate's source and assumptions.
Storage budgets are operational planning criteria, not robot/model limits.

Avoid a whole-drive recursive scan. Start with named project evidence roots,
exact VHDX files and known archive destinations. Record permission failures,
changing files, links and nested Git roots as unresolved entries. Keep source,
active outputs, databases, containers/volumes and user media protected until
their owner has classified them. Directory names, age and an ended process
alone do not establish disposability.

## Candidate manifest and review

Each candidate must have an absolute resolved path, type, logical/allocated
size where available, producing batch/owner, disposition and rationale,
reference/protection result, archive destination and restore plan. Include
links, nested source checkouts, uncommitted work and current consumers in that
review. Resolve uncertainty by protecting the item and naming the missing fact.

Keep a protected set derived from the current stage packet, accepted-run
manifest, active pointers, source/config references and resume inputs. Follow
their dependencies transitively: a protected summary can depend on an older
failed attempt or source snapshot. A grep over tracked docs is insufficient
when pointers, generated configurations or binary/model assets also refer to it.
The producer signs off the protection set; the infrastructure owner checks it.
Record exact candidate leaves, not a wildcard parent that includes protected runs.

## Current X2 protection and division of work

The accepted truth 10/10 version, its per-case measured evidence and matching
source/model/config snapshots stay protected. Obtain its exact frozen manifest
and evidence whitelist from the runtime owner; do not infer all ten cases from
a PASS label, latest timestamp or successes from different fingerprints.

The current truth/radar evidence family is rooted at
`reviews/issue-46/x2-truth-1-10-shared-flow-20261120T172200Z/` under the canonical
checkout. The known radar continuation includes
`lio-live-continuous-map-shadow-case07-20261122T150220Z/`. Treat this family as
protected while the exact accepted/resume dependency manifest is unresolved;
individual unrelated leaves need a separate producer-reviewed disposition.
These are current lookup anchors, not a new assertion of physical acceptance.

Keep radar continuation source and uncommitted files, estimator/model pins,
recorded LiDAR/IMU/encoder inputs, map state, timing alignment and diagnostic
comparison evidence needed to reproduce or resume. Preserve data establishing
the first control/stop disagreement as well as success summaries.

For this handoff, Local02 owns the one-time archive and the later combined
Docker/WSL shutdown, trim/offline compaction and service recovery. Infrastructure
performs read-only investigation and manifest design. Avoid concurrent archive,
move, deletion or compaction on the same paths. Do not run prune, shutdown,
trim or privileged compaction from a read-only inspection.

## Authorized execution handoff

Present the exact manifest, expected effect, backup/restore test, service impact
and executor to the user before a new destructive batch. After approval the
executor rechecks paths and protection immediately before acting, preserves
checksummed archives and per-file metadata, verifies restore samples and
retained consumers, then records both Windows and WSL space again.

Offline compaction has a separate service window: record active distro/container
state, finish source/evidence writes, confirm the owner-approved shutdown,
perform the supported offline procedure, restore the original services and
verify the retained task inputs. Keep failures and recovery actions in the
same execution record. This document does not authorize a shutdown or deletion.
