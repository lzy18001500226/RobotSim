#!/usr/bin/env python3
"""Summarize an already completed Issue #46 cone A/B trace; runs no physics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    index = (len(values) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return values[low]
    return values[low] * (high - index) + values[high] * (index - low)


def summarize_run(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    stages = sorted({row["stage"] for row in rows})
    result: dict[str, Any] = {"samples": len(rows), "stages": {}}

    for stage in stages:
        sample = [row for row in rows if row["stage"] == stage]
        contacts = [contact for row in sample for contact in row["contacts"]]
        stage_result: dict[str, Any] = {
            "samples": len(sample),
            "contact_count_min_max": [
                min(row["contact_count_all"] for row in sample),
                max(row["contact_count_all"] for row in sample),
            ],
            "table_contact_samples": sum(row["table_contact_count"] > 0 for row in sample),
            "hand_signed_vertical_support_mean_N": statistics.mean(
                row["hand_signed_world_vertical_support_N"] for row in sample
            ),
            "table_signed_vertical_support_mean_N": statistics.mean(
                row["table_signed_world_vertical_support_N"] for row in sample
            ),
            "bottle_z_min_max_m": [
                min(row["bottle_world_position_m"][2] for row in sample),
                max(row["bottle_world_position_m"][2] for row in sample),
            ],
            "solver_niter_p95_max": [
                percentile(
                    [float(max(row["solver"]["solver_niter"] or [0])) for row in sample],
                    0.95,
                ),
                max(max(row["solver"]["solver_niter"] or [0]) for row in sample),
            ],
            "solver_gradient_max": max(
                max(
                    (entry["gradient"] for entry in row["solver"]["solver_iteration_stats_initial_and_performed"]),
                    default=0.0,
                )
                for row in sample
            ),
            "solver_fwdinv_abs_max": max(
                max((abs(value) for value in row["solver"]["solver_fwdinv"]), default=0.0)
                for row in sample
            ),
            "solver_warning_samples": sum(bool(row["solver"]["warnings"]) for row in sample),
            "contacts_by_pair": {},
            "actuators": {},
        }

        for pair in sorted({tuple(contact["pair"]) for contact in contacts}):
            pair_contacts = [contact for contact in contacts if tuple(contact["pair"]) == pair]
            stage_result["contacts_by_pair"]["/".join(pair)] = {
                "observations": len(pair_contacts),
                "normal_force_mean_max_N": [
                    statistics.mean(contact["normal_force_N"] for contact in pair_contacts),
                    max(contact["normal_force_N"] for contact in pair_contacts),
                ],
                "tangential_force_mean_max_N": [
                    statistics.mean(contact["tangential_force_magnitude_N"] for contact in pair_contacts),
                    max(contact["tangential_force_magnitude_N"] for contact in pair_contacts),
                ],
                "signed_world_vertical_force_mean_N": statistics.mean(
                    contact["force_on_bottle_world_N"][2] for contact in pair_contacts
                ),
                "relative_tangential_slip_mean_p95_max_m_s": [
                    statistics.mean(contact["relative_tangential_speed_m_s"] for contact in pair_contacts),
                    percentile([contact["relative_tangential_speed_m_s"] for contact in pair_contacts], 0.95),
                    max(contact["relative_tangential_speed_m_s"] for contact in pair_contacts),
                ],
                "active_cone_utilization_mean_max": [
                    statistics.mean(
                        contact["cone_utilization"]["active_model_full_condim4_measure"]
                        for contact in pair_contacts
                    ),
                    max(
                        contact["cone_utilization"]["active_model_full_condim4_measure"]
                        for contact in pair_contacts
                    ),
                ],
                "active_cone_utilization_ge_0_99_samples": sum(
                    contact["cone_utilization"]["active_model_full_condim4_measure"] >= 0.99
                    for contact in pair_contacts
                ),
                "peak_utilization_sample": max(
                    (
                        {
                            "utilization": contact["cone_utilization"]["active_model_full_condim4_measure"],
                            "step": row["step"],
                            "time_s": row["time_s"],
                            "efc_states": [entry["state"] for entry in contact["contact_constraint_rows"]],
                        }
                        for row in sample
                        for contact in row["contacts"]
                        if tuple(contact["pair"]) == pair
                    ),
                    key=lambda item: item["utilization"],
                ),
                "contact_position_world_min_max_m": [
                    [min(contact["contact_position_world_m"][axis] for contact in pair_contacts) for axis in range(3)],
                    [max(contact["contact_position_world_m"][axis] for contact in pair_contacts) for axis in range(3)],
                ],
                "constraint_row_states": {
                    state: sum(
                        entry["state"] == state
                        for contact in pair_contacts
                        for entry in contact["contact_constraint_rows"]
                    )
                    for state in sorted(
                        {
                            entry["state"]
                            for contact in pair_contacts
                            for entry in contact["contact_constraint_rows"]
                        }
                    )
                },
            }

        for actuator in ("left", "right", "lift"):
            values = [row["actuators"][actuator] for row in sample]
            errors = [abs(value["tracking_error_q_target_minus_actual"]) for value in values]
            forces = [abs(value["actuator_force_N"]) for value in values]
            qpos = [value["qpos"] for value in values]
            targets = [value["target_q"] for value in values]
            stage_result["actuators"][actuator] = {
                "tracking_error_abs_mean_p95_max": [
                    statistics.mean(errors), percentile(errors, 0.95), max(errors)
                ],
                "effort_abs_mean_max_N": [statistics.mean(forces), max(forces)],
                "actual_qpos_first_last_m": [qpos[0], qpos[-1]],
                "actual_qpos_min_max_m": [min(qpos), max(qpos)],
                "target_q_first_last_m": [targets[0], targets[-1]],
                "target_q_min_max_m": [min(targets), max(targets)],
                "saturated_samples": sum(value["saturated"] for value in values),
            }

        result["stages"][stage] = stage_result

    z0 = rows[0]["bottle_world_position_m"][2]
    result["maximum_bottle_rise_m"] = max(row["bottle_world_position_m"][2] for row in rows) - z0
    result["table_contact_samples_total"] = sum(row["table_contact_count"] > 0 for row in rows)
    result["table_contact_fraction"] = result["table_contact_samples_total"] / len(rows)
    result["initial_final_bottle_z_m"] = [rows[0]["bottle_world_position_m"][2], rows[-1]["bottle_world_position_m"][2]]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "classification": "POSTPROCESSING_ONLY_NO_SIMULATION",
        "elliptic": summarize_run(args.evidence_dir / "steps_elliptic.jsonl"),
        "pyramidal": summarize_run(args.evidence_dir / "steps_pyramidal.jsonl"),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
