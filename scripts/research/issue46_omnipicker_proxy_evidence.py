#!/usr/bin/env python3
"""Summarize a completed P0 collision-proxy run without rerunning physics."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import issue46_omnipicker_collision_proxy as proxy  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def font(size: int) -> ImageFont.ImageFont:
    candidates = [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                  Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf")]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def panel_transform(points: np.ndarray, box: tuple[int, int, int, int],
                    margin: int = 30) -> Any:
    x0, y0, x1, y1 = box
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 0.05)
    scale = min((x1 - x0 - 2 * margin) / spans[0],
                (y1 - y0 - 2 * margin) / spans[1])
    center = (mins + maxs) * 0.5
    panel_center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5])

    def transform(values: np.ndarray) -> list[tuple[int, int]]:
        projected = (values - center) * scale
        pixel = np.empty_like(projected)
        pixel[:, 0] = panel_center[0] + projected[:, 0]
        pixel[:, 1] = panel_center[1] - projected[:, 1]
        return [tuple(np.rint(row).astype(int)) for row in pixel]

    return transform


def draw_geometry_comparison(run_dir: Path, identity: dict[str, Any]) -> None:
    canvas = Image.new("RGB", (1800, 1180), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, row_font, small_font = font(30), font(23), font(17)
    draw.text((40, 22), "Original X2 OmniPicker fingertip collision STL vs SIMULATION_ONLY proxy",
              fill=(25, 33, 42), font=title_font)
    draw.text((42, 65), "Source collision triangles shown; visual meshes are not used in this geometry overlay.",
              fill=(65, 73, 82), font=small_font)
    bodies = list(proxy.PAD_MESHES)
    pad_info = identity["pad_collision_geometry"]

    for row_index, body in enumerate(bodies):
        y0 = 110 + row_index * 520
        draw.text((42, y0), body, fill=(25, 33, 42), font=row_font)
        source = proxy.read_binary_stl(proxy.p0.mesh_source(proxy.PAD_MESHES[body]))
        details = pad_info[body]
        contact = source[int(details["source_triangle_index"])].copy()
        normal = np.cross(contact[1] - contact[0], contact[2] - contact[0])
        normal /= np.linalg.norm(normal)
        tangent = contact[1] - contact[0]
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        center = contact.mean(axis=0)
        local_mask = np.linalg.norm(source.mean(axis=1) - center, axis=1) <= 0.0015
        patch = source[local_mask]
        contact_index = int(details["source_triangle_index"])
        if not np.any(local_mask[contact_index]):
            patch = np.concatenate((patch, contact[None, :, :]), axis=0)
        proxy_faces = proxy.read_binary_stl(run_dir / f"proxy_{proxy.PAD_MESHES[body]}")

        left = (50, y0 + 42, 850, y0 + 485)
        right = (930, y0 + 42, 1730, y0 + 485)
        draw.rectangle(left, outline=(110, 120, 130), width=2)
        draw.rectangle(right, outline=(110, 120, 130), width=2)
        draw.text((left[0] + 12, left[1] + 8), "Source STL local tangent view (mm)",
                  fill=(42, 50, 58), font=small_font)
        draw.text((right[0] + 12, right[1] + 8), "Source STL section + proxy inset (mm)",
                  fill=(42, 50, 58), font=small_font)

        tangent_points = np.concatenate([patch.reshape(-1, 3), contact], axis=0)
        tangent_xy = np.column_stack(((tangent_points - center) @ tangent,
                                      (tangent_points - center) @ bitangent)) * 1000.0
        transform = panel_transform(tangent_xy, (left[0] + 18, left[1] + 42,
                                                  left[2] - 18, left[3] - 15))
        count = 0
        for tri in patch:
            coords = np.column_stack(((tri - center) @ tangent, (tri - center) @ bitangent)) * 1000.0
            pixels = transform(coords)
            draw.polygon(pixels, fill=(218, 224, 229), outline=(130, 140, 148))
            count += 1
        contact_xy = np.column_stack(((contact - center) @ tangent,
                                      (contact - center) @ bitangent)) * 1000.0
        contact_pixels = transform(contact_xy)
        draw.polygon(contact_pixels, fill=(242, 157, 70), outline=(183, 84, 25), width=4)
        outer = proxy_faces[0]
        outer_xy = np.column_stack(((outer - center) @ tangent,
                                    (outer - center) @ bitangent)) * 1000.0
        draw.polygon(transform(outer_xy), fill=None, outline=(0, 100, 100), width=4)

        section_points = np.concatenate([patch.reshape(-1, 3), contact,
                                         proxy_faces.reshape(-1, 3)], axis=0)
        section_x = (section_points - center) @ tangent * 1000.0
        section_y = (section_points - center) @ (-normal) * 1000.0
        section_xy = np.column_stack((section_x, section_y))
        section_transform = panel_transform(section_xy, (right[0] + 18, right[1] + 42,
                                                          right[2] - 18, right[3] - 15))
        for tri in patch:
            coords = np.column_stack(((tri - center) @ tangent * 1000.0,
                                      (tri - center) @ (-normal) * 1000.0))
            draw.polygon(section_transform(coords), fill=(224, 229, 233), outline=(145, 153, 160))
        for face in proxy_faces:
            coords = np.column_stack(((face - center) @ tangent * 1000.0,
                                      (face - center) @ (-normal) * 1000.0))
            draw.polygon(section_transform(coords), fill=(116, 213, 203), outline=(0, 112, 109))
        draw.text((left[0] + 12, left[3] - 31),
                  "Gray = nearby STL facets; orange = selected STL face; teal = proxy outline",
                  fill=(55, 63, 71), font=small_font)
        draw.text((right[0] + 12, right[3] - 31),
                  f"{count} local facets; face {details['source_contact_triangle_area_mm2']:.4f} mm2; inset {details['proxy_thickness_m'] * 1000:.3f} mm",
                  fill=(55, 63, 71), font=small_font)

    draw.text((45, 1150), "Proxy outer vertices match the selected original STL triangle; no outward offset or added exclusion.",
              fill=(50, 58, 66), font=small_font)
    canvas.save(run_dir / "proxy_geometry_comparison.png")


def make_review_brightness_copies(run_dir: Path) -> list[str]:
    outputs = []
    for source_name in ("C_preload_stop.png", "C_preload_stop_closeup.png"):
        source_path = run_dir / source_name
        if not source_path.is_file():
            continue
        image = Image.open(source_path).convert("RGB")
        image = ImageEnhance.Brightness(image).enhance(1.8)
        image = ImageEnhance.Contrast(image).enhance(1.08)
        output_name = f"review_display_{source_name}"
        image.save(run_dir / output_name)
        outputs.append(output_name)
    return outputs


def summarize(run_dir: Path, baseline_dir: Path) -> dict[str, Any]:
    identity = load_json(run_dir / "runtime_identity.json")
    static = load_json(run_dir / "B2_STATIC_GEOMETRY_GATE.json")
    dynamic = load_json(run_dir / "C_RESULT.json")
    baseline = load_json(baseline_dir / "free-cylinder-reference/C_RESULT.json")
    baseline_geometry = load_json(baseline_dir / "fixed-object-baseline/FREE_CONTACT_GEOMETRY.json")
    trace = [json.loads(line) for line in (run_dir / "C_PRELOAD_TRACE.jsonl").read_text(encoding="utf-8").splitlines()]
    preload_rows = [row for row in trace if row.get("preload_feedback") is not None]
    bilateral_angles, force_angles, lateral_forces, vertical_forces = [], [], [], []
    narrow_normal, wide_normal = [], []
    for row in preload_rows:
        contacts = {item["body2"]: item for item in row["contacts"]
                    if item["body2"] in proxy.PAD_MESHES}
        if len(contacts) != 2:
            continue
        narrow, wide = contacts["R_hand_narrow3_Link"], contacts["R_hand_wide3_Link"]
        bilateral_angles.append(angle_deg(np.asarray(narrow["normal_world"]),
                                          np.asarray(wide["normal_world"])))
        narrow_force = np.asarray(narrow["force_on_bottle_world_N"])
        wide_force = np.asarray(wide["force_on_bottle_world_N"])
        force_angles.append(angle_deg(narrow_force, wide_force))
        combined = narrow_force + wide_force
        lateral_forces.append(float(np.linalg.norm(combined[:2])))
        vertical_forces.append(float(combined[2]))
        narrow_normal.append(float(narrow["normal_force_N"]))
        wide_normal.append(float(wide["normal_force_N"]))

    preload = dynamic["stage_metrics"]["PRELOAD"]["preload_feedback_metrics"]
    hold = dynamic["stage_metrics"]["HOLD"]
    force_by_pad = preload["normal_force_by_pad"]
    error_by_pad = {}
    for body, details in identity["pad_collision_geometry"].items():
        source_mesh = proxy.read_binary_stl(proxy.p0.mesh_source(proxy.PAD_MESHES[body]))
        source_face = source_mesh[int(details["source_triangle_index"])]
        proxy_face = proxy.read_binary_stl(run_dir / f"proxy_{proxy.PAD_MESHES[body]}")[0]
        pairwise = np.linalg.norm(source_face[:, None, :] - proxy_face[None, :, :], axis=2)
        error_by_pad[body] = float(np.max(np.min(pairwise, axis=1)))

    if static["status"] != "PASS":
        classification = "PROXY_CONTACT_GEOMETRY_FAIL"
    elif dynamic["failure_classification"] == "PRELOAD_EXCESSIVE_SLIP":
        classification = "PROXY_LATERAL_ESCAPE"
    elif dynamic["status"] != "PASS":
        classification = "PROXY_PRELOAD_FAIL"
    else:
        classification = "PROXY_ISOLATED_LIFT_PASS"

    summary = {
        "experiment": "Issue #46 source-derived fingertip collision proxy A/B",
        "classification": classification,
        "model_classification": "SIMULATION_ONLY_COLLISION_PROXY; not source-faithful hardware acceptance",
        "source_branch_head": {"branch": identity["robot_branch"], "sha": identity["robot_head"]},
        "upstream": {"repository": identity["source_repository"], "commit": identity["source_commit"],
                     "urdf_sha256": identity["source_urdf_sha256"]},
        "runtime": {"mujoco": identity["mujoco_native"],
                    "native_library_sha256": identity["native_library_sha256"],
                    "timestep_s": identity["timestep_s"], "integrator": identity["integrator"],
                    "solver": identity["solver"], "solver_iterations": identity["solver_iterations"]},
        "A_original_source_collision_reference": {
            "rerun": False,
            "source_result": str(baseline_dir / "free-cylinder-reference/C_RESULT.json"),
            "classification": baseline["failure_classification"],
            "lift_m": baseline["maximum_lift_above_initial_m"],
            "slip_m": baseline["max_relative_slip_from_hold_start_m"],
            "normal_angle_deg": baseline_geometry["force_normal_angle_deg"],
            "lateral_resultant_xy_N": baseline_geometry["resultant_pad_force_lateral_xy_N"],
            "hold_pad_mean_normal_force_N": {
                name: baseline["stage_metrics"]["HOLD"]["pad_force_metrics"][name]["mean_normal_force_when_contact_N"]
                for name in proxy.PAD_MESHES},
        },
        "B_proxy_static_geometry": {
            "status": static["status"],
            "classification": "SIMULATION_ONLY_COLLISION_PROXY",
            "only_collision_bodies_replaced": sorted(proxy.PAD_MESHES),
            "source_outer_facet_max_vertex_error_m": error_by_pad,
            "open_no_initial_robot_contact": not bool(static["open"]["initial_contacts"]),
            "open_minimum_nonpad_bottle_clearance_m": static["open"]["minimum_nonpad_distance_m"],
            "approach_minimum_nonpad_bottle_clearance_m": static["approach"]["minimum_nonpad_bottle_clearance_m"],
            "minimum_table_clearance_m": static["approach"]["minimum_gripper_table_clearance_lower_bound_m"],
            "bilateral_target_aperture": static["closure"]["bilateral_contact_target_aperture"],
            "nonpad_closure_collision_free": static["closure"]["non_gripping_collision_free_through_target"],
        },
        "B_proxy_physics": {
            "stage": dynamic["stage"], "failure_detail": dynamic["failure_classification"],
            "one_second_bilateral_hold_fraction": hold["bilateral_pad_contact_fraction"],
            "preload_duration_s": preload["stable_duration_s"],
            "preload_stable_force_gate": preload["stable_force_gate_passed"],
            "target_normal_force_per_pad_N": preload["target_normal_force_per_pad_N"],
            "measured_preload_mean_normal_force_N": {
                name: force_by_pad[name]["mean_N"] for name in proxy.PAD_MESHES},
            "measured_preload_final_normal_force_N": {
                name: force_by_pad[name]["final_N"] for name in proxy.PAD_MESHES},
            "measured_preload_peak_normal_force_N": {
                name: force_by_pad[name]["peak_N"] for name in proxy.PAD_MESHES},
            "preload_normal_angle_deg_mean_min_max": {
                "mean": float(np.mean(bilateral_angles)),
                "min": float(np.min(bilateral_angles)), "max": float(np.max(bilateral_angles))},
            "preload_world_force_angle_deg_mean_min_max": {
                "mean": float(np.mean(force_angles)),
                "min": float(np.min(force_angles)), "max": float(np.max(force_angles))},
            "preload_resultant_xy_force_N_mean_peak": {
                "mean": float(np.mean(lateral_forces)), "peak": float(np.max(lateral_forces))},
            "preload_resultant_vertical_force_N_mean_min_max": {
                "mean": float(np.mean(vertical_forces)), "min": float(np.min(vertical_forces)),
                "max": float(np.max(vertical_forces))},
            "measured_hand_vertical_support_mean_N": preload["mean_measured_hand_vertical_support_N"],
            "table_support_mean_N": preload["mean_table_support_N"],
            "ideal_capacity_upper_bound_mean_N": preload["ideal_vertical_friction_capacity_upper_bound_mean_N"],
            "minimum_required_hand_support_N": dynamic["minimum_required_hand_support_N"],
            "maximum_relative_slip_m": dynamic["max_relative_slip_from_hold_start_m"],
            "slip_gate_m": 0.005,
            "maximum_lift_m": dynamic["maximum_lift_above_initial_m"],
            "lift_stages_attempted": dynamic["lift_attempted"],
            "maximum_pad_penetration_m": dynamic["max_contact_penetration_m"],
            "maximum_source_limit_violations": len(dynamic["source_limit_violations"]),
            "maximum_mimic_residual_rad": dynamic["maximum_mimic_residual_rad"],
            "peak_jaw_tracking_error_rad": max(
                dynamic["stage_metrics"]["PRELOAD"]["jaw_tracking_metrics"][name]["max_abs_post_step_position_error_rad"]
                for name in dynamic["stage_metrics"]["PRELOAD"]["jaw_tracking_metrics"]),
            "effort_saturated_steps": preload["effort_saturated_steps"],
            "peak_actuator_torque_Nm": {
                name: dynamic["stage_metrics"]["PRELOAD"]["actuation_metrics"][name]["peak_applied_torque_nm"]
                for name in dynamic["stage_metrics"]["PRELOAD"]["actuation_metrics"]},
            "effort_caps_Nm": {name: dynamic["stage_metrics"]["PRELOAD"]["actuation_metrics"][name]["effort_cap_nm"]
                               for name in dynamic["stage_metrics"]["PRELOAD"]["actuation_metrics"]},
            "source_joint_ranges_unchanged": identity["source_joint_ranges_unchanged"],
            "mass_inertia_unchanged": identity["body_mass_inertia_unchanged"],
            "controller_profile_unchanged": identity["controller_profile_unchanged"],
            "bottle_qpos_writes": dynamic["bottle_qpos_writes_during_rollout"],
            "bottle_welds_or_equalities": dynamic["bottle_welds_or_equalities"],
            "mocap_carry": dynamic["mocap_carry"],
            "artificial_bottle_force": dynamic["artificial_bottle_force"],
            "hidden_bottle_support": dynamic["hidden_bottle_support"],
        },
        "evidence": {"static_trace": "B2_CLEARANCE_TRACE.jsonl", "physics_trace": "C_PRELOAD_TRACE.jsonl",
                      "physics_result": "C_RESULT.json", "video": dynamic["video"],
                      "geometry_figure": "proxy_geometry_comparison.png"},
        "reproduction_command": load_json(run_dir / "result.json")["reproduction_command"],
        "raw_data_unmodified_by_summary_tool": True,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=proxy.BASELINE_ROOT)
    args = parser.parse_args()
    summary = summarize(args.run_dir, args.baseline_dir)
    identity_path = args.run_dir / "runtime_identity.json"
    identity = load_json(identity_path)
    raw_identity = args.run_dir / "runtime_identity_runner_raw.json"
    if not raw_identity.exists():
        shutil.copy2(identity_path, raw_identity)
    corrected_errors = summary["B_proxy_static_geometry"]["source_outer_facet_max_vertex_error_m"]
    for body, error in corrected_errors.items():
        identity["pad_collision_geometry"][body]["proxy_outer_face_vertex_error_m"] = error
    identity["proxy_outer_face_vertex_error_method"] = (
        "maximum nearest-vertex Euclidean distance after binary STL round-trip; source facet vertex order is ignored"
    )
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["evidence"]["sha256"] = {}
    for name in ["B2_STATIC_GEOMETRY_GATE.json", "B2_CLEARANCE_TRACE.jsonl",
                 "C_RESULT.json", "C_PRELOAD_TRACE.jsonl", "SIMULATION_ONLY_COLLISION_PROXY.xml",
                 "SOURCE_FAITHFUL_CONTROL.xml", "C_preload_feedback.mp4",
                 "proxy_geometry_comparison.png", "executed_runner_snapshot.py",
                 "runtime_identity.json", "runtime_identity_runner_raw.json"]:
        path = args.run_dir / name
        if path.is_file():
            summary["evidence"]["sha256"][name] = proxy.sha256(path)
    draw_geometry_comparison(args.run_dir, identity)
    display_copies = make_review_brightness_copies(args.run_dir)
    summary["evidence"]["display_only_brightness_copies"] = display_copies
    summary["evidence"]["sha256"]["proxy_geometry_comparison.png"] = proxy.sha256(
        args.run_dir / "proxy_geometry_comparison.png")
    for name in display_copies:
        summary["evidence"]["sha256"][name] = proxy.sha256(args.run_dir / name)
    summary["evidence"]["sha256"]["analysis_runner_sha256"] = proxy.sha256(Path(__file__).resolve())
    (args.run_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runner_result_path = args.run_dir / "result.json"
    raw_copy = args.run_dir / "result_runner_summary_raw.json"
    if not raw_copy.exists():
        shutil.copy2(runner_result_path, raw_copy)
    corrected = load_json(runner_result_path)
    corrected["failure_classification"] = summary["classification"]
    corrected["physical_failure_detail"] = summary["B_proxy_physics"]["failure_detail"]
    corrected["summary_corrected_from_raw_physics_result"] = True
    runner_result_path.write_text(json.dumps(corrected, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    print(json.dumps({"classification": summary["classification"],
                      "analysis_summary": str(args.run_dir / "analysis_summary.json"),
                      "geometry_figure": str(args.run_dir / "proxy_geometry_comparison.png"),
                      "raw_runner_summary_preserved": str(raw_copy)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
