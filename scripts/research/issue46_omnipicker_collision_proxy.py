#!/usr/bin/env python3
"""Single source-derived fingertip collision proxy diagnostic for Issue #46.

The pinned visual meshes, link inertias, joint tree, mimic pair, operational
controller, actuator caps, bottle, table, and contact parameters remain owned
by the existing isolated P0 runner. Only the two distal fingertip collision
meshes are replaced in this SIMULATION_ONLY model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import issue46_omnipicker_p0_isolated as p0  # noqa: E402

BASELINE_ROOT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "omnipicker-p0-contact-geometry-20261011"
)
BASELINE_GEOMETRY = BASELINE_ROOT / "fixed-object-baseline/FREE_CONTACT_GEOMETRY.json"
BASELINE_FREE_RESULT = BASELINE_ROOT / "free-cylinder-reference/C_RESULT.json"
PROXY_DEPTH_CEILING_M = 0.0005
PAD_MESHES = {
    "R_hand_narrow3_Link": "narrow3_Link.stl",
    "R_hand_wide3_Link": "wide3_Link.stl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_binary_stl(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) < 84:
        raise ValueError(f"Truncated binary STL: {path}")
    count = struct.unpack_from("<I", payload, 80)[0]
    if len(payload) != 84 + 50 * count:
        raise ValueError(f"Expected a binary STL: {path}")
    triangles = np.empty((count, 3, 3), dtype=np.float64)
    offset = 84
    for index in range(count):
        row = struct.unpack_from("<12fH", payload, offset)
        triangles[index] = np.asarray(row[3:12], dtype=np.float64).reshape(3, 3)
        offset += 50
    return triangles


def ray_inside(point: np.ndarray, triangles: np.ndarray) -> bool:
    directions = np.asarray([[1.0, 0.371, 0.529], [0.217, 1.0, 0.413],
                             [0.319, 0.577, 1.0]], dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    votes = 0
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    for direction in directions:
        ray = np.broadcast_to(direction, edge2.shape)
        h = np.cross(ray, edge2)
        det = np.einsum("ij,ij->i", edge1, h)
        nonparallel = np.abs(det) > 1e-12
        inv_det = np.zeros_like(det)
        inv_det[nonparallel] = 1.0 / det[nonparallel]
        s = point - triangles[:, 0]
        u = inv_det * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, edge1)
        v = inv_det * np.einsum("ij,ij->i", ray, q)
        t = inv_det * np.einsum("ij,ij->i", edge2, q)
        hits = nonparallel & (u > 1e-10) & (v > 1e-10) & (u + v < 1.0 - 1e-10) & (t > 1e-10)
        votes += int(np.count_nonzero(hits) % 2)
    return votes >= 2


def write_binary_stl(path: Path, triangles: np.ndarray) -> None:
    header = b"RobotSim SIMULATION_ONLY source-facet triangular-prism proxy"
    header = header[:80].ljust(80, b" ")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-15:
                raise ValueError("Degenerate proxy facet")
            normal /= norm
            values = np.concatenate((normal, triangle.reshape(-1)))
            stream.write(struct.pack("<12fH", *values.tolist(), 0))


def prism_from_source_facet(source_mesh: Path, triangle_index: int,
                            proxy_path: Path) -> dict[str, Any]:
    source_triangles = read_binary_stl(source_mesh)
    if not 0 <= triangle_index < len(source_triangles):
        raise ValueError(f"Source triangle index out of range: {triangle_index}")
    source_volume = float(np.einsum(
        "ij,ij->i", source_triangles[:, 0],
        np.cross(source_triangles[:, 1], source_triangles[:, 2])).sum() / 6.0)
    if source_volume <= 0.0:
        raise ValueError(f"Source STL winding is not outward: signed volume {source_volume}")
    front = source_triangles[triangle_index].copy()
    normal = np.cross(front[1] - front[0], front[2] - front[0])
    area = float(np.linalg.norm(normal) * 0.5)
    normal /= np.linalg.norm(normal)
    sample_barycentric = np.asarray([
        [1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6], [0.45, 0.45, 0.10], [0.45, 0.10, 0.45],
        [0.10, 0.45, 0.45],
    ])

    def interior_status(depth: float) -> list[bool]:
        points = sample_barycentric @ (front - depth * normal)
        return [ray_inside(point, source_triangles) for point in points]

    # Derive a single inset from the source volume. The initial 0.5 mm cap was
    # rejected when one of seven surface samples exited the original STL.
    low, high = 0.0, PROXY_DEPTH_CEILING_M
    for _ in range(24):
        middle = (low + high) * 0.5
        if all(interior_status(middle)):
            low = middle
        else:
            high = middle
    if low <= 1e-7:
        raise RuntimeError("Source facet has no validated positive inward extrusion depth")
    proxy_thickness = 0.9 * low
    back = front - proxy_thickness * normal

    # Sample interior sections to verify that the selected thickness stays in
    # the original mesh volume after applying a 10% depth margin.
    interior_samples = []
    for depth_fraction in (0.25, 0.5, 0.75, 1.0):
        statuses = interior_status(proxy_thickness * depth_fraction)
        interior_samples.append({"depth_fraction": depth_fraction,
                                 "inside_source_stl_count": sum(statuses),
                                 "sample_count": len(statuses)})
    if any(row["inside_source_stl_count"] != row["sample_count"] for row in interior_samples):
        raise RuntimeError(f"Prism extrusion leaves source mesh interior: {interior_samples}")

    # Closed triangular prism: source surface triangle, reversed back triangle,
    # and three side quads. Its only outward/contact face exactly matches STL.
    facets = [front, back[[2, 1, 0]]]
    for index in range(3):
        next_index = (index + 1) % 3
        facets.extend([
            np.asarray([front[index], front[next_index], back[next_index]]),
            np.asarray([front[index], back[next_index], back[index]]),
        ])
    proxy = np.asarray(facets, dtype=np.float64)
    proxy_volume = float(np.einsum(
        "ij,ij->i", proxy[:, 0], np.cross(proxy[:, 1], proxy[:, 2])).sum() / 6.0)
    if proxy_volume < 0.0:
        proxy = proxy[:, [2, 1, 0]]
        proxy_volume = -proxy_volume
    write_binary_stl(proxy_path, proxy)
    outer_face_vertex_distances = np.linalg.norm(
        front[:, None, :] - proxy[0][None, :, :], axis=2)
    return {
        "source_stl": str(source_mesh), "source_stl_sha256": sha256(source_mesh),
        "source_triangle_count": len(source_triangles),
        "source_signed_volume_m3": source_volume,
        "source_triangle_index": triangle_index,
        "source_contact_triangle_local_vertices_m": front.tolist(),
        "source_contact_triangle_area_mm2": area * 1e6,
        "source_contact_triangle_normal_local": normal.tolist(),
        "proxy_stl": str(proxy_path), "proxy_stl_sha256": sha256(proxy_path),
        "proxy_triangle_count": len(proxy), "proxy_convex_hull_vertex_count": 6,
        "proxy_outer_face_vertex_error_m": float(
            np.max(np.min(outer_face_vertex_distances, axis=1))),
        "proxy_thickness_m": proxy_thickness,
        "source_derived_maximum_sampled_inset_m": low,
        "source_inset_depth_headroom_fraction": 0.10,
        "proxy_volume_m3": proxy_volume,
        "proxy_surface_area_contact_face_mm2": area * 1e6,
        "interior_samples": interior_samples,
        "geometry_rationale": "Use the exact recorded source-STL contact facet as the prism outer face; derive the maximum inward extrusion that keeps seven barycentric samples inside the source volume, then use 90% of that depth. No rectangular expansion or outward offset is used.",
    }


def replace_pad_collision_mesh(urdf_path: Path, pad: str, proxy_mesh: Path) -> None:
    tree = ET.parse(urdf_path)
    robot = tree.getroot()
    link = next((node for node in robot.findall("link") if node.get("name") == pad), None)
    if link is None:
        raise ValueError(f"Missing source fingertip link {pad}")
    collisions = link.findall("collision")
    if len(collisions) != 1:
        raise ValueError(f"Expected exactly one fingertip collision geom on {pad}, got {len(collisions)}")
    collision = collisions[0]
    geometry = collision.find("geometry")
    if geometry is None or geometry.find("mesh") is None:
        raise ValueError(f"Expected original collision mesh on {pad}")
    mesh = geometry.find("mesh")
    if "scale" in mesh.attrib:
        raise ValueError(f"Unexpected source mesh scale on {pad}")
    mesh.set("filename", proxy_mesh.name)
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)


def body_properties(model: mujoco.MjModel, names: list[str]) -> dict[str, Any]:
    result = {}
    for name in names:
        bid = p0.accepted.body_id(model, name)
        result[name] = {"mass_kg": float(model.body_mass[bid]),
                        "inertial_position_m": model.body_ipos[bid].tolist(),
                        "inertial_quaternion_wxyz": model.body_iquat[bid].tolist(),
                        "inertia_diagonal_kg_m2": model.body_inertia[bid].tolist()}
    return result


def joint_model_equal(control: mujoco.MjModel, proxy: mujoco.MjModel) -> bool:
    if (control.njnt, control.nq, control.nv, control.nu, control.neq) != (
            proxy.njnt, proxy.nq, proxy.nv, proxy.nu, proxy.neq):
        return False
    names_control = [mujoco.mj_id2name(control, mujoco.mjtObj.mjOBJ_JOINT, i)
                     for i in range(control.njnt)]
    names_proxy = [mujoco.mj_id2name(proxy, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(proxy.njnt)]
    if names_control != names_proxy:
        return False
    for attribute in ("jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_limited",
                      "jnt_range", "jnt_axis", "jnt_pos", "jnt_stiffness",
                      "jnt_actfrclimited", "jnt_actfrcrange", "jnt_actgravcomp",
                      "jnt_margin", "jnt_solimp", "jnt_solref", "qpos0",
                      "dof_damping", "dof_frictionloss", "dof_armature",
                      "eq_type", "eq_obj1id", "eq_obj2id", "eq_data",
                      "eq_active0", "eq_solimp", "eq_solref"):
        if not np.array_equal(getattr(control, attribute), getattr(proxy, attribute)):
            return False
    for attribute in ("actuator_trntype", "actuator_trnid", "actuator_gear",
                      "actuator_ctrllimited", "actuator_ctrlrange",
                      "actuator_forcelimited", "actuator_forcerange",
                      "actuator_gainprm", "actuator_biasprm", "actuator_dynprm",
                      "actuator_dyntype", "actuator_gaintype", "actuator_biastype",
                      "actuator_actadr", "actuator_actnum", "actuator_actlimited",
                      "actuator_actrange"):
        if not np.array_equal(getattr(control, attribute), getattr(proxy, attribute)):
            return False
    return True


def native_library_identity() -> list[dict[str, str]]:
    package = Path(mujoco.__file__).resolve().parent
    files = sorted(package.glob("libmujoco.so*"))
    return [{"path": str(path), "sha256": sha256(path)} for path in files]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=BASELINE_ROOT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_root = p0.SOURCE_ROOT.resolve()
    source_urdf = p0.SOURCE_URDF.resolve()
    if not source_root.is_dir() or not source_urdf.is_file():
        raise FileNotFoundError(f"Pinned X2 source is unavailable: {source_urdf}")
    source_head = __import__("subprocess").check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    source_status = __import__("subprocess").check_output(
        ["git", "-C", str(source_root), "status", "--porcelain"], text=True).strip()
    if source_head != p0.accepted.X2_PIN or source_status:
        raise RuntimeError(f"Pinned vendor checkout mismatch/dirty: {source_head}, dirty={bool(source_status)}")

    baseline_geometry_path = args.baseline / "fixed-object-baseline/FREE_CONTACT_GEOMETRY.json"
    if not baseline_geometry_path.is_file():
        raise FileNotFoundError(baseline_geometry_path)
    baseline_geometry = json.loads(baseline_geometry_path.read_text(encoding="utf-8"))
    triangle_by_body = {
        row["finger_body"]: row["nearest_original_finger_stl_surface"]["source_triangle_index"]
        for row in baseline_geometry["contacts"]
    }
    if set(triangle_by_body) != set(PAD_MESHES):
        raise RuntimeError(f"Baseline does not identify exactly the two pad facets: {triangle_by_body}")

    subtree_path = output / "generated_source_right_wrist_subtree.urdf"
    subtree_identity = p0.write_wrist_subtree_urdf(subtree_path)
    proxy_geometry = {}
    for body, filename in PAD_MESHES.items():
        source_mesh = p0.mesh_source(filename)
        proxy_mesh = output / f"proxy_{filename}"
        proxy_geometry[body] = prism_from_source_facet(
            source_mesh, int(triangle_by_body[body]), proxy_mesh)
        replace_pad_collision_mesh(subtree_path, body, proxy_mesh)

    canonical = p0.accepted.load_canonical_module(p0.CANONICAL_HELPER)
    pose = p0.source_pose()
    spec, model, build = p0.build_model(subtree_path, pose, canonical, include_bottle=True)
    spec.to_file(str(output / "SIMULATION_ONLY_COLLISION_PROXY.xml"))

    # Compare against the same source subtree with original fingertip collision
    # meshes. Compile it independently; no vendor file or historical evidence
    # is edited.
    control_subtree = output / "generated_source_right_wrist_control.urdf"
    control_identity = p0.write_wrist_subtree_urdf(control_subtree)
    control_spec, control_model, _ = p0.build_model(control_subtree, pose, canonical, include_bottle=True)
    control_spec.to_file(str(output / "SOURCE_FAITHFUL_CONTROL.xml"))
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                  for body_id in range(1, model.nbody)]
    control_bodies = body_properties(control_model, body_names)
    proxy_bodies = body_properties(model, body_names)
    mass_properties_equal = control_bodies == proxy_bodies
    joints_equal = joint_model_equal(control_model, model)
    control_controller = p0.right_controller(control_model)
    proxy_controller = p0.right_controller(model)
    controller_equal = control_controller == proxy_controller
    source_joint_ranges = {}
    joint_ranges_unchanged = True
    for joint in p0.accepted.DRIVER_FOLLOWER["right"][:2]:
        jid = p0.accepted.joint_id(model, joint)
        ctrl_jid = p0.accepted.joint_id(control_model, joint)
        source_joint_ranges[joint] = {
            "proxy_range_rad": model.jnt_range[jid].tolist(),
            "control_range_rad": control_model.jnt_range[ctrl_jid].tolist(),
            "proxy_limited": bool(model.jnt_limited[jid]),
            "control_limited": bool(control_model.jnt_limited[ctrl_jid]),
        }
        joint_ranges_unchanged &= (
            np.array_equal(model.jnt_range[jid], control_model.jnt_range[ctrl_jid])
            and bool(model.jnt_limited[jid]) == bool(control_model.jnt_limited[ctrl_jid]))

    identity = {
        "classification": "SIMULATION_ONLY_COLLISION_PROXY",
        "robot_branch": __import__("subprocess").check_output(
            ["git", "-C", str(SCRIPT_DIR.parents[1]), "branch", "--show-current"], text=True).strip(),
        "robot_head": __import__("subprocess").check_output(
            ["git", "-C", str(SCRIPT_DIR.parents[1]), "rev-parse", "HEAD"], text=True).strip(),
        "source_repository": p0.accepted.X2_REPO, "source_commit": source_head,
        "source_worktree_clean": not bool(source_status),
        "source_urdf": str(source_urdf), "source_urdf_sha256": sha256(source_urdf),
        "mujoco_python": mujoco.__version__, "mujoco_native": mujoco.mj_versionString(),
        "native_library_sha256": native_library_identity(),
        "numpy": np.__version__, "platform": platform.platform(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "timestep_s": float(model.opt.timestep), "integrator": int(model.opt.integrator),
        "solver": int(model.opt.solver), "solver_iterations": int(model.opt.iterations),
        "gravity_m_s2": model.opt.gravity.tolist(),
        "baseline_free_result": str(args.baseline / "free-cylinder-reference/C_RESULT.json"),
        "baseline_free_result_sha256": sha256(args.baseline / "free-cylinder-reference/C_RESULT.json"),
        "baseline_contact_geometry": str(baseline_geometry_path),
        "baseline_contact_geometry_sha256": sha256(baseline_geometry_path),
        "pad_collision_geometry": proxy_geometry,
        "only_replaced_collision_bodies": sorted(PAD_MESHES),
        "visual_meshes_unchanged": True,
        "source_joint_ranges_unchanged": joint_ranges_unchanged,
        "source_joint_ranges": source_joint_ranges,
        "entire_joint_model_unchanged": joints_equal,
        "control_joint_names": p0.accepted.joint_names(control_model),
        "proxy_joint_names": p0.accepted.joint_names(model),
        "body_mass_inertia_unchanged": mass_properties_equal,
        "controller_profile_unchanged": controller_equal,
        "controller_profile": proxy_controller,
        "control_pad_body_properties": control_bodies,
        "proxy_pad_body_properties": proxy_bodies,
        "control_subtree_sha256": control_identity["generated_urdf_sha256"],
        "proxy_subtree_sha256": p0.sha256(subtree_path),
        "original_source_subtree_identity": control_identity,
        "historical_simulation_only_exclusions": [list(x) for x in p0.accepted.HISTORICAL_EXCLUSIONS],
        "historical_exclusions_active_in_isolated_subtree": [],
        "historical_exclusion_scope_note": "The detached right-wrist subtree omits the body pairs covered by the 11 accepted full-X2 exclusions; the list is recorded, no exclusion is added or changed in this fixture.",
        "new_collision_exclusions": [],
        "no_weld_or_object_attachment": True,
        "no_bottle_qpos_writes_during_rollout": True,
        "bottle": {"diameter_m": 2 * p0.accepted.CYLINDER_RADIUS,
                   "height_m": 2 * p0.accepted.CYLINDER_HALF_HEIGHT,
                   "mass_kg": p0.accepted.CYLINDER_MASS,
                   "friction": p0.accepted.CYLINDER_FRICTION},
    }
    p0.write_json(output / "runtime_identity.json", identity)

    geometry = p0.geometry_gate(model, pose, output, canonical)
    geometry["collision_model_classification"] = "SIMULATION_ONLY_COLLISION_PROXY"
    geometry["source_faithful_geometry"] = False
    geometry["proxy_collision_bodies"] = sorted(PAD_MESHES)
    geometry["proxy_surface_correspondence"] = {
        body: {"source_triangle_index": details["source_triangle_index"],
               "source_contact_triangle_area_mm2": details["source_contact_triangle_area_mm2"],
               "proxy_outer_face_vertex_error_m": details["proxy_outer_face_vertex_error_m"],
               "proxy_thickness_m": details["proxy_thickness_m"],
               "interior_samples": details["interior_samples"]}
        for body, details in proxy_geometry.items()
    }
    geometry["new_collision_exclusions"] = []
    p0.write_json(output / "B2_STATIC_GEOMETRY_GATE.json", geometry)
    if geometry["status"] == "PASS":
        dynamic = p0.physical_run(spec, model, build, geometry, output, contact_only=False)
    else:
        dynamic = {"status": "NOT_RUN", "reason": "The proxy failed the static source-limited collision/approach/closure gate."}
    p0.write_json(output / "C_RESULT.json", dynamic)

    dynamic_failure = dynamic.get("failure_classification")
    failure_map = {
        "PRELOAD_EXCESSIVE_SLIP": "PROXY_LATERAL_ESCAPE",
        "PRELOAD_FORCE_NOT_REACHED": "PROXY_PRELOAD_FAIL",
        "PRELOAD_TRACKING_LIMIT": "PROXY_PRELOAD_FAIL",
        "BILATERAL_CONTACT_FAIL": "PROXY_CONTACT_GEOMETRY_FAIL",
    }
    if geometry["status"] != "PASS":
        experiment_failure = "PROXY_CONTACT_GEOMETRY_FAIL"
    elif dynamic.get("status") == "NOT_RUN":
        experiment_failure = "PROXY_CONTACT_GEOMETRY_FAIL"
    elif dynamic.get("status") == "FAIL":
        experiment_failure = failure_map.get(dynamic_failure, "PROXY_PRELOAD_FAIL")
    else:
        experiment_failure = None
    result = {
        "status": "PASS" if dynamic.get("final_gate_checks", {}).get("lift_50mm") is True else
                  "FAIL" if dynamic.get("status") == "FAIL" else
                  "NOT_RUN" if dynamic.get("status") == "NOT_RUN" else "INCOMPLETE",
        "static_geometry_status": geometry["status"],
        "physical_status": dynamic.get("status", "NOT_RUN"),
        "failure_classification": experiment_failure,
        "physical_failure_detail": dynamic_failure,
        "maximum_lift_m": dynamic.get("maximum_lift_above_initial_m", 0.0),
        "maximum_relative_slip_m": dynamic.get("max_relative_slip_from_hold_start_m"),
        "maximum_penetration_m": dynamic.get("max_contact_penetration_m"),
        "source_limit_violations": dynamic.get("source_limit_violations", []),
        "source_joint_ranges_unchanged": joint_ranges_unchanged,
        "body_mass_inertia_unchanged": mass_properties_equal,
        "historical_exclusion_policy_count": 11,
        "historical_exclusions_added_to_isolated_fixture": 0,
        "new_collision_exclusions": [],
        "welds_or_equalities": [], "mocap_carry": False,
        "bottle_qpos_writes_during_rollout": 0,
        "geometry_result": "B2_STATIC_GEOMETRY_GATE.json",
        "dynamic_result": "C_RESULT.json",
        "identity": "runtime_identity.json",
        "reproduction_command": (
            "PYTHONPATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib/python3.10/site-packages:/tmp/robotsim-issue46-omnipicker-p0-pin:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages "
            "LD_LIBRARY_PATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib:/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib64 MUJOCO_GL=egl "
            f"python3 {Path(__file__).resolve()} --output {output}")
    }
    p0.write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
