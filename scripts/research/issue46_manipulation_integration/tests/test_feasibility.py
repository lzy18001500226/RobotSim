import dataclasses
import unittest

from scripts.research.issue46_manipulation_integration.approach import ToolPose
from scripts.research.issue46_manipulation_integration.contracts import PoseSample, PoseSource, TimeDomain
from scripts.research.issue46_manipulation_integration.feasibility import (
    ApproachObservation, ClosureObservation, ContactBand, FeasibilityThresholds,
    CollisionInventory, GeometryPair, GripperContract, GripperFeasibilityRequest, JointCommandProfile,
    ObjectGeometry, StaticGateStatus, evaluate_gripper_feasibility,
    collision_inventory_digest,
)


def make_case(*, linkage_first=False, blocked_approach=False, pad_reachable=True, missing_pair=False):
    obj = ObjectGeometry("bottle-1", "cylinder", (0.070, 0.230), "bottle_link",
                         ("bottle_collision",), 0.55)
    band = ContactBand("bottle_link", (0, 0, 1), -0.04, 0.04)
    limits = {"root_joint": (-1.0, 0.0), "follower_joint": (-1.0, 1.0)}
    profiles = (
        JointCommandProfile(0, 0.0, {"root_joint": 0.0, "follower_joint": 0.0}),
        JointCommandProfile(1, -0.5, {"root_joint": -0.5, "follower_joint": 0.5}),
        JointCommandProfile(2, -1.0, {"root_joint": -1.0, "follower_joint": 1.0}),
    )
    gripper = GripperContract(
        "candidate-b-static@sha", "root_joint", -1.0, 0.0, 0.0, -1.0,
        limits, profiles[0].joint_positions_rad, profiles, "candidate-b-kinematic-range.json", True,
        {"narrow": ("narrow_pad",), "wide": ("wide_pad",)},
        ("link_loop",),
    )
    object_pose = PoseSample("bottle-1", "world", (0.3, 0.04, 0.915), (0, 0, 0, 1),
                             10, PoseSource.SIM_GROUND_TRUTH, TimeDomain.SIMULATION,
                             model_identity="x2-model@sha", received_steady_ns=100)
    arm_limits = {"right_shoulder": (-2.0, 2.0)}
    closure_pairs = tuple((a, "bottle_collision") for a in gripper.all_collision_geom_names)
    path_pairs = (("right_arm", "bottle_collision"), ("gripper_base", "table"), *closure_pairs)
    approaches = []
    for i, phase in enumerate(("PREGRASP", "APPROACH")):
        observed_pairs = path_pairs[:-1] if missing_pair else path_pairs
        pairs = tuple(GeometryPair(a, b, -0.002 if blocked_approach and i == 1 else 0.010,
                                   "collision" if blocked_approach and i == 1 else "clearance")
                      for a, b in observed_pairs)
        approaches.append(ApproachObservation(
            i, phase, "x2-model@sha",
            ToolPose("world", (0.3, 0.04, 1.055 - i*0.14), (0, 0, 0, 1)),
            object_pose, {"right_shoulder": 0.3}, pairs,
        ))
    closure = []
    for i, cmd in enumerate((0.0, -0.5, -1.0)):
        both = i == 2 and pad_reachable
        link_hit = (linkage_first and i >= 1) or (both and i > 2)
        closure.append(ClosureObservation(
            i, cmd,
            {"narrow": 0.00005 if both else 0.006, "wide": 0.00005 if both else 0.006},
            {"narrow": ((0.035, 0.0, 0.0),) if both else (),
             "wide": ((-0.035, 0.0, 0.0),) if both else ()},
            {"narrow": ("narrow_pad",) if both else (), "wide": ("wide_pad",) if both else ()},
            0.00005 if link_hit else 0.004,
            closure_pairs,
        ))
    inventory_geoms = tuple(sorted({name for pair in path_pairs for name in pair}))
    inventory = CollisionInventory("x2-model@sha", inventory_geoms, path_pairs,
                                   collision_inventory_digest("x2-model@sha", inventory_geoms, path_pairs))
    request = GripperFeasibilityRequest(
        "x2-model@sha", obj, band, gripper, arm_limits, inventory, 150, 100,
        tuple(approaches), tuple(closure),
        0.025, FeasibilityThresholds(0.001, 0.0001, 0.015, 0.05),
    )
    return request


class GripperFeasibilityTests(unittest.TestCase):
    def test_geometry_pass_requires_complete_path_and_closure_evidence(self):
        result = evaluate_gripper_feasibility(make_case())
        self.assertEqual(result.status, StaticGateStatus.PASS_STATIC_GEOMETRY)
        self.assertTrue(result.feasible)
        self.assertEqual(result.first_bilateral_contact_step, 2)
        self.assertIsNone(result.first_linkage_contact_step)
        self.assertEqual(result.evidence_scope, "STATIC_GEOMETRY_ONLY")

    def test_rejects_loop_contact_before_bilateral_pads(self):
        result = evaluate_gripper_feasibility(make_case(linkage_first=True))
        self.assertFalse(result.feasible)
        self.assertIn("LINKAGE_CONTACT_PRECEDES_BILATERAL_PADS", result.blockers)

    def test_rejects_unreachable_small_object_even_if_approach_is_clear(self):
        result = evaluate_gripper_feasibility(make_case(pad_reachable=False))
        self.assertIn("BILATERAL_PAD_CONTACT_NOT_REACHED_IN_RANGE", result.blockers)

    def test_rejects_obstructed_approach_with_geom_pair_witness(self):
        result = evaluate_gripper_feasibility(make_case(blocked_approach=True))
        self.assertIn("APPROACH_COLLISION_OR_CLEARANCE", result.blockers)
        self.assertEqual(result.first_obstruction.key, ("bottle_collision", "right_arm"))

    def test_rejects_incomplete_collision_inventory(self):
        result = evaluate_gripper_feasibility(make_case(missing_pair=True))
        self.assertIn("APPROACH_COLLISION_PAIR_COVERAGE_INCOMPLETE", result.blockers)

    def test_contract_rejects_missing_or_out_of_range_open_joint_map(self):
        request = make_case()
        with self.assertRaisesRegex(ValueError, "every and only"):
            dataclasses.replace(request.gripper, open_joint_positions_rad={"root_joint": 0.0})
        with self.assertRaisesRegex(ValueError, "outside"):
            dataclasses.replace(request.gripper, open_joint_positions_rad={
                "root_joint": 0.1, "follower_joint": 0.0})

    def test_non_object_contact_is_not_silently_ignored(self):
        request = make_case()
        sample = dataclasses.replace(request.closure_samples[-1],
                                     other_contacts=(GeometryPair("link_loop", "table", -0.001, "unintended"),))
        changed = dataclasses.replace(request, closure_samples=(*request.closure_samples[:-1], sample))
        result = evaluate_gripper_feasibility(changed)
        self.assertIn("UNINTENDED_GRIPPER_OBJECT_CONTACT", result.blockers)

    def test_collision_inventory_digest_is_bound_to_pair_contents(self):
        request = make_case()
        inventory = request.collision_inventory
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            dataclasses.replace(inventory, enabled_pairs=inventory.enabled_pairs[:-1])

    def test_nonfinite_linkage_clearance_blocks_instead_of_looking_clear(self):
        request = make_case()
        sample = dataclasses.replace(request.closure_samples[-1], linkage_min_clearance_m=float("nan"))
        changed = dataclasses.replace(request, closure_samples=(*request.closure_samples[:-1], sample))
        result = evaluate_gripper_feasibility(changed)
        self.assertIn("LINKAGE_CLEARANCE_NONFINITE", result.blockers)

    def test_contact_points_must_fall_inside_requested_object_band(self):
        request = make_case()
        sample = dataclasses.replace(
            request.closure_samples[-1],
            pad_contact_points_object_m_by_side={"narrow": ((0.035, 0.0, 0.08),),
                                                 "wide": ((-0.035, 0.0, -0.08),)},
        )
        changed = dataclasses.replace(request, closure_samples=(*request.closure_samples[:-1], sample))
        result = evaluate_gripper_feasibility(changed)
        self.assertIn("BILATERAL_PAD_CONTACT_NOT_REACHED_IN_RANGE", result.blockers)

    def test_unverified_open_close_range_is_blocked(self):
        request = make_case()
        unverified = dataclasses.replace(request.gripper, range_verified_static=False)
        changed = dataclasses.replace(request, gripper=unverified)
        result = evaluate_gripper_feasibility(changed)
        self.assertIn("GRIPPER_OPEN_CLOSE_RANGE_UNVERIFIED", result.blockers)

    def test_stale_or_missing_object_pose_receipt_blocks_geometry_use(self):
        request = make_case()
        stale = dataclasses.replace(request.approach_samples[0].object_pose, received_steady_ns=0)
        first = dataclasses.replace(request.approach_samples[0], object_pose=stale)
        changed = dataclasses.replace(request, approach_samples=(first, *request.approach_samples[1:]))
        result = evaluate_gripper_feasibility(changed)
        self.assertIn("OBJECT_POSE_STALE_OR_UNTIMED", result.blockers)


if __name__ == "__main__":
    unittest.main()
