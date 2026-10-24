import unittest

from scripts.research.issue46_manipulation_integration.pr90_trial_gate import evaluate_trial_authorization

class TrialAuthorizationTests(unittest.TestCase):
    def base(self):
        return {
            "classification":"SIMULATION_ONLY_STATIC_GEOMETRY_AND_IK",
            "physics_steps":0,
            "mujoco":"3.3.6",
            "PR90_head":"facdaa0d6e5f8152168d2e812d864b660eac9f96",
            "PR90_model_sha256":"32353b81b9893fb049da55962b72a65bbae865189e1d404579502140515d6c45",
            "official_URDF_revision":"575cc6b988f976c23550e0db85aa1e5475d3652d",
            "official_URDF_sha256":"35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d",
            "object":{"diameter_m":0.070,"height_m":0.230,"mass_kg":0.550},
            "candidates":[
                {"candidate":{"id":"candidate-1"},"ik_status":"BLOCKED_IK","arm_limits_pass":False,
                 "path_status":"NOT_RUN","minimum_path_clearance_mm":None,"endpoint_open_clear":False}
            ],
            "refined_closures":[]
        }

    def test_current_style_blocker_never_authorizes_dynamics(self):
        result=self.base()
        gate=evaluate_trial_authorization(result)
        self.assertFalse(gate["dynamic_trial_authorized"])
        self.assertEqual("BLOCKED_STATIC_GATES",gate["status"])
        self.assertIn("SWEPT_APPROACH_NOT_PROVED_CLEAR",gate["candidate_decisions"][0]["blockers"])

    def test_all_static_gates_are_required(self):
        result=self.base()
        c=result["candidates"][0]
        c.update({"ik_status":"PASS_STATIC_IK","arm_limits_pass":True,"path_status":"PASS_STATIC_SWEPT",
                  "minimum_path_clearance_mm":2.1,"endpoint_open_clear":True})
        result["refined_closures"]=[{
            "candidate_id":"candidate-1","closure_status":"PASS_STATIC_PAD_FIRST","pad_first_topology":True,
            "first_bilateral_in_contact_band":{"contacts":[{"body":"R_hand_narrow3_Link"},{"body":"R_hand_wide3_Link"}]},
            "first_nonpad_object_contact":None,"first_internal_self_contact":None,
            "source_meshes_at_first_bilateral":[{"source_surface":{"intersects_cylinder_volume":False}}]
        }]
        self.assertTrue(evaluate_trial_authorization(result)["dynamic_trial_authorized"])

    def test_source_mesh_penetration_blocks_even_when_other_gates_pass(self):
        result=self.base()
        result["candidates"][0].update({"ik_status":"PASS_STATIC_IK","arm_limits_pass":True,"path_status":"PASS_STATIC_SWEPT",
                  "minimum_path_clearance_mm":2.1,"endpoint_open_clear":True})
        result["refined_closures"]=[{
            "candidate_id":"candidate-1","closure_status":"PASS_STATIC_PAD_FIRST","pad_first_topology":True,
            "first_bilateral_in_contact_band":{"contacts":[{"body":"R_hand_narrow3_Link"},{"body":"R_hand_wide3_Link"}]},
            "first_nonpad_object_contact":None,"first_internal_self_contact":None,
            "source_meshes_at_first_bilateral":[{"source_surface":{"intersects_cylinder_volume":True}}]
        }]
        gate=evaluate_trial_authorization(result)
        self.assertFalse(gate["dynamic_trial_authorized"])
        self.assertIn("ORIGINAL_SOURCE_STL_INTERSECTS_OBJECT",gate["candidate_decisions"][0]["blockers"])

    def test_nonzero_physics_input_is_rejected(self):
        result=self.base();result["physics_steps"]=1
        gate=evaluate_trial_authorization(result)
        self.assertTrue(any("STATIC_SCREEN_HAS_NONZERO_PHYSICS_STEPS" in x for x in gate["global_blockers"]))

if __name__=="__main__":
    unittest.main()
