import unittest
from scripts.research.issue46.scene.table_edge_grid import initial_positions, observation_goal

BOUNDS = {"min": [1.3213823326277394, -1.842376733945703, -.010973995898385602],
          "max": [1.8778913084723745, -.9405936266511858, .5468309428515006]}

class TableGridTests(unittest.TestCase):
    def test_all_eight_cm_steps_and_far_endpoint(self):
        rows = initial_positions(BOUNDS, [1.42642523479425, -1.6415328181824196])
        self.assertEqual(len(rows), 11)
        for a, b in zip(rows[:9], rows[1:10]):
            self.assertAlmostEqual(b["initial_xy_m"][1]-a["initial_xy_m"][1], .08)
        self.assertAlmostEqual(rows[-1]["initial_xy_m"][1], BOUNDS["max"][1]-.055)
        self.assertGreater(rows[-1]["offset_xy_m"][1], .6)
        self.assertTrue(all(r["initialization_only_not_control_target"] for r in rows))

    def test_observation_depends_on_table_not_bottle_position(self):
        goal = observation_goal(BOUNDS)
        self.assertAlmostEqual(goal["xy_m"][0], .8213823326277394)
        self.assertAlmostEqual(goal["xy_m"][1], -1.3914851802984445)
        self.assertEqual(goal["yaw_rad"], 0.)
        self.assertEqual(goal["source"], "TABLE_GEOMETRY_ONLY")
        for y in (-1.78, -1.2, -1.0):
            initial_positions(BOUNDS, [1.42642523479425, y])
            self.assertEqual(goal, observation_goal(BOUNDS))

    def test_invalid_table_or_centerline_rejected(self):
        for bounds in ({"min": [0,0,0], "max": [1,.5,1]},
                       {"min": [0,0,0], "max": [1,float("nan"),1]}):
            with self.assertRaises(ValueError): observation_goal(bounds)
        with self.assertRaises(ValueError):
            initial_positions(BOUNDS, [BOUNDS["min"][0], -1.5])

if __name__ == "__main__": unittest.main()
