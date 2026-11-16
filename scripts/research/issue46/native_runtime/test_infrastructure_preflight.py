import unittest
from unittest.mock import patch
from scripts.research.issue46.native_runtime import infrastructure_preflight as infra


class ClockPreflightTests(unittest.TestCase):
    def samples(self, rate=1., step=0.):
        return [dict(raw_s=float(i),monotonic_s=i*rate,realtime_s=1000+i*rate+(step if i>15 else 0.)) for i in range(36)]
    def test_agreeing_clocks_pass(self):self.assertTrue(infra.assess_clock(self.samples())['passed'])
    def test_observed_fast_clock_fails(self):self.assertIn('ADJUSTED_CLOCK_RATE',infra.assess_clock(self.samples(1.027))['failures'])
    def test_backward_wall_step_fails(self):self.assertIn('REALTIME_DISCONTINUITY',infra.assess_clock(self.samples(step=-.3))['failures'])
    def test_forward_wall_step_fails(self):self.assertFalse(infra.assess_clock(self.samples(step=.3))['passed'])
    def test_short_window_cannot_pass(self):self.assertFalse(infra.assess_clock(self.samples()[:4])['passed'])
    def test_nonfinite_data_cannot_pass(self):
        samples=self.samples();samples[4]['raw_s']=float('nan');self.assertFalse(infra.assess_clock(samples)['passed'])
    def test_missing_pty_does_not_wait_or_modify_mounts(self):
        with patch.object(infra.os,'openpty',side_effect=OSError(19,'No such device')),patch.object(infra.time,'sleep') as sleep:
            self.assertEqual(infra.inspect_runtime()['failures'],['PTY_UNAVAILABLE']);sleep.assert_not_called()
    def test_failure_records_not_run_before_raising(self):
        with patch.object(infra,'inspect_runtime',return_value={'passed':False,'failures':['ADJUSTED_CLOCK_RATE']}),patch('scripts.agent.path_governance.write_new_text') as write:
            with self.assertRaisesRegex(RuntimeError,'INFRASTRUCTURE_PREFLIGHT'):infra.require_healthy_runtime('/unused')
            self.assertEqual(write.call_count,2)
            self.assertIn('"physical_episode_started": false',write.call_args_list[1].args[1])


if __name__=='__main__':unittest.main()
