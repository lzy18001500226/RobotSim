import unittest
from scripts.research.issue46.perception.native_rgb_video_stream import Frames, source_state, encoder_command

class StreamContractTests(unittest.TestCase):
    def test_fresh_stale_future_ended_and_replay_are_distinct(self):
        now=10_000_000_000
        self.assertEqual(source_state("live",now-100_000_000,now),"LIVE")
        self.assertEqual(source_state("live",now-201_000_000,now),"STALE / NOT LIVE")
        self.assertEqual(source_state("live",now+1,now),"STALE / NOT LIVE")
        self.assertEqual(source_state("live",now,now,ended=True),"RUN ENDED / LAST FRAME")
        self.assertEqual(source_state("replay",now,now),"REPLAY / NO PHYSICS")
        self.assertEqual(source_state("replay",now,now,failure="FK mismatch"),"OBSERVER ERROR")

    def test_buffer_replaces_frame_and_never_builds_a_queue(self):
        f=Frames("replay")
        for i in range(100):
            f.put(bytes([i]),{"host_monotonic_ns":i,"sim_time_s":i/10})
        rgb,status=f.snapshot()
        self.assertEqual(rgb,bytes([99]))
        self.assertEqual(status["frames"],100)
        self.assertEqual(status["sim_time_s"],9.9)
        self.assertTrue(status["observer_only"])
        self.assertEqual(status["new_mj_step_calls"],0)
        self.assertEqual(status["state"],"REPLAY / NO PHYSICS")

    def test_low_latency_browser_video_contract(self):
        cmd=encoder_command("/existing/ffmpeg",10)
        self.assertEqual(cmd[cmd.index("-c:v")+1],"libx264")
        self.assertEqual(cmd[cmd.index("-threads")+1],"1")
        self.assertIn("empty_moov",cmd[cmd.index("-movflags")+1])
        self.assertEqual(cmd[-2:],["mp4","pipe:1"])

if __name__=="__main__": unittest.main()
