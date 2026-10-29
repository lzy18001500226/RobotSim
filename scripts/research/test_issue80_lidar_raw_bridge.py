from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from issue80_lidar_raw_bridge import RAW_SCHEMA, decode_frame, point_bytes, read_new_frames


class RawBridgeTests(unittest.TestCase):
    def test_world_coordinate_field_is_explicit_and_does_not_relabel_local_points(self) -> None:
        payload = {
            "schema": RAW_SCHEMA,
            "xyz": [[1.0, 2.0, 3.0]],
            "world_xyz": [[4.0, 5.0, 6.0]],
        }
        self.assertEqual(decode_frame(payload), [(1.0, 2.0, 3.0)])
        self.assertEqual(decode_frame(payload, coordinate_field="world_xyz"), [(4.0, 5.0, 6.0)])
        with self.assertRaisesRegex(ValueError, "world_xyz"):
            decode_frame({"schema": RAW_SCHEMA, "xyz": [[1.0, 2.0, 3.0]]}, coordinate_field="world_xyz")
    def test_decodes_only_finite_xyz(self) -> None:
        frame = {"schema": RAW_SCHEMA, "xyz": [[1, 2, 3], None, ["nan", 0, 1], [4.0, 5.0, 6.0]]}
        self.assertEqual(decode_frame(frame), [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
        self.assertEqual(len(point_bytes(decode_frame(frame))), 24)

    def test_defers_partial_jsonl_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            good = {"schema": RAW_SCHEMA, "xyz": [[1, 2, 3]]}
            path.write_text(json.dumps(good) + "\n{", encoding="utf-8")
            offset, frames = read_new_frames(path, 0)
            self.assertEqual(frames, [good])
            with path.open("a", encoding="utf-8") as stream:
                stream.write("}\n")
            _, frames = read_new_frames(path, offset)
            self.assertEqual(frames, [{}])


if __name__ == "__main__":
    unittest.main()
