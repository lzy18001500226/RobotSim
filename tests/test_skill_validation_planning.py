from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from urllib.parse import unquote


SKILL_PATH = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "robotsim-validation-planning"
    / "SKILL.md"
)


class ValidationPlanningSkillTests(unittest.TestCase):
    def test_frontmatter_is_valid_and_has_required_fields(self) -> None:
        text = SKILL_PATH.read_text(encoding="utf-8")
        match = re.match(r"\A---\n(?P<frontmatter>.*?)\n---\n", text, re.DOTALL)
        self.assertIsNotNone(match, "SKILL.md must start with YAML frontmatter")

        fields: dict[str, str] = {}
        for line in match.group("frontmatter").splitlines():
            field = re.fullmatch(r'([a-z][a-z0-9_-]*): ("(?:[^"\\]|\\.)*")', line)
            self.assertIsNotNone(field, f"invalid simple YAML frontmatter field: {line!r}")
            name, encoded_value = field.groups()
            self.assertNotIn(name, fields, f"duplicate frontmatter field: {name}")
            value = json.loads(encoded_value)
            self.assertIsInstance(value, str)
            fields[name] = value

        self.assertEqual(fields.get("name"), SKILL_PATH.parent.name)
        self.assertTrue(fields.get("description", "").strip())

    def test_local_markdown_reference_paths_exist(self) -> None:
        text = SKILL_PATH.read_text(encoding="utf-8")
        targets = re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text)
        external_prefixes = ("http://", "https://", "mailto:", "#")
        local_targets = [target for target in targets if not target.startswith(external_prefixes)]
        self.assertTrue(local_targets, "SKILL.md should link to its RobotSim source documents")

        for target in local_targets:
            path = unquote(target.split("#", 1)[0])
            self.assertTrue(path, f"local reference must name a file: {target}")
            resolved = (SKILL_PATH.parent / path).resolve()
            self.assertTrue(resolved.is_file(), f"missing local reference: {target}")


if __name__ == "__main__":
    unittest.main()
