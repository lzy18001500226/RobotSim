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


def _simple_yaml_scalar(raw_value: str) -> str | None:
    """Read a simple YAML string scalar; this is not a general YAML parser."""
    value = raw_value.strip()
    if not value:
        return ""

    if value.startswith('"'):
        quoted = re.fullmatch(r'("(?:\\.|[^"\\])*")(?:\s+#.*)?', value)
        if quoted is None:
            return None
        try:
            parsed = json.loads(quoted.group(1))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, str) else None

    if value.startswith("'"):
        quoted = re.fullmatch(r"'((?:''|[^'])*)'(?:\s+#.*)?", value)
        if quoted is None:
            return None
        return quoted.group(1).replace("''", "'")

    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


class ValidationPlanningSkillTests(unittest.TestCase):
    def test_required_skill_frontmatter_metadata(self) -> None:
        text = SKILL_PATH.read_text(encoding="utf-8")
        lines = text.splitlines()
        self.assertTrue(lines and lines[0].strip() == "---", "SKILL.md must start with frontmatter")
        closing = next(
            (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
            None,
        )
        self.assertIsNotNone(closing, "SKILL.md frontmatter must have a closing delimiter")

        fields: dict[str, str] = {}
        for line in lines[1:closing]:
            field = re.fullmatch(r"(name|description):(?:[ \t]*(.*))?", line)
            if field is None:
                continue
            name, raw_value = field.groups()
            self.assertNotIn(name, fields, f"duplicate required frontmatter field: {name}")
            value = _simple_yaml_scalar(raw_value or "")
            self.assertIsNotNone(value, f"unsupported scalar for required field {name!r}")
            fields[name] = value

        self.assertIn("name", fields)
        self.assertIn("description", fields)
        self.assertTrue(fields["name"].strip())
        self.assertTrue(fields["description"].strip())
        self.assertEqual(fields.get("name"), SKILL_PATH.parent.name)

    def test_required_scalar_reader_accepts_quoted_and_plain_values(self) -> None:
        self.assertEqual(_simple_yaml_scalar('"double quoted"'), "double quoted")
        self.assertEqual(_simple_yaml_scalar("'single quoted'"), "single quoted")
        self.assertEqual(_simple_yaml_scalar("plain-value"), "plain-value")

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
