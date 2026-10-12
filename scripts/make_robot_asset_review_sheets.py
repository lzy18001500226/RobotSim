#!/usr/bin/env python3
"""Compose human-review sheets from the existing static asset PNG evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "assets/robots/registry.json"
HASHES_PATH = ROOT / "assets/robots/source_hashes.json"
COMPILED_PATH = ROOT / "assets/robots/compiled_model_signatures.json"
CONTACT_SHEET = "review_contact_sheet.png"
PANEL_SOURCES = (
    ("overview_front.png", "Full body - front"),
    ("overview_side.png", "Full body - side"),
    ("overview_rear.png", "Full body - rear"),
    ("left_hand_closeup.png", "Left hand close-up"),
    ("right_hand_closeup.png", "Right hand close-up"),
    ("left_hand_front.png", "Left palm/front"),
    ("right_hand_front.png", "Right palm/front"),
    ("visual_collision_overlay.png", "Visual + collision overlay"),
    ("visual_geometry_only.png", "Visual geometry only"),
    ("collision_geometry_only.png", "Collision geometry only"),
    ("driver_lower_limits_static.png", "Static driver lower-limit pose"),
    ("driver_upper_limits_static.png", "Static driver upper-limit pose"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_manifest_hash(registry: dict[str, Any], hashes: dict[str, Any], compiled: dict[str, Any]) -> str:
    payload = {"registry": registry, "source_hashes": hashes, "compiled": compiled}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def wrap_line(draw: ImageDraw.ImageDraw, text: str, selected_font: ImageFont.ImageFont,
              max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposed = word if not current else f"{current} {word}"
        if current and draw.textbbox((0, 0), proposed, font=selected_font)[2] > max_width:
            lines.append(current)
            current = word
        else:
            current = proposed
    if current:
        lines.append(current)
    return lines


def mount_text(mounts: dict[str, Any]) -> str:
    if not mounts:
        return "Mounts: none (base model only)"
    return "Mounts: " + "; ".join(f"{side}={description}" for side, description in mounts.items())


def panel_paths(directory: Path, metadata: dict[str, Any]) -> list[tuple[Path, str]]:
    image_records = {record["path"]: record["sha256"] for record in metadata.get("generated_images", [])}
    selected: list[tuple[Path, str]] = []
    for filename, label in PANEL_SOURCES:
        path = directory / filename
        if not path.is_file():
            continue
        expected = image_records.get(filename)
        if expected is None:
            raise ValueError(f"PNG has no recorded source hash in metadata: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"PNG SHA-256 mismatch for {path}: expected {expected}, got {actual}")
        selected.append((path, label))
    required = {"overview_front.png", "overview_side.png", "right_hand_closeup.png",
                "visual_collision_overlay.png", "visual_geometry_only.png", "collision_geometry_only.png"}
    missing = sorted(required - {path.name for path, _ in selected})
    if missing:
        raise ValueError(f"Required existing review PNGs are missing in {directory}: {missing}")
    return selected


def draw_sheet(asset: dict[str, Any], source_profile: dict[str, Any], metadata: dict[str, Any],
               manifest_hash: str, source_dir: Path, output: Path, overwrite: bool) -> dict[str, Any]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing review sheet: {output}")
    if metadata.get("asset_id") != asset["asset_id"]:
        raise ValueError(f"Evidence/registry asset ID mismatch at {source_dir}")
    if metadata.get("source_model_path") != asset["model_path"]:
        raise ValueError(f"Evidence/registry model path mismatch for {asset['asset_id']}")
    if metadata.get("source_model_sha256") is None:
        raise ValueError(f"Missing source model SHA-256 for {asset['asset_id']}")

    panels = panel_paths(source_dir, metadata)
    columns = 3
    tile_w, image_h, caption_h = 420, 286, 38
    tile_h = image_h + caption_h
    rows = math.ceil(len(panels) / columns)
    width = columns * tile_w
    header_height = 178
    canvas = Image.new("RGB", (width, header_height + rows * tile_h), (245, 247, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = font(19)
    detail_font = font(12)
    panel_font = font(13)
    small_font = font(11)

    draw.rectangle((0, 0, width, header_height), fill=(24, 35, 43))
    draw.text((18, 12), f"{asset['asset_id']}  |  {asset['review_state']}", font=title_font, fill=(255, 255, 255))
    details = [
        f"Source: {source_profile['repository']} @ {source_profile['commit']}",
        f"Model: {asset['model_path']} | model SHA-256: {metadata['source_model_sha256']}",
        f"Assembly: {asset['assembly_kind']}",
        mount_text(asset.get("mounts", {})),
        f"Registry manifest SHA-256: {manifest_hash}",
        "Evidence reused from existing static PNGs; no new model render or physics steps.",
    ]
    y = 43
    for detail in details:
        for line in wrap_line(draw, detail, detail_font, width - 36):
            draw.text((18, y), line, font=detail_font, fill=(222, 231, 236))
            y += 17

    for index, (path, caption) in enumerate(panels):
        x = (index % columns) * tile_w
        y = header_height + (index // columns) * tile_h
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), fill=(255, 255, 255), outline=(199, 207, 211))
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_w - 12, image_h - 8), Image.Resampling.LANCZOS)
            px = x + (tile_w - image.width) // 2
            py = y + 4 + (image_h - 8 - image.height) // 2
            canvas.paste(image, (px, py))
        label_lines = wrap_line(draw, caption, panel_font, tile_w - 18)
        line_y = y + image_h + 5
        for line in label_lines[:2]:
            draw.text((x + 9, line_y), line, font=panel_font, fill=(33, 43, 49))
            line_y += 15
        draw.text((x + 9, y + tile_h - 14), path.name, font=small_font, fill=(99, 109, 114))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return {
        "asset_id": asset["asset_id"],
        "review_state": asset["review_state"],
        "source_model_path": asset["model_path"],
        "source_model_sha256": metadata["source_model_sha256"],
        "source_repository": source_profile["repository"],
        "source_commit": source_profile["commit"],
        "assembly_configuration": asset["assembly_kind"],
        "mounts": asset.get("mounts", {}),
        "manifest_sha256": manifest_hash,
        "contact_sheet": str(output),
        "contact_sheet_sha256": sha256_file(output),
        "source_pngs": [{"path": path.name, "sha256": sha256_file(path), "label": label}
                        for path, label in panels],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--source-hashes", type=Path, default=HASHES_PATH)
    parser.add_argument("--compiled-signatures", type=Path, default=COMPILED_PATH)
    parser.add_argument("--asset-id", action="append", help="Limit to one or more validated asset IDs")
    parser.add_argument("--overwrite", action="store_true", help="Replace only this script's review_contact_sheet.png and index outputs")
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    source_hashes = json.loads(args.source_hashes.read_text(encoding="utf-8"))
    compiled = json.loads(args.compiled_signatures.read_text(encoding="utf-8"))
    manifest_hash = canonical_manifest_hash(registry, source_hashes, compiled)
    profiles = registry["source_profiles"]
    selected = set(args.asset_id or [])
    assets = [asset for asset in registry["assets"]
              if asset.get("review_state") == "VALIDATED_PENDING_HUMAN_REVIEW"
              and (not selected or asset["asset_id"] in selected)]
    if selected - {asset["asset_id"] for asset in assets}:
        raise ValueError(f"Requested IDs are not validated pending review: {sorted(selected - {asset['asset_id'] for asset in assets})}")
    if not selected and len(assets) != 9:
        raise ValueError(f"Expected nine validated review candidates, found {len(assets)}")

    records = []
    for asset in assets:
        source_profile = profiles[asset["source_profile"]]
        source_dir = args.evidence_root / asset["asset_id"]
        metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
        output = source_dir / CONTACT_SHEET
        records.append(draw_sheet(asset, source_profile, metadata, manifest_hash,
                                  source_dir, output, args.overwrite))

    if not selected:
        index_path = args.evidence_root / "REVIEW_INDEX.md"
        manifest_path = args.evidence_root / "review_sheets_manifest.json"
        if not args.overwrite and (index_path.exists() or manifest_path.exists()):
            raise FileExistsError("Refusing to overwrite existing REVIEW_INDEX.md or review_sheets_manifest.json")
        static_limit_pose_assets = [
            record["asset_id"] for record in records
            if any("driver_" in item["path"] and "_static.png" in item["path"]
                   for item in record["source_pngs"])
        ]
        limit_pose_note = (f" Static driver lower/upper-limit pose evidence exists only for {static_limit_pose_assets}; "
                           "those images are not commanded OPEN/CLOSED motion.") if static_limit_pose_assets else ""
        lines = [
            "# RobotSim Asset Review Index",
            "",
            "All nine entries remain `VALIDATED_PENDING_HUMAN_REVIEW`; none is approved by this packet. Historical Dex3 remains `REJECTED` and is intentionally excluded.",
            "",
            f"Registry manifest SHA-256: `{manifest_hash}`",
            "",
            "The sheets reuse the existing PNG review evidence. No candidate has captured commanded OPEN/CLOSED motion in the available PNG evidence." + limit_pose_note + " No new model renders or physics steps were run.",
            "",
            "| Asset ID | Source model | Assembly configuration | Source revision | Model SHA-256 | Contact sheet | Sheet SHA-256 |",
            "|---|---|---|---|---|---|---|",
        ]
        for record in records:
            rel = Path(record["contact_sheet"]).relative_to(args.evidence_root).as_posix()
            lines.append(
                f"| `{record['asset_id']}` | `{record['source_model_path']}` | {record['assembly_configuration']} | `{record['source_commit']}` | `{record['source_model_sha256']}` | [`{rel}`]({rel}) | `{record['contact_sheet_sha256']}` |"
            )
        lines += ["", "## Assembly mounts", ""]
        for record in records:
            lines.append(f"- `{record['asset_id']}`: `{json.dumps(record['mounts'], sort_keys=True, ensure_ascii=True)}`")
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest_path.write_text(json.dumps({
            "schema_version": 1,
            "registry_manifest_sha256": manifest_hash,
            "review_state_changes": [],
            "candidates": records,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"candidate_count": len(records), "registry_manifest_sha256": manifest_hash,
                      "sheets": [{"asset_id": item["asset_id"], "path": item["contact_sheet"],
                                  "sha256": item["contact_sheet_sha256"]} for item in records]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
