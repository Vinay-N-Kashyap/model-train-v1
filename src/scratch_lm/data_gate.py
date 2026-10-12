"""Validate a local text corpus and its provenance before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class DataGateError(ValueError):
    """Raised when a corpus is missing required provenance or split details."""


REQUIRED_TEXT_FIELDS = (
    "dataset_name",
    "source",
    "version",
    "license",
)
REQUIRED_SPLITS = ("train", "validation", "test")


def validate_corpus(
    corpus_path: Path,
    manifest_path: Path | None = None,
    *,
    block_size: int = 256,
) -> tuple[bytes, dict[str, Any]]:
    """Return corpus bytes and manifest only when basic use and split gates pass."""
    corpus_path = Path(corpus_path)
    manifest_path = Path(manifest_path) if manifest_path else corpus_path.with_name(
        "corpus.manifest.json"
    )

    if not corpus_path.is_file():
        raise DataGateError(f"Corpus file does not exist: {corpus_path}")
    if not manifest_path.is_file():
        raise DataGateError(f"Required provenance manifest is missing: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataGateError(f"Cannot read a valid UTF-8 JSON manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DataGateError("Manifest root must be a JSON object")

    missing = [
        field
        for field in REQUIRED_TEXT_FIELDS
        if not isinstance(manifest.get(field), str) or not manifest[field].strip()
    ]
    if missing:
        raise DataGateError(f"Manifest is missing required fields: {', '.join(missing)}")
    placeholders = [
        field
        for field in REQUIRED_TEXT_FIELDS
        if manifest[field].strip().lower().startswith("replace_with_")
        or manifest[field].strip().lower() in {"unknown", "tbd", "placeholder"}
    ]
    if placeholders:
        raise DataGateError(f"Replace manifest placeholders: {', '.join(placeholders)}")
    if manifest["rights_reviewed_by_owner"] is not True:
        raise DataGateError("Project owner must explicitly confirm the dataset rights review")

    permitted_uses = manifest.get("permitted_uses")
    if not isinstance(permitted_uses, list) or not any(
        isinstance(use, str) and use.strip().lower() == "model_training"
        for use in permitted_uses
    ):
        raise DataGateError("Manifest must explicitly allow 'model_training'")

    personal_data = manifest.get("personal_data")
    if not isinstance(personal_data, str) or personal_data not in {
        "none",
        "removed",
        "consented",
    }:
        raise DataGateError(
            "personal_data must be 'none', 'removed', or 'consented' after review"
        )
    if personal_data == "consented":
        for field in ("consent_or_legal_basis", "retention_policy"):
            if not isinstance(manifest.get(field), str) or not manifest[field].strip():
                raise DataGateError(
                    f"Manifest must document {field} when personal data is retained"
                )

    split_plan = manifest.get("split_plan")
    if not isinstance(split_plan, dict):
        raise DataGateError("Manifest split_plan must be an object")
    fractions: list[float] = []
    for split in REQUIRED_SPLITS:
        fraction = split_plan.get(split)
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise DataGateError(f"split_plan.{split} must be a number")
        if not 0.0 < fraction < 1.0:
            raise DataGateError(f"split_plan.{split} must be greater than 0 and less than 1")
        fractions.append(float(fraction))
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise DataGateError("Train, validation, and test fractions must add up to 1")
    if split_plan.get("method") != "sequential_bytes":
        raise DataGateError("split_plan.method must be 'sequential_bytes' for this baseline")

    raw = corpus_path.read_bytes()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataGateError("Corpus must be valid UTF-8 text") from exc

    # Ensure every partition can supply at least one input/target training window.
    boundaries = [int(len(raw) * fractions[0]), int(len(raw) * (fractions[0] + fractions[1]))]
    lengths = (boundaries[0], boundaries[1] - boundaries[0], len(raw) - boundaries[1])
    if any(length <= block_size + 1 for length in lengths):
        raise DataGateError(
            f"Each split must contain more than {block_size + 1} bytes; "
            f"split sizes are {lengths}"
        )
    return raw, manifest


def split_corpus(raw: bytes, split_plan: dict[str, Any]) -> tuple[bytes, bytes, bytes]:
    """Split bytes in the declared train/validation/test order."""
    train_end = int(len(raw) * float(split_plan["train"]))
    validation_end = int(len(raw) * (float(split_plan["train"]) + float(split_plan["validation"])))
    return raw[:train_end], raw[train_end:validation_end], raw[validation_end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/corpus.txt"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()
    try:
        raw, manifest = validate_corpus(args.data, args.manifest, block_size=args.block_size)
    except DataGateError as exc:
        raise SystemExit(f"Data gate: FAIL - {exc}") from exc
    print(
        "Data gate: PASS - "
        f"dataset={manifest['dataset_name']!r}, bytes={len(raw)}, "
        f"splits={manifest['split_plan']['train']}/"
        f"{manifest['split_plan']['validation']}/"
        f"{manifest['split_plan']['test']}"
    )


if __name__ == "__main__":
    main()
