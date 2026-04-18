"""Register an on-disk model file into models/versions.json.

The HTTP upload flow (POST /upload-model) is the canonical way to introduce a
new model, but during local development it is friction to re-upload a 25 MB
.keras every time the container rebuilds. This script writes the metadata
entry directly so ModelManager.load_latest() picks the file up at startup.

Usage:
    python scripts/register_model.py models/best_model_phase2.keras v1_phase2 \
        --accuracy 0.991 --notes "Charles handover, MobileNetV2"
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def register(filepath: Path, version: str, accuracy: float | None, f1: float | None, notes: str) -> dict:
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    models_dir = filepath.parent
    versions_file = models_dir / "versions.json"

    entry = {
        "version": version,
        "filename": filepath.name,
        "filepath": str(filepath.resolve()),
        "extension": filepath.suffix,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {"accuracy": accuracy, "f1_score": f1},
        "notes": notes,
    }

    existing = json.loads(versions_file.read_text()) if versions_file.exists() else []
    existing.append(entry)
    versions_file.write_text(json.dumps(existing, indent=2))
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filepath", type=Path, help="Path to the model file (.keras or .pkl)")
    parser.add_argument("version", help="Version label, e.g. v1_phase2")
    parser.add_argument("--accuracy", type=float, default=None)
    parser.add_argument("--f1", type=float, default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    entry = register(args.filepath, args.version, args.accuracy, args.f1, args.notes)
    print(f"Registered {entry['version']} → {entry['filepath']}")


if __name__ == "__main__":
    main()
