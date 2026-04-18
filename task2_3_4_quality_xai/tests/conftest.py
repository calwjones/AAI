"""Redirect ModelManager's on-disk state to a temp dir so tests don't pollute
the real models/ directory (which holds the production .keras file)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Patch before model_upload is imported anywhere
import model_upload  # noqa: E402

_tmp_models = Path(tempfile.mkdtemp(prefix="brfn-test-models-"))
model_upload.MODELS_DIR = _tmp_models
model_upload.METADATA_FILE = _tmp_models / "versions.json"
