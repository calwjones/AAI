from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import model_upload  # noqa: E402

_tmp_models = Path(tempfile.mkdtemp(prefix="brfn-test-models-"))
model_upload.MODELS_DIR = _tmp_models
model_upload.METADATA_FILE = _tmp_models / "versions.json"
