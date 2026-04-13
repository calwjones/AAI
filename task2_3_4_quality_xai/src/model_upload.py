import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

METADATA_FILE = MODELS_DIR / "versions.json"


class ModelManager:
    def __init__(self):
        self.active_model = None
        self.active_version = None
        self.active_extension = None

    def load_latest(self):
        versions = self._read_metadata()
        if not versions:
            print("No models found — waiting for upload.")
            return
        latest = sorted(versions, key=lambda v: v["uploaded_at"])[-1]
        self._load_from_path(latest["filepath"], latest["version"], latest["extension"])

    def _load_from_path(self, filepath: str, version: str, extension: str):
        path = Path(filepath)
        if not path.exists():
            print(f"Model file not found: {filepath}")
            return

        try:
            if extension == ".pkl":
                with open(path, "rb") as f:
                    self.active_model = pickle.load(f)

            elif extension == ".keras":
                from tensorflow import keras
                self.active_model = keras.models.load_model(str(path))

            self.active_version = version
            self.active_extension = extension
            print(f"Model loaded: {version} ({extension})")

        except Exception as e:
            print(f"Failed to load model: {e}")
            self.active_model = None

    def save_and_load(self, model_bytes: bytes, filename: str, version: str, metrics: dict, notes: str) -> dict:
        ext = "." + filename.rsplit(".", 1)[-1]
        safe_version = version.replace(" ", "_").replace("/", "-")
        save_name = f"{safe_version}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}{ext}"
        save_path = MODELS_DIR / save_name

        with open(save_path, "wb") as f:
            f.write(model_bytes)

        metadata = {
            "version": version,
            "filename": save_name,
            "filepath": str(save_path),
            "extension": ext,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "notes": notes,
        }
        self._append_metadata(metadata)
        self._load_from_path(str(save_path), version, ext)

        return metadata

    def is_loaded(self) -> bool:
        return self.active_model is not None

    def list_versions(self) -> list:
        return self._read_metadata()

    def _read_metadata(self) -> list:
        if not METADATA_FILE.exists():
            return []
        with open(METADATA_FILE, "r") as f:
            return json.load(f)

    def _append_metadata(self, entry: dict):
        versions = self._read_metadata()
        versions.append(entry)
        with open(METADATA_FILE, "w") as f:
            json.dump(versions, f, indent=2)
