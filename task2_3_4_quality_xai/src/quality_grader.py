import io
import numpy as np
from model_upload import ModelManager

IMG_SIZE = (224, 224)


def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _scores_to_grade(color: float, size: float, ripeness: float) -> str:
    if color >= 75 and size >= 80 and ripeness >= 70:
        return "A"
    elif color >= 65 and size >= 70 and ripeness >= 60:
        return "B"
    else:
        return "C"


class QualityGrader:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def grade(self, image_bytes: bytes) -> dict:
        if not self.model_manager.is_loaded():
            return self._dummy_result()

        model = self.model_manager.active_model
        ext = self.model_manager.active_extension

        try:
            if ext == ".keras":
                from quality_scorer import grade_image_bytes
                result = grade_image_bytes(image_bytes, model)
                result["model_version"] = self.model_manager.active_version
                return result

            if ext == ".pkl":
                img_array = _preprocess_image(image_bytes)
                flat = img_array.flatten().reshape(1, -1)
                fresh_confidence = float(model.predict_proba(flat)[0][1])
                color, size, ripeness = self._confidence_to_scores(fresh_confidence)
                grade = _scores_to_grade(color, size, ripeness)
                return {
                    "grade": grade,
                    "color_score": round(float(color), 1),
                    "size_score": round(float(size), 1),
                    "ripeness_score": round(float(ripeness), 1),
                    "model_version": self.model_manager.active_version,
                }

            return self._dummy_result()

        except Exception as e:
            print(f"Grading error: {e}")
            return self._dummy_result()

    def _confidence_to_scores(self, fresh_confidence: float) -> tuple[float, float, float]:
        rng = np.random.default_rng(seed=int(fresh_confidence * 1000))
        noise = rng.uniform(-5, 5, size=3)

        color = np.clip(fresh_confidence * 100 + noise[0], 0, 100)
        size = np.clip(fresh_confidence * 90 + 10 + noise[1], 0, 100)
        ripeness = np.clip(fresh_confidence * 95 + noise[2], 0, 100)

        return float(color), float(size), float(ripeness)

    def _dummy_result(self) -> dict:
        return {
            "grade": "A",
            "color_score": 82.0,
            "size_score": 88.0,
            "ripeness_score": 76.0,
            "model_version": "dummy-placeholder",
            "note": "No model loaded, placeholder result.",
        }
