"""Quality scoring for the BRFN fruit/veg grader.

Classifier predicts fresh vs. rotten. Colour/Size/Ripeness scores are
derived from image features plus fresh-confidence, then mapped to A/B/C:

    A: Color >= 75, Size >= 80, Ripeness >= 70
    B: Color >= 65, Size >= 70, Ripeness >= 60
    C: otherwise
"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


IMG_SIZE = 224


def load_quality_model(model_path):
    return load_model(model_path)


def compute_color_score(img_array):
    img_uint8 = (img_array * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1].mean() / 255.0
    value = hsv[:, :, 2].mean() / 255.0
    brightness_penalty = 1.0 - abs(value - 0.55) * 0.5

    raw = saturation * 0.7 + brightness_penalty * 0.3
    return round(float(np.clip(raw * 120, 0, 100)), 1)


def compute_size_score(img_array):
    img_uint8 = (img_array * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_ratio = np.count_nonzero(binary) / binary.size

    if fg_ratio < 0.2:
        score = fg_ratio / 0.2 * 60
    elif fg_ratio > 0.85:
        score = 85 - (fg_ratio - 0.85) * 100
    else:
        score = 70 + (fg_ratio - 0.2) / 0.5 * 30

    return round(float(np.clip(score, 0, 100)), 1)


def compute_ripeness_score(img_array, fresh_confidence):
    r_mean = img_array[:, :, 0].mean()
    g_mean = img_array[:, :, 1].mean()
    b_mean = img_array[:, :, 2].mean()

    warmth = (r_mean + g_mean * 0.5) / (b_mean + 0.01)
    warmth_norm = np.clip(warmth / 4.0, 0, 1)

    raw = fresh_confidence * 0.6 + warmth_norm * 0.4
    return round(float(np.clip(raw * 100, 0, 100)), 1)


def assign_grade(color_score, size_score, ripeness_score):
    if color_score >= 75 and size_score >= 80 and ripeness_score >= 70:
        return 'A'
    if color_score >= 65 and size_score >= 70 and ripeness_score >= 60:
        return 'B'
    return 'C'


def _grade_array(img_array, model):
    rotten_prob = float(model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0][0])
    fresh_confidence = 1.0 - rotten_prob

    color = compute_color_score(img_array)
    size = compute_size_score(img_array)
    ripeness = compute_ripeness_score(img_array, fresh_confidence)

    return {
        'prediction': 'Fresh' if fresh_confidence > 0.5 else 'Rotten',
        'confidence': round(max(fresh_confidence, rotten_prob), 4),
        'color_score': color,
        'size_score': size,
        'ripeness_score': ripeness,
        'grade': assign_grade(color, size, ripeness),
    }


def grade_image(filepath, model):
    img = load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
    return _grade_array(img_to_array(img) / 255.0, model)


def grade_image_bytes(image_bytes, model):
    img_np = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return _grade_array(img_resized.astype(np.float32) / 255.0, model)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print('Usage: python quality_scorer.py <model_path> <image_path>')
        sys.exit(1)

    model = load_quality_model(sys.argv[1])
    result = grade_image(sys.argv[2], model)
    for k, v in result.items():
        print(f'  {k}: {v}')
